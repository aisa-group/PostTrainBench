#!/usr/bin/env python3
"""Shard a staged HuggingFace cache into Kaggle-uploadable datasets.

Upstream serves the cache as one 160 GB directory overlay-mounted into the
sandbox (`run_task.sh:180-195`). Kaggle serves it as read-only *datasets*, and
a dataset is capped at **1,000 files** without `ExtraDatasetsQuota`
(`Kaggle.Sdk/datasets/models/DatasetConstants.cs:42`). The cache holds ~396 HF
repos, which unpacks well past that, so it ships as several datasets and
`template/environment/preflight.sh` merges them back into one tree.

Merging is free because every shard keeps the cache's own top-level layout —
`hub/` for models, `datasets/` for dataset builders, plus whatever else the
hub client writes (recent `huggingface_hub` also creates `xet/`). preflight
mirrors each top-level directory and symlinks its children, so shards land
side by side with no dedup and no ordering constraint. A repo is never split
across shards, which is what makes that true.

Shards are built with **hard links**, so this costs no extra disk and cannot
corrupt the source. Both trees must therefore be on one filesystem.

    # 1. stage the cache (upstream's own downloader, unmodified)
    HF_HOME=/path/to/cache python containers/download_hf_cache/download_resources.py

    # 2. shard it
    python src/kaggle_harbor/stage_hf_cache.py \\
        --cache /path/to/cache --output /path/to/shards --owner <kaggle-user>

    # 3. upload each shard, then push tasks with the printed --hf-cache value
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Margin under DatasetConstants.cs:42's 1,000. Leaves room for the
# dataset-metadata.json Kaggle adds and for a repo growing between the count
# and the upload.
DEFAULT_MAX_FILES = 800

# Not a platform limit — a practical one. A failed 160 GB upload has to start
# over, so keep each piece small enough to retry cheaply.
DEFAULT_MAX_GB = 45

# Written by the hub client during transfer, not read back by transformers or
# datasets at load time. Excluded by default: it is pure download scratch and
# on a Xet-backed pull it is both large and file-dense, which is exactly what
# the 1,000-file cap punishes.
DEFAULT_EXCLUDE_TOPLEVEL = ("xet",)

# Model families, in the order shards are emitted. A repo whose name matches
# none of these is training data and gets bin-packed into the `data` shards.
# Grouping by family is not cosmetic: models are byte-bound and datasets are
# file-bound, so mixing them makes both constraints bind in every shard.
FAMILIES = (
    ("qwen3",    lambda n: n.startswith("Qwen3-")),
    ("qwen25",   lambda n: n.startswith("Qwen2.5-")),
    ("deepseek", lambda n: n.lower().startswith("deepseek")),
    ("gemma",    lambda n: n.startswith("gemma-")),
    ("smollm3",  lambda n: n.startswith("SmolLM3-")),
)


def family_of(repo: dict) -> str | None:
    """Model family for a hub model repo, else None (= training data)."""
    if repo["top"] != "hub" or not repo["name"].startswith("models--"):
        return None
    bare = repo["name"].removeprefix("models--").split("--")[-1]
    for name, match in FAMILIES:
        if match(bare):
            return name
    return None


def measure(repo: Path) -> tuple[int, int, int]:
    """(file count, bytes on disk, symlink count) for one repo directory.

    **`lstat`, not `stat`, and that is the whole point.** The hub cache stores
    each file once under `blobs/<sha>` and points at it from
    `snapshots/<rev>/<name>` with a *relative symlink*. `stat()` follows those
    links, so every weight file gets counted twice and the cache measures
    ~2x its real size — 369 GB instead of 183 GB on this one. `lstat()` counts
    the link as the few bytes it is, which is what actually has to be uploaded.
    """
    files = 0
    size = 0
    links = 0
    for dirpath, _dirnames, filenames in os.walk(repo):
        for name in filenames:
            p = Path(dirpath) / name
            files += 1
            try:
                st = p.lstat()
            except OSError:
                continue
            size += st.st_size
            if p.is_symlink():
                links += 1
    return files, size, links


def collect(cache: Path, exclude: tuple[str, ...]) -> list[dict]:
    """Every repo in the cache, as {top, name, files, bytes}.

    A "repo" is a child of a top-level directory — `hub/models--org--name`,
    `datasets/owner___name`. That is the granularity preflight.sh links at, so
    it is the granularity a shard has to preserve.
    """
    repos = []
    for top in sorted(cache.iterdir()):
        if not top.is_dir() or top.name in exclude:
            continue
        for repo in sorted(top.iterdir()):
            # The datasets library drops sibling .lock FILES next to the repo
            # directories (e.g. _home_kaggle_..._<hash>.lock). They are not
            # repos, they must not be shipped, and copytree dies on them.
            if not repo.is_dir():
                continue
            if repo.name.startswith("."):
                # .locks / .no_exist are hub bookkeeping; preflight.sh
                # deliberately does not link them and HF rebuilds them.
                continue
            files, size, links = measure(repo)
            repos.append(
                {"top": top.name, "name": repo.name, "path": repo,
                 "files": files, "bytes": size, "links": links}
            )
    return repos


def pack(repos: list[dict], max_files: int, max_bytes: int) -> list[list[dict]]:
    """First-fit-decreasing by file count, with a byte ceiling as well.

    Descending order matters: a single repo bigger than the limit would
    otherwise be discovered only after the shards had filled around it.
    """
    shards: list[list[dict]] = []
    totals: list[tuple[int, int]] = []
    for repo in sorted(repos, key=lambda r: (-r["files"], -r["bytes"])):
        if repo["files"] > max_files:
            raise SystemExit(
                f"{repo['top']}/{repo['name']} alone has {repo['files']} files, "
                f"over the {max_files} per-shard limit. A repo cannot be split "
                f"without breaking the merge, so raise --max-files (the hard "
                f"cap is 1000 without ExtraDatasetsQuota) or request the quota."
            )
        for i, (nf, nb) in enumerate(totals):
            if nf + repo["files"] <= max_files and nb + repo["bytes"] <= max_bytes:
                shards[i].append(repo)
                totals[i] = (nf + repo["files"], nb + repo["bytes"])
                break
        else:
            shards.append([repo])
            totals.append((repo["files"], repo["bytes"]))
    return shards


def _flatten_hub_repo(repo: Path) -> None:
    """Replace snapshots/ symlinks with hard links to the blob, then drop blobs/.

    The Kaggle CLI zips with `shutil.make_archive`
    (`kaggle_api_extended.py:411`), i.e. `ZipFile.write()`, which **follows
    symlinks** — measured: a 100 KB blob plus one link to it uploads as 200 KB.
    Left alone, the hub cache's `snapshots/<rev>/x -> ../../blobs/<sha>` layout
    would double a 195 GB cache to ~370 GB.

    Resolving the link and deleting `blobs/` leaves every file exactly once, at
    the path `from_pretrained` actually resolves. Verified against the real
    Qwen3-1.7B-Base repo with HF_HUB_OFFLINE=1: config and tokenizer load from
    a blobs-less cache, size unchanged.

    Hard links again, so this still costs no disk — the blob is not copied,
    just given a second name before the first is removed.
    """
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return
    for link in snapshots.rglob("*"):
        if not link.is_symlink():
            continue
        target = link.resolve()
        if not target.is_file():
            continue  # already-broken link; leave it for the caller to notice
        link.unlink()
        os.link(target, link)
    shutil.rmtree(repo / "blobs", ignore_errors=True)


def group(repos: list[dict], max_bytes: int) -> list[tuple[str, list[dict]]]:
    """One shard per model family; training data bin-packed into `data-NN`.

    Returns (suffix, repos) so shards are named for what they hold — a failure
    reading "gemma shard did not mount" beats "shard 04 did not mount".
    """
    out: list[tuple[str, list[dict]]] = []
    for fam, _ in FAMILIES:
        sel = [r for r in repos if family_of(r) == fam]
        if sel:
            out.append((fam, sel))
    data = [r for r in repos if family_of(r) is None]
    for i, chunk in enumerate(pack(data, DEFAULT_MAX_FILES, max_bytes), start=1):
        out.append((f"data-{i:02d}", chunk))
    return out


def build(shards: list[tuple[str, list[dict]]], output: Path, owner: str,
          slug: str) -> list[str]:
    """Hard-link each shard into place and write its dataset metadata."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    ids = []
    for suffix, shard in shards:
        name = f"{slug}-{suffix}"
        root = output / name
        for repo in shard:
            dst = root / repo["top"] / repo["name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copy_function=os.link -> hard links, so this costs no disk and
            # the source cannot be modified through the copy. symlinks=True
            # keeps snapshots/ as links for now; _flatten_hub_repo resolves
            # them immediately after.
            shutil.copytree(repo["path"], dst, copy_function=os.link,
                            symlinks=True)
            _flatten_hub_repo(dst)
        # DecompressFileActivity.cs:44-77 — a zip that is the ONLY file in a
        # dataset extracts to the ROOT, with no wrapper folder. A shard holding
        # just hub/ would therefore land its repos at <mount>/models--X instead
        # of <mount>/hub/models--X, and preflight.sh would mirror the wrong
        # level. A second top-level entry keeps the wrapper.
        ph = root / "_placeholder"
        ph.mkdir(parents=True, exist_ok=True)
        (ph / "README.txt").write_text(
            "Forces Kaggle to preserve this dataset's directory structure on\n"
            "extract. A lone archive would otherwise unwrap to the root and\n"
            "break the hub/ + datasets/ layout preflight.sh expects.\n"
        )
        (root / "dataset-metadata.json").write_text(json.dumps({
            "title": f"PostTrainBench HF cache: {suffix}",
            "id": f"{owner}/{name}",
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2) + "\n")
        ids.append(f"{owner}/{name}")
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True,
                   help="The staged HF_HOME to shard.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to build the shard trees. Must be on the same "
                        "filesystem as --cache (hard links).")
    p.add_argument("--owner", required=True, help="Kaggle username.")
    p.add_argument("--slug", default="ptb-hf-cache",
                   help="Dataset slug stem; shards get -01, -02, ...")
    p.add_argument("--mount-prefix", default="/mnt/hf-cache",
                   help="Container path stem; shards mount at <prefix>-01, ...")
    p.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    p.add_argument("--exclude-toplevel", default=",".join(DEFAULT_EXCLUDE_TOPLEVEL),
                   help="Comma-separated top-level dirs to leave out.")
    p.add_argument("--dry-run", action="store_true",
                   help="Measure and plan; build nothing.")
    args = p.parse_args()

    exclude = tuple(x.strip() for x in args.exclude_toplevel.split(",") if x.strip())
    repos = collect(args.cache, exclude)
    if not repos:
        raise SystemExit(f"no repos found under {args.cache}")

    total_files = sum(r["files"] for r in repos)
    total_bytes = sum(r["bytes"] for r in repos)
    total_links = sum(r["links"] for r in repos)
    print(f"{len(repos)} repos, {total_files} files "
          f"({total_links} of them symlinks), "
          f"{total_bytes / 1e9:.1f} GB on disk (excluding {exclude or 'nothing'})")
    if total_links:
        print("  NOTE: the hub cache is symlink-based (snapshots -> blobs). "
              "Those links must survive the Kaggle round-trip or the mounted "
              "cache is unusable; dereferencing them would roughly double the "
              "upload.")
    for top in sorted({r["top"] for r in repos}):
        sel = [r for r in repos if r["top"] == top]
        print(f"  {top}/: {len(sel)} repos, {sum(r['files'] for r in sel)} files, "
              f"{sum(r['bytes'] for r in sel) / 1e9:.1f} GB")

    shards = group(repos, int(args.max_gb * 1e9))
    print(f"\n{len(shards)} shard(s):")
    for suffix, shard in shards:
        print(f"  {suffix:10s}: {len(shard):4d} repos, "
              f"{sum(r['files'] for r in shard):5d} files, "
              f"{sum(r['bytes'] for r in shard) / 1e9:6.1f} GB")

    mounts = [f"{args.mount_prefix}-{suffix}" for suffix, _ in shards]
    if args.dry_run:
        print("\nDRY RUN — nothing built.")
    else:
        ids = build(shards, args.output, args.owner, args.slug)
        print(f"\nbuilt {len(ids)} shard tree(s) under {args.output} (hard links)")
        print("\nupload each, in order:")
        for did in ids:
            print(f"  kaggle datasets create -p {args.output}/{did.split('/')[1]} -r zip")

    print("\n--hf-cache value for build_tasks.py:")
    print(f"  {','.join(mounts)}")
    print("\nmounts[] for the tasks/push request (fill in each version):")
    print(json.dumps(
        [{"datasetVersionSlug": f"{args.owner}/{args.slug}-{sfx}/versions/1",
          "mountPath": m} for (sfx, _), m in zip(shards, mounts)],
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
