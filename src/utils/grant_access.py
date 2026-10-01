#!/usr/bin/env python3
"""Gives the users in POST_TRAIN_BENCH_USERS_ACCESS (colon-separated, e.g. "brank:hbhatnagar") read/write access to
result directories, through POSIX ACLs: rwx on directories, rw on files (plus x where the owner has it), and a default
ACL on every directory so files created there later inherit it.

A plain `setfacl -R -m u:<user>:rwX` is not enough. Some files are written with mode 600 (e.g. safetensors weights,
written through a temp file), and their ACL mask is then ---, which cancels every named user. Raising the mask would
also give the owning group access it never had (group::r-x from a default ACL becomes effective). So this rewrites each
ACL instead: every existing entry keeps exactly its current effective permissions, the users are added, and the mask
is recomputed.

Only entries owned by the calling user can be changed; others are skipped and counted (their owner must run this).
Symlinks are not followed. Unset or empty POST_TRAIN_BENCH_USERS_ACCESS grants nothing; an unknown user is an error.

Usage:
  python3 src/utils/grant_access.py <path>...                   users from POST_TRAIN_BENCH_USERS_ACCESS
  python3 src/utils/grant_access.py --users a:b <path>...
src/run_task.sh runs it on the result dir when it exits; so do the scripts that add files to existing result dirs.
"""
from __future__ import annotations

import argparse
import os
import pwd
import stat
import subprocess
import sys

ENV_VAR = "POST_TRAIN_BENCH_USERS_ACCESS"
CHUNK = 500  # paths per getfacl / setfacl call
GROUP_CLASS = ("user:", "group:")  # named users, named groups and group:: (masked entries); user:: is not


def parse_users(spec: str) -> list[str]:
    users = [u for u in spec.split(":") if u]
    for user in users:
        try:
            pwd.getpwnam(user)
        except KeyError:
            raise ValueError(f"{ENV_VAR}: unknown user {user!r}") from None
    return users


def perms_and(a: str, b: str) -> str:
    return "".join(x if x == y and x != "-" else "-" for x, y in zip(a, b))


def perms_or(a: str, b: str) -> str:
    return "".join(x if x != "-" else y for x, y in zip(a, b))


def new_acl(entries: list[tuple[str, str]], is_dir: bool, uids: list[str]) -> list[tuple[str, str]]:
    """The rewritten access (or default) ACL entries [(tag, perms)] for one inode; see the module docstring."""
    acl = dict(entries)
    for base in ("user::", "group::", "other::"):
        if base not in acl:
            raise ValueError(f"ACL without {base}: {entries}")
    mask = acl.pop("mask::", None)
    out = {}
    for tag, perms in acl.items():
        masked = mask is not None and tag != "user::" and tag.startswith(GROUP_CLASS)
        out[tag] = perms_and(perms, mask) if masked else perms
    want = "rwx" if is_dir else "rw" + ("x" if "x" in out["user::"] else "-")
    for uid in uids:
        tag = f"user:{uid}:"
        out[tag] = perms_or(out.get(tag, "---"), want)
    group_class = [p for tag, p in out.items() if tag != "user::" and tag.startswith(GROUP_CLASS)]
    new_mask = "---"
    for perms in group_class:
        new_mask = perms_or(new_mask, perms)
    out["mask::"] = new_mask
    return sorted(out.items(), key=lambda kv: (entry_rank(kv[0]), kv[0]))


def entry_rank(tag: str) -> int:
    """getfacl's entry order: user::, user:<id>:, group::, group:<id>:, mask::, other::."""
    ranks = {"user::": 0, "group::": 2, "mask::": 4, "other::": 5}
    if tag in ranks:
        return ranks[tag]
    return {"user:": 1, "group:": 3}[tag[:tag.index(":") + 1]]


def rewrite_block(block: list[str], is_dir: bool, uids: list[str]) -> list[str]:
    """Rewrites one `getfacl -n -E` block (header comments + entries) into the input of `setfacl --restore`."""
    header = [line for line in block if line.startswith("# file: ") or line.startswith("# flags: ")]
    if not header or not header[0].startswith("# file: "):
        raise ValueError(f"unexpected getfacl block: {block}")
    access, default = [], []
    for line in block:
        if line.startswith("#"):
            continue
        tag, perms = line.rsplit(":", 1)
        tag += ":"
        if len(perms) != 3:
            raise ValueError(f"unexpected getfacl line: {line!r}")
        if tag.startswith("default:"):
            default.append((tag[len("default:"):], perms))
        else:
            access.append((tag, perms))
    acl = new_acl(access, is_dir, uids)
    lines = header + [f"{tag}{perms}" for tag, perms in acl]
    if is_dir:
        if not default:
            # New default ACL: new files keep the directory's owner/group/other permissions.
            base = dict(acl)
            default = [("user::", base["user::"]), ("group::", base["group::"]), ("other::", base["other::"])]
        lines += [f"default:{tag}{perms}" for tag, perms in new_acl(default, True, uids)]
    return lines + [""]


def collect(paths: list[str]) -> tuple[list[tuple[str, bool]], int]:
    """([(path, is_dir)] of the inodes under paths owned by the caller, number of skipped inodes of other owners)."""
    me = os.getuid()
    own, skipped = [], 0

    def visit(path: str) -> None:
        nonlocal skipped
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return
        if st.st_uid != me:
            skipped += 1
        else:
            own.append((path, stat.S_ISDIR(st.st_mode)))

    for root in paths:
        if not os.path.lexists(root):
            raise FileNotFoundError(root)
        visit(root)
        if os.path.isdir(root) and not os.path.islink(root):
            for dirpath, dirnames, filenames in os.walk(root):
                for name in dirnames + filenames:
                    visit(os.path.join(dirpath, name))
    return own, skipped


def grant(paths: list[str], users: list[str]) -> None:
    me = pwd.getpwuid(os.getuid()).pw_name
    uids = [str(pwd.getpwnam(u).pw_uid) for u in users if u != me]
    own, skipped = collect(paths)
    for i in range(0, len(own), CHUNK):
        chunk = own[i:i + CHUNK]
        out = subprocess.run(["getfacl", "--absolute-names", "--physical", "-n", "-E", "--"] + [p for p, _ in chunk],
                             check=True, capture_output=True, text=True).stdout
        blocks = [b.splitlines() for b in out.strip("\n").split("\n\n")]
        if len(blocks) != len(chunk):
            raise RuntimeError(f"getfacl returned {len(blocks)} blocks for {len(chunk)} paths")
        restore = []
        for (path, is_dir), block in zip(chunk, blocks):
            restore += rewrite_block(block, is_dir, uids)
        subprocess.run(["setfacl", "--restore=-"], input="\n".join(restore) + "\n", check=True, text=True)
    print(f"grant_access: gave {','.join(users)} access to {len(own)} entries under {' '.join(paths)}"
          + (f"; skipped {skipped} entries owned by other users" if skipped else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", help=f"colon-separated user names (default: ${ENV_VAR})")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    spec = args.users if args.users is not None else os.environ.get(ENV_VAR, "")
    users = parse_users(spec)
    if not users:
        print(f"grant_access: {ENV_VAR} is not set; no access granted", file=sys.stderr)
        return
    grant(args.paths, users)


if __name__ == "__main__":
    main()
