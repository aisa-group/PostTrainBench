#!/usr/bin/env python3
"""Regression test for src/utils/grant_access.py, on a real ACL-capable filesystem (use /fast, where the results live).

Builds a small tree like a result dir: a directory whose default ACL already names another user, a mode-600 file
created there (so its mask is --- and the named user has no access, as with Hardik's safetensors weights), a plain 644
file, an executable, a file naming a second user, a symlink out of the tree. After grant(), the target user has rwx on
every directory and rw (rwx for the executable) on every file; the owning group, other, and every other named entry
keep exactly their previous effective permissions; the symlink's target is untouched; a second run changes nothing;
a file created afterwards inherits access.

Usage: python3 dev_utils/grant_access/test_grant_access.py <base dir on /fast> <target user> <second user>
"""
from __future__ import annotations

import os
import pwd
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "utils"))
from grant_access import grant  # noqa: E402


def effective(path: str) -> dict[str, str]:
    """{tag: effective perms} of path's access ACL (named entries masked), from getfacl."""
    out = subprocess.run(["getfacl", "--absolute-names", "--physical", "-n", "-E", "--", path],
                         check=True, capture_output=True, text=True).stdout
    acl = {}
    for line in out.splitlines():
        if line and not line.startswith("#") and not line.startswith("default:"):
            tag, perms = line.rsplit(":", 1)
            acl[tag + ":"] = perms
    mask = acl.pop("mask::", None)
    if mask is None:
        return acl
    return {t: p if t in ("user::", "other::") else "".join(a if a == b else "-" for a, b in zip(p, mask))
            for t, p in acl.items()}


def main() -> None:
    base, target, second = sys.argv[1], sys.argv[2], sys.argv[3]
    target_uid, second_uid = pwd.getpwnam(target).pw_uid, pwd.getpwnam(second).pw_uid
    tree = tempfile.mkdtemp(prefix="grant_access_test_", dir=base)
    outside = os.path.join(tree + "_outside")
    open(outside, "w").close()
    os.chmod(outside, 0o600)

    sub = os.path.join(tree, "run", "final_model")
    os.makedirs(sub)
    subprocess.run(["setfacl", "-R", "-m", f"u:{second}:rwx", "-m", f"d:u:{second}:rwx", tree], check=True)
    weights = os.path.join(sub, "model.safetensors")
    fd = os.open(weights, os.O_CREAT | os.O_WRONLY, 0o600)  # like a temp file renamed into place
    os.close(fd)
    plain = os.path.join(tree, "run", "metrics.json")
    open(plain, "w").close()
    os.chmod(plain, 0o644)
    script = os.path.join(tree, "run", "solve.sh")
    open(script, "w").close()
    os.chmod(script, 0o750)
    os.symlink(outside, os.path.join(tree, "run", "link"))
    inodes = [tree, os.path.join(tree, "run"), sub, weights, plain, script]

    assert effective(weights).get(f"user:{second_uid}:") == "---", effective(weights)
    before = {p: effective(p) for p in inodes + [outside]}

    grant([tree], [target, pwd.getpwuid(os.getuid()).pw_name])
    after = {p: effective(p) for p in inodes + [outside]}
    for path in inodes:
        want = "rwx" if os.path.isdir(path) else ("rwx" if path == script else "rw-")
        assert after[path][f"user:{target_uid}:"] == want, (path, after[path])
        for tag, perms in before[path].items():
            assert after[path][tag] == perms, (path, tag, before[path], after[path])
        assert set(after[path]) == set(before[path]) | {f"user:{target_uid}:"}, (path, after[path])
    assert after[outside] == before[outside], (before[outside], after[outside])

    grant([tree], [target])
    assert {p: effective(p) for p in inodes} == {p: after[p] for p in inodes}, "second run changed ACLs"

    later = os.path.join(sub, "config.json")
    open(later, "w").close()
    assert effective(later)[f"user:{target_uid}:"].startswith("rw"), effective(later)

    subprocess.run(["rm", "-rf", tree, outside], check=True)
    print("grant_access: all tests passed")


if __name__ == "__main__":
    main()
