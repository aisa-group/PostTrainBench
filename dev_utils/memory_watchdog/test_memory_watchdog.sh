#!/bin/bash
# Regression test of src/utils/memory_watchdog.py in a real HTCondor job (request_memory 8 GB, see
# test_memory_watchdog.sub), with the agent sandbox started like src/run_task.sh does (apptainer -c --cleanenv --pid --no-init --writable-tmpfs).
#   A: a parent holds 3 GB and forks 6 readers (per-process RSS adds up to ~21 GB, the job uses ~3 GB): no kill.
#   B: one of 4 worker processes allocates without end (~1 GB/s): only that worker is killed (exit -9), the others and
#      the parent finish, and the job is not held.
# Prints PASS/FAIL per check and exits 1 on any failure. Run from the repo root: condor_submit_bid 35 dev_utils/memory_watchdog/test_memory_watchdog.sub
set -uo pipefail
SIF="$(grep '^POST_TRAIN_BENCH_CONTAINERS_DIR=' .env | cut -d= -f2- | tr -d '"')/standard.sif"
OUT=$(mktemp -d)
FAIL=0
check() { if [ "$1" = 0 ]; then echo "PASS: $2"; else echo "FAIL: $2"; FAIL=1; fi; }
sandbox() { apptainer exec -c --cleanenv --pid --no-init --writable-tmpfs "$SIF" python3 -c "$1"; }

python3 src/utils/memory_watchdog.py --check || exit 1
python3 src/utils/memory_watchdog.py --log "$OUT/watchdog.log" --fraction 0.8 &
WATCHDOG=$!
sleep 2

echo "== A: shared memory, no kill"
sandbox '
import os, time
buf = bytearray(3 * 1024**3)          # 3 GB, touched
for i in range(0, len(buf), 4096): buf[i] = 1
kids = []
for _ in range(6):
    pid = os.fork()
    if pid == 0:
        s = sum(buf[i] for i in range(0, len(buf), 4096))   # read only: pages stay shared
        time.sleep(15); os._exit(0)
    kids.append(pid)
codes = [os.waitpid(k, 0)[1] for k in kids]
print("A exit codes:", codes)
'
check $? "A: sandbox exited 0"
[ ! -s "$OUT/watchdog.log" ] || ! grep -q killed "$OUT/watchdog.log"; check $? "A: no kill logged"

echo "== B: one runaway worker"
sandbox '
import multiprocessing as mp, time
def worker(i):
    if i == 2:
        chunks = []
        while True:                   # ~1 GB/s without end
            chunks.append(bytearray(100 * 1024**2)); time.sleep(0.1)
    time.sleep(40)
procs = [mp.Process(target=worker, args=(i,)) for i in range(4)]
for p in procs: p.start()
for p in procs: p.join()
print("B exit codes:", [p.exitcode for p in procs])
assert [p.exitcode for p in procs] == [0, 0, -9, 0], "unexpected exit codes"
print("B: parent survived")
'
check $? "B: sandbox exited 0 with only the runaway worker killed"
[ "$(grep -c killed "$OUT/watchdog.log")" = 1 ]; check $? "B: exactly one kill logged"
cat "$OUT/watchdog.log"

kill $WATCHDOG
exit $FAIL
