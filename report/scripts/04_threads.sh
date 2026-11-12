#!/bin/bash
# pydecontx with 8 threads on each task, twice, pinned to 8 idle cores.
# Usage: bash 04_threads.sh <tasks_dir> <runs_dir> <python>
HERE=$(cd "$(dirname "$0")" && pwd)
for t in "$1"/*/; do
  for r in 1 2; do
    cores=$("$3" - <<'PY'
import os, time
def snap():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = list(map(int, f[1:])); d[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return d
a = snap(); time.sleep(1); b = snap()
allowed = sorted(os.sched_getaffinity(0))
busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in allowed}
print(",".join(map(str, sorted(allowed, key=lambda c: (busy[c], c))[:8])))
PY
)
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c "$cores" "$3" "$HERE/03_run_pydecontx.py" "$t" "$2/pydecontx_8t/$(basename "$t")/run$r" "$r" 8
  done
done
