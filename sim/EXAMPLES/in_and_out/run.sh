#!/bin/sh
# Some environments set LD_PRELOAD (e.g. to libcuda.so); it can spam ld.so warnings.
# This simulator doesn't need it.
unset LD_PRELOAD

../../datacenter/old_tests/htsim_ndp_in_out -o logfile -conns 4 -nodes 128 -cwnd 23 -strat perm -q 8 > ts_in_out
python3 process_data.py logfile