#!/bin/bash
# Runner of tests/sparse_rank_bench.cpp (phase 1 of the Stata-path plan, design
# note section 6): every shape, view and team, one fresh process per
# measurement, REPS rounds, host load recorded before each measurement.
# Today's branch (the old arm) is serial, so it runs once per shape, view and
# round, next to the new arm at the decision team (DECISION_TEAM, 16), the two
# arms alternating which goes first from one round to the next; the other teams
# run the new arm only. The shapes are generated once and cached in DATA_DIR
# (synthetic). Not part of the package.
# usage: sparse_rank_bench.sh <bench binary> <new output csv> <data dir> [REPS]
set -u
BIN=${1:?bench binary}
OUT=${2:?new output csv}
DATA=${3:?data directory}
REPS=${4:-5}
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$DATA" || exit 2
echo "round,load1,shape,view,arm,team,observed_team,n,range,levels,path,ms,maxrss_kb,digest" > "$OUT"
SHAPES=${SHAPES:-"F C E S"}
VIEWS=${VIEWS:-"ascending codes lite nested"}
TEAMS=${TEAMS:-"1 2 8 16 24 48"}
DECISION_TEAM=${DECISION_TEAM:-16}
measure() {
  local round=$1 shape=$2 view=$3 arm=$4 team=$5 load line
  load=$(cut -d' ' -f1 /proc/loadavg)
  line=$(env -u OMP_THREAD_LIMIT -u OMP_NUM_THREADS -u OMP_WAIT_POLICY \
         SPARSE_RANK_BENCH_DATA_DIR="$DATA" "$BIN" "$shape" "$view" "$arm" "$team")
  echo "$round,$load,$line" >> "$OUT"
}
# Generate the shapes once, outside the measurements.
for shape in $SHAPES; do
  env SPARSE_RANK_BENCH_DATA_DIR="$DATA" "$BIN" "$shape" lite new 1 > /dev/null
done
for round in $(seq 1 "$REPS"); do
  for shape in $SHAPES; do
    for view in $VIEWS; do
      for team in $TEAMS; do
        if [ "$team" -eq "$DECISION_TEAM" ]; then
          if [ $((round % 2)) -eq 1 ]; then arms="old new"; else arms="new old"; fi
        else
          arms="new"
        fi
        for arm in $arms; do
          measure "$round" "$shape" "$view" "$arm" "$team"
        done
      done
    done
  done
done
echo "BENCH_DONE $(($(wc -l < "$OUT") - 1)) measurements in $OUT"
