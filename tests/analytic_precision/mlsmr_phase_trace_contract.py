"""Offline useful-worker guard using the exact phase observer body in the core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser()
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
out=args.output.resolve();assert out.is_relative_to(ROOT) and not out.exists()
out.mkdir(parents=True)
source=(ROOT/'src/fe_absorption.cpp').read_text()
match=re.search(r'class MlsmrPhaseTrace \{.*?\n\};',source,re.S);assert match
body=match.group()
prefix='''#include "hdfe/parallel_work_observer.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>
using hdfe::detail::ParallelWorkObserver;
'''
harness='''
int main(int argc, char** argv) {
    const int team=std::atoi(argv[1]);
    omp_set_dynamic(0);
    MlsmrPhaseTrace rows, levels, narrow, empty;
    std::vector<int> used(team,0);
    std::atomic<std::uint64_t> checksum{0};
    std::atomic<int> row_team{0}, level_team{0}, narrow_team{0};
    rows.begin(team,10);
#pragma omp parallel for num_threads(team) schedule(dynamic,1)
    for (int chunk=0;chunk<10;++chunk) {
        const int tid=omp_get_thread_num();
        used[tid]=1;
        row_team.store(omp_get_num_threads());
        rows.observe();
        std::uint64_t value=chunk+1;
        for (int i=0;i<1000000;++i) value=value*6364136223846793005ULL+1;
        checksum.fetch_xor(value);
    }
    rows.end();
    const int oracle_rows=std::accumulate(used.begin(),used.end(),0);
    std::fill(used.begin(),used.end(),0);
    levels.begin(team,48);
#pragma omp parallel for num_threads(team) schedule(static)
    for (int chunk=0;chunk<48;++chunk) {
        used[omp_get_thread_num()]=1;
        level_team.store(omp_get_num_threads());
        levels.observe();
        std::uint64_t value=chunk+1;
        for (int i=0;i<100000;++i) value=value*6364136223846793005ULL+1;
        checksum.fetch_xor(value);
    }
    levels.end();
    const int oracle_levels=std::accumulate(used.begin(),used.end(),0);
    narrow.begin(team,1);
#pragma omp parallel for num_threads(team) schedule(static)
    for (int chunk=0;chunk<1;++chunk) {
        narrow_team.store(omp_get_num_threads());
        narrow.observe();
        checksum.fetch_xor(1);
    }
    narrow.end();
    empty.begin(team,0);empty.end();
    rows.report("rows");levels.report("levels");narrow.report("narrow");empty.report("empty");
    std::cout << "oracle " << oracle_rows << ' ' << oracle_levels << ' '
              << row_team.load() << ' ' << level_team.load() << ' '
              << narrow_team.load() << ' '
              << checksum.load() << '\\n';
}
'''
def check(name,class_body,expect_failure=False):
    path=out/(name+'.cpp');path.write_text(prefix+class_body+harness)
    command=['g++','-std=c++17','-O2','-fopenmp','-DHDFE_USE_OPENMP','-I',str(ROOT/'include'),str(path),'-o',str(out/name)]
    build=subprocess.run(command,capture_output=True,text=True,timeout=60);assert build.returncode==0,build.stderr
    rows=[]
    for team in (16,48):
        run=subprocess.run([str(out/name),str(team)],capture_output=True,text=True,timeout=30,
            env=dict(os.environ,OMP_DYNAMIC='FALSE'))
        assert run.returncode==0,run.stderr
        _,nrows,nlevels,rteam,lteam,nteam,_=run.stdout.strip().split()
        nrows,nlevels,rteam,lteam,nteam=map(int,(nrows,nlevels,rteam,lteam,nteam))
        records={p:dict(units=int(u),team=int(t),active=int(a)) for p,u,t,a in re.findall(
            r'phase=(\w+) units=(\d+) team=(\d+) active_workers=(\d+)',run.stderr)}
        ok=(1<=nrows<=min(10,rteam) and nlevels==lteam
            and records['rows']==dict(units=10,team=rteam,active=nrows)
            and records['levels']==dict(units=48,team=lteam,active=nlevels)
            and records['narrow']==dict(units=1,team=nteam,active=1)
            and records['empty']==dict(units=0,team=0,active=0))
        applicable=not expect_failure or nteam>1
        rows.append(dict(team=team,passed=ok,oracle_rows=nrows,oracle_levels=nlevels,
                         actual_teams=[rteam,lteam,nteam],control_applicable=applicable,observed=records))
        if applicable:
            assert ok != expect_failure,(name,rows)
    return rows
positive=check('phase_positive',body)
needle='observer_.active_workers()';assert body.count(needle)==1
negative=check('phase_negative',body.replace(needle,'observer_.max_team_size()'),True)
negative_rejected=any(row['control_applicable'] for row in negative)
report=dict(status='PASS' if negative_rejected else 'NOT_RUN_RUNTIME_SERIAL',
    source_sha256=hashlib.sha256((ROOT/'src/fe_absorption.cpp').read_bytes()).hexdigest(),
    positive=positive,negative=negative,scope='Exact core trace class; independent per-worker useful-work slots; team echoed as active is rejected')
with (out/'receipt.json').open('x') as handle:json.dump(report,handle,indent=2)
if negative_rejected:
    print('PASS actual phase workers and rejected team-as-workers negative control')
else:
    print('NOT_RUN team-as-workers negative control: runtime formed only serial teams')
    raise SystemExit(2)
