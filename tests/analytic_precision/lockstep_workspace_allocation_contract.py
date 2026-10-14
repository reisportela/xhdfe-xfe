"""Record exact core scratch size expressions; bulk allocations intercepted."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--compiler',default='g++')
args=ap.parse_args();out=args.output.resolve()
assert out.is_relative_to(ROOT) and not out.exists()
out.mkdir(parents=True)
source=(ROOT/'src/fe_absorption.cpp').read_text()
start=source.index('        const std::unique_ptr<double[]> u_store = allocate(row_size);')
end=source.index('        ParallelWorkObserver* team_observer',start)
body=source[start:end]
prefix=r'''
#include "lockstep_memory_plan.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
using namespace hdfe::detail;
void run(const char* name, std::size_t n, int lanes,
         const std::vector<int>& groups_per_dim,
         const std::vector<uint8_t>& contiguous, int capacity) {
    const int dims=static_cast<int>(groups_per_dim.size());
    std::size_t total_fe=0,private_levels=0,contiguous_dims=0;
    for (int d=0;d<dims;++d) {
        total_fe+=groups_per_dim[d];
        if (contiguous[d]) ++contiguous_dims;
        else private_levels+=groups_per_dim[d];
    }
    const std::size_t stride=static_cast<std::size_t>(lanes);
    const std::size_t row_size=n*stride,fe_size=total_fe*stride;
    ScopedDeterministicParallelCapacity captured(capacity);
    const auto plan=lockstep_memory_plan(n,stride,total_fe,private_levels,contiguous_dims);
    const int row_chunks=plan.execution_chunks(deterministic_parallel_chunk_count(n));
    if (row_chunks==0) std::exit(2);
    std::vector<std::size_t> requests;
    // Intercept storage, retaining the exact requested counts. This is not
    // physical allocation, OOM, liveness or numerical solver execution.
    auto allocate=[&](std::size_t elements) {
        requests.push_back(elements);
        return std::unique_ptr<double[]>(new double[1]);
    };
    auto fill_levels=[](double*,double) {};
'''
suffix=r'''
    std::size_t private_elements=0,carry_elements=0,level_elements=0;
    if (requests.size()<8 || requests[0]!=row_size) std::exit(2);
    for (std::size_t i=1;i<8;++i) level_elements+=requests[i];
    for (std::size_t i=8;i<requests.size();++i) private_elements+=requests[i];
    for (const auto& values:carried_sums) carry_elements+=values.size();
    const std::size_t actual=(private_elements+carry_elements+row_partials.size())*sizeof(double);
    const std::size_t model=row_chunks*plan.chunk_bytes;
    const bool ok=actual==model && level_elements*sizeof(double)==plan.level_bytes &&
                  actual<=plan.budget && actual+plan.level_bytes<=3*plan.row_bytes;
    std::cout << name << ' ' << capacity << ' ' << row_chunks << ' '
              << actual << ' ' << model << ' ' << plan.max_private_bytes << ' '
              << plan.eligible << ' ' << ok << '\n';
    if (!ok) std::exit(1);
}
int main() {
    for (int capacity:{48,64,96}) {
        run("ready",46156187,5,{5074679,610115},{1,0},capacity);
        run("akm",55947171,7,{5948793,799301},{1,0},capacity);
        run("N9_physical_geometry",300000,32,{5000,10000},{1,0},capacity);
        // Legal size expressions are tested even where routing diverts the
        // historical partition; this does not assert a default lockstep fit.
        run("carry_partition_boundary",266000,250,{10000,11184},{1,0},capacity);
    }
}
'''
program=prefix+body+suffix
header=(ROOT/'src/lockstep_memory_plan.hpp').read_text()
needle='!add(private_levels, contiguous_dims, chunk_entries)';assert header.count(needle)==1
records=[]
partial_needle='!add(chunk_entries, 1, chunk_entries)';assert header.count(partial_needle)==1
for name,h in [('positive',header),
               ('negative_omitted_carries',header.replace(needle,'!add(private_levels, 0, chunk_entries)')),
               ('negative_omitted_partials',header.replace(partial_needle,'!add(chunk_entries, 0, chunk_entries)'))]:
    case=out/name;case.mkdir();(case/'lockstep_memory_plan.hpp').write_text(h)
    cpp=case/'probe.cpp';cpp.write_text(program)
    cmd=[args.compiler,'-std=c++17','-O2','-I',str(case),'-I',str(ROOT/'include'),str(cpp),'-o',str(case/'probe')]
    build=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
    (case/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0
    run=subprocess.run([str(case/'probe')],text=True,capture_output=True,timeout=30)
    (case/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==(0 if name=='positive' else 1),(name,run.returncode)
    records.append(dict(name=name,returncode=run.returncode,rows=run.stdout.splitlines()))
with (out/'receipt.json').open('x') as f:json.dump(dict(status='PASS_REQUESTED_SIZES',
    compiler=args.compiler,core_sha256=hashlib.sha256(source.encode()).hexdigest(),
    extracted_body_sha256=hashlib.sha256(body.encode()).hexdigest(),rows=records,
    negative_omitted_carries_rejected=True,negative_omitted_partials_rejected=True,
    scope='Exact selected persistent allocate/assign/row-partial expressions, twelve geometries/capacities; startup reduction partial and other workspace families excluded; bulk storage intercepted; no whole-scratch/RSS or numerical certification'),f,indent=2)
print('PASS selected core scratch requests; omitted carries and partials rejected')
