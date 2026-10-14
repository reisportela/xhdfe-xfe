"""Offline allocation/publication guard for the exact shared-TLS core lambda."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def extract(source):
    scope=source[source.index('AbsorptionResult absorb_fixed_effects_mlsmr('):]
    match=re.search(r'    bool shared_zt_ready = false;\n    auto prepare_shared_zt = \[&\]\(\) \{.*?\n    \};',scope,re.S)
    assert match,'shared preparation body absent'
    zt=scope[scope.index('    auto compute_Zt = '):scope.index('    auto apply_A = ')]
    at=scope[scope.index('    auto apply_lsmr_At_impl ='):scope.index('    auto apply_lsmr_At = ')]
    assert zt.index('prepare_shared_zt();')<zt.index('Eigen::VectorXd b(')<zt.index('#pragma omp')
    assert at.index('if (tls_override == nullptr) prepare_shared_zt();')<at.index('Eigen::VectorXd out(')<at.index('#pragma omp')
    setup=scope[:match.start()]
    assert 'tls_zt[static_cast<std::size_t>(d)].assign(' not in setup
    return match.group()


PREFIX='''#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <vector>
struct ProbeVector {
    std::vector<double> values;
    static int zero_calls, fail_at;
    static ProbeVector Zero(int size) {
        ++zero_calls;
        if (zero_calls==fail_at) throw std::bad_alloc();
        return {std::vector<double>(size,0.0)};
    }
};
int ProbeVector::zero_calls=0;
int ProbeVector::fail_at=-1;
namespace Eigen { using VectorXd=ProbeVector; }
void require(bool ok,const char* why) {
    if (!ok) {std::cerr << "FAIL " << why << '\\n';std::exit(1);}
}
'''
SUFFIX='''
    require(!shared_zt_ready && ProbeVector::zero_calls==0,"no eager allocation");
    if (failure) {
        ProbeVector::fail_at=2;
        bool caught=false;
        try { prepare_shared_zt(); } catch (const std::bad_alloc&) {caught=true;}
        require(caught && !shared_zt_ready,"failed allocation not published");
        ProbeVector::fail_at=-1;
    }
    prepare_shared_zt();
    require(shared_zt_ready,"successful preparation published");
    int allocated_dims=0;
    for (int d=0;d<dims;++d) {
        if (use_group_owner_scatter[d]) {
            require(tls_zt[d].empty(),"owner traversal does not allocate TLS");
        } else {
            ++allocated_dims;
            require(tls_zt[d].size()==std::size_t(scatter_chunks_per_dim[d]),"exact chunks");
            for (const auto& v:tls_zt[d]) {
                require(v.values.size()==std::size_t(groups_per_dim[d]),"exact levels");
                for (double x:v.values) require(x==0.0,"zero initialization");
            }
        }
    }
    const int calls=ProbeVector::zero_calls;
    for (auto& dim:tls_zt) for (auto& v:dim) for (double& x:v.values) x=3.0;
    prepare_shared_zt();
    require(ProbeVector::zero_calls==calls,"prepare only once");
    for (const auto& dim:tls_zt) for (const auto& v:dim)
        for (double x:v.values) require(x==3.0,"do not overwrite active sums");
}
int main() {
    for (int owners=0;owners<4;++owners) scenario(owners,false);
    scenario(0,true);
    std::cout << "PASS shared allocation: delayed, exact, owner-aware, idempotent, failure unpublished\\n";
}
'''


def program(body):
    setup='''void scenario(int owners,bool failure) {
    const int dims=2;
    std::vector<int> groups_per_dim{3,7},scatter_chunks_per_dim{2,4};
    std::vector<std::uint8_t> use_group_owner_scatter{
        std::uint8_t(owners&1),std::uint8_t((owners>>1)&1)};
    std::vector<std::vector<Eigen::VectorXd>> tls_zt(dims);
    ProbeVector::zero_calls=0;ProbeVector::fail_at=-1;
'''
    return PREFIX+setup+body+SUFFIX


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compiler',default='g++')
    args=ap.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
    source=(ROOT/'src/fe_absorption.cpp').read_text();body=extract(source)
    rows=[]
    for name,text,expected in [('positive',body,0),('premature_publish',
        body.replace('        for (int d = 0; d < dims; ++d) {',
                     '        shared_zt_ready = true;\n        for (int d = 0; d < dims; ++d) {'),1)]:
        cpp=out/(name+'.cpp');cpp.write_text(program(text))
        command=[args.compiler,'-std=c++17','-O2','-Wall','-Wextra','-pedantic',str(cpp),'-o',str(out/name)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=60)
        assert run.returncode==0,run.stderr
        run=subprocess.run([str(out/name)],capture_output=True,text=True,timeout=30)
        assert run.returncode==expected,(name,run.stdout,run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    needle='if (tls_override == nullptr) prepare_shared_zt();';assert source.count(needle)==1
    caught=False
    try:extract(source.replace(needle,''))
    except (AssertionError,ValueError):caught=True
    assert caught,'missing consumer initialization not rejected'
    report=dict(status='PASS',compiler=args.compiler,rows=rows,missing_consumer_negative_rejected=caught,
        core_sha256=hashlib.sha256((ROOT/'src/fe_absorption.cpp').read_bytes()).hexdigest(),
        scope='Exact source lambda with allocation-failure probe; estimator/inference verified separately')
    with (out/'receipt.json').open('x') as handle:json.dump(report,handle,indent=2)
    print('PASS shared scatter allocation and negative controls')


if __name__=='__main__':main()
