"""Independent PSD/range/Schur checks on the actual intercept-repair helper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import resource
import shlex
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--captured',type=Path)
    ap.add_argument('--production-flags-from',type=Path,
                    help='Use CXX_DEFINES/CXX_FLAGS from an actual CMake flags.make')
    ap.add_argument('--source',type=Path,default=ROOT/'src/hdfe_regressor_v11.cpp')
    ap.add_argument('--only',help='Run one named matrix, preserving its applicability')
    args=ap.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
    source=args.source.read_text()
    start=source.index('// CovMatrix/BlockMatrix are Eigen::MatrixXd (ordinary path) or the OlsWide')
    end=source.index('\nstruct ClusterInterceptMeat',start)
    helper=source[start:end]
    assert source.count('repair_intercept_covariance(cov_full,')==2
    cpp='''#include <Eigen/Eigenvalues>
#include <type_traits>
#include "wide_float.hpp"
#include "wide_covariance.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cfenv>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace detail=hdfe::detail;
'''+helper+'''
int main() {
    // Match the IEEE environment of the loaded module; an executable linked
    // with -ffast-math otherwise enables FTZ/DAZ before main, as covered by
    // the existing wide_product_probe.
    if (std::fesetenv(FE_DFL_ENV)) return 2;
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() & ~0x8040U);
#endif
    int cases; if (!(std::cin>>cases)) return 2;
    std::cout<<std::setprecision(17);
    for (int c=0;c<cases;++c) {
        int n;std::cin>>n;Eigen::MatrixXd V(n,n),raw(n-1,n-1);Eigen::VectorXd scale(n);
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) std::cin>>V(i,j);
        for (int i=0;i<n-1;++i) for (int j=0;j<n-1;++j) std::cin>>raw(i,j);
        for (int i=0;i<n;++i) std::cin>>scale[i];
        if (!std::cin) return 2;
        repair_intercept_covariance(V,raw,scale);
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) std::cout<<V(i,j)<<' ';
        std::cout<<'\\n';
    }
}
'''
    (out/'probe.cpp').write_text(cpp)
    flags=['-std=c++17','-O2']
    if args.production_flags_from:
        text=args.production_flags_from.read_text()
        values=dict(line.split(' = ',1) for line in text.splitlines() if ' = ' in line)
        flags=shlex.split(values['CXX_DEFINES'])+shlex.split(values['CXX_FLAGS'])
    cmd=['g++',*flags,'-I',str(ROOT/'src'),'-I',str(ROOT/'include'),
         '-I',str(ROOT/'r/xhdfe/src/eigen'),str(out/'probe.cpp'),'-o',str(out/'probe')]
    build=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (out/'build.log').write_text(build.stdout+build.stderr);assert build.returncode==0
    fixtures=[]
    def add(name,B,c,a,raw=None,scale=None):
        B=np.asarray(B,float);c=np.asarray(c,float);p=len(c);V=np.zeros((p+1,p+1))
        V[:p,:p]=B;V[:p,p]=c;V[p,:p]=c;V[p,p]=a
        fixtures.append(dict(name=name,V=V,raw=B if raw is None else np.asarray(raw,float),
                             scale=np.ones(p+1) if scale is None else np.asarray(scale,float)))
    add('singular_outside_range',[[2,0],[0,0]],[2,3],1)
    add('zero_block',np.zeros((2,2)),[2,-3],4)
    add('negative_original_direction',[[0,0],[0,3]],[4,6],1,raw=[[-1,0],[0,3]])
    add('positive_schur_defect',[[2,0],[0,3]],[2,6],1)
    add('already_positive',[[2,0],[0,3]],[1,1],4)
    for exponent in [-40,40]:
        scaling=np.array([2.**exponent,2.**-exponent,1.])
        V=np.array([[2,0,2],[0,0,3],[2,3,1]],float)/np.outer(scaling,scaling)
        add('rescaled_singular_'+str(exponent),V[:2,:2],V[:2,2],V[2,2],scale=scaling)
    add('wide_spectrum',[[2.**40,0],[0,2.**-40]],[2.**20,2.**-20],1)
    add('division_first_overflow',[[2.**-1040,0],[0,1]],[2.**-10,0],1)
    add('opposite_coordinates',[[2,1],[1,2]],[2,-2],1)
    # Common covariance-unit changes have exact power-of-two transport. Grade
    # in the original units below so the independent Schur oracle cannot lose
    # the coordinate square that this control is designed to detect.
    for fixture in list(fixtures):
        if fixture['name'] not in ('singular_outside_range', 'positive_schur_defect',
                                   'already_positive', 'wide_spectrum', 'opposite_coordinates'):
            continue
        for exponent in (-930, 930):
            unit=2.**exponent
            fixtures.append(dict(name=fixture['name']+'_units_'+str(exponent),
                V=fixture['V']*unit,raw=fixture['raw']*unit,
                scale=fixture['scale'].copy(),common_unit=unit))
    if args.captured:
        import pandas as pd
        table=pd.read_stata(args.captured)
        assert len(table)==40
        for row in table.to_dict('records'):
            V=np.array([[row[f'fit{i}{j}'] for j in range(1,5)] for i in range(1,5)])
            raw=np.array([[row[f'raw{i}{j}'] for j in range(1,4)] for i in range(1,4)])
            add('captured_seed_'+str(int(row['seed'])),V[:3,:3],V[:3,3],V[3,3],
                raw=raw,scale=[row['sd1'],row['sd2'],row['sd3'],1])
    negative_fixture=fixtures[0]
    if args.only:
        fixtures=[row for row in fixtures if row['name']==args.only]
        assert fixtures, 'unknown matrix'
    lines=[str(len(fixtures))]
    for row in fixtures:
        lines.append(str(len(row['V'])))
        lines.append(' '.join(format(x,'.17g') for x in row['V'].ravel()))
        lines.append(' '.join(format(x,'.17g') for x in row['raw'].ravel()))
        lines.append(' '.join(format(x,'.17g') for x in row['scale']))
    run=subprocess.run([str(out/'probe')],input='\n'.join(lines)+'\n',capture_output=True,text=True,timeout=60)
    (out/'run.log').write_text(run.stdout+run.stderr);assert run.returncode==0
    actual=run.stdout.splitlines();assert len(actual)==len(fixtures)
    records=[];limits=[]
    def indefinite(V):
        magnitude=np.max(np.abs(V))
        if magnitude>0: V=V/magnitude
        u=2.**-53;allowance=256*u/(1-256*u)*np.linalg.norm(V,2)
        return bool(float(np.linalg.eigvalsh((V+V.T)/2).min()) < -allowance)
    for row,line in zip(fixtures,actual):
        unit=row.get('common_unit',1.)
        before=row['V']/unit;n=len(before);V=np.fromstring(line,sep=' ').reshape(n,n)/unit
        if row['name']=='division_first_overflow':
            # Its slope variance 2^-1040 is subnormal, outside the normal
            # covariance range accepted by the public inference path. Both
            # baseline and candidate helpers fail here with production flags;
            # retain the measurement as a known limit, never a numerical PASS.
            limits.append(dict(name=row['name'],status='OUTSIDE_NORMAL_RANGE',
                actual_intercept_variance=float(V[-1,-1]),
                exact_intercept_variance=2.**1020,
                relative_error=float(abs(V[-1,-1]/(2.**1020)-1)),
                subnormal_input_diagonal=float(before[0,0])))
            continue
        assert np.isfinite(V).all() and np.array_equal(V[:n-1,:n-1],before[:n-1,:n-1])
        assert np.array_equal(V,V.T) and not indefinite(V),row['name']
        s=row['scale'];raw=(row['raw']/unit)*np.outer(s[:-1],s[:-1]);eig,Q=np.linalg.eigh((raw+raw.T)/2)
        keep=eig>0;coords=Q[:,keep].T@(before[:-1,-1]*s[:-1]*s[-1])
        c=Q[:,keep]@coords;needed=np.sum(coords*coords/eig[keep])
        expected=before.copy();expected[:-1,-1]=c/(s[:-1]*s[-1]);expected[-1,:-1]=expected[:-1,-1]
        expected[-1,-1]=max(before[-1,-1],needed/(s[-1]*s[-1]))
        se=np.sqrt(np.maximum(0,np.diag(expected)))
        scale=np.maximum(np.abs(expected),np.outer(se,se))
        error=float(np.max(np.abs(V-expected)/np.maximum(scale,np.finfo(float).tiny)))
        assert error<=1e-8,(row['name'],error)
        records.append(dict(name=row['name'],status='PASS',slope_block_exact=True,oracle_scaled_error=error,
                            original_indefinite=indefinite(before),repaired_indefinite=False))
    negative=negative_fixture['V'].copy();negative[-1,-1]=2
    assert indefinite(negative_fixture['V']) and indefinite(negative)
    status='PASS_WITH_KNOWN_LIMIT' if limits else 'PASS'
    with (out/'receipt.json').open('x') as f:json.dump(dict(status=status,rows=records,known_limits=limits,
        compile_command=cmd,
        original_and_diagonal_only_negative_rejected=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scope='Actual helper extracted unchanged; analytic singular/zero/range/scaled controls and optional captured synthetic matrices; no performance certification'),f,indent=2)
    print('INTERCEPT_PSD_CONTRACT',status,len(records),'normal matrices;',len(limits),'known limits',flush=True)

if __name__=='__main__':main()
