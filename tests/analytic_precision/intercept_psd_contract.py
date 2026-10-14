"""Independent PSD/range/Schur checks on the actual intercept-repair helper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--captured',type=Path)
    args=ap.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
    source=(ROOT/'src/hdfe_regressor_v11.cpp').read_text()
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
namespace detail=hdfe::detail;
'''+helper+'''
int main() {
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
    cmd=['g++','-std=c++17','-O2','-I',str(ROOT/'src'),'-I',str(ROOT/'include'),
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
    if args.captured:
        import pandas as pd
        table=pd.read_stata(args.captured)
        assert len(table)==40
        for row in table.to_dict('records'):
            V=np.array([[row[f'fit{i}{j}'] for j in range(1,5)] for i in range(1,5)])
            raw=np.array([[row[f'raw{i}{j}'] for j in range(1,4)] for i in range(1,4)])
            add('captured_seed_'+str(int(row['seed'])),V[:3,:3],V[:3,3],V[3,3],
                raw=raw,scale=[row['sd1'],row['sd2'],row['sd3'],1])
    lines=[str(len(fixtures))]
    for row in fixtures:
        lines.append(str(len(row['V'])))
        lines.append(' '.join(format(x,'.17g') for x in row['V'].ravel()))
        lines.append(' '.join(format(x,'.17g') for x in row['raw'].ravel()))
        lines.append(' '.join(format(x,'.17g') for x in row['scale']))
    run=subprocess.run([str(out/'probe')],input='\n'.join(lines)+'\n',capture_output=True,text=True,timeout=60)
    (out/'run.log').write_text(run.stdout+run.stderr);assert run.returncode==0
    actual=run.stdout.splitlines();assert len(actual)==len(fixtures)
    records=[]
    def indefinite(V):
        u=2.**-53;allowance=256*u/(1-256*u)*np.linalg.norm(V,2)
        return bool(float(np.linalg.eigvalsh((V+V.T)/2).min()) < -allowance)
    for row,line in zip(fixtures,actual):
        before=row['V'];n=len(before);V=np.fromstring(line,sep=' ').reshape(n,n)
        assert np.isfinite(V).all() and np.array_equal(V[:n-1,:n-1],before[:n-1,:n-1])
        assert np.array_equal(V,V.T) and not indefinite(V),row['name']
        s=row['scale'];raw=row['raw']*np.outer(s[:-1],s[:-1]);eig,Q=np.linalg.eigh((raw+raw.T)/2)
        keep=eig>0;coords=Q[:,keep].T@(before[:-1,-1]*s[:-1]*s[-1])
        c=Q[:,keep]@coords;needed=np.sum(coords*coords/eig[keep])
        expected=before.copy();expected[:-1,-1]=c/(s[:-1]*s[-1]);expected[-1,:-1]=expected[:-1,-1]
        expected[-1,-1]=max(before[-1,-1],needed/(s[-1]*s[-1]))
        scale=np.maximum(np.abs(expected),np.sqrt(np.maximum(0,np.outer(np.diag(expected),np.diag(expected)))))
        error=float(np.max(np.abs(V-expected)/np.maximum(scale,np.finfo(float).tiny)))
        assert error<=1e-8,(row['name'],error)
        records.append(dict(name=row['name'],status='PASS',slope_block_exact=True,oracle_scaled_error=error,
                            original_indefinite=indefinite(before),repaired_indefinite=False))
    negative=fixtures[0]['V'].copy();negative[-1,-1]=2
    assert indefinite(fixtures[0]['V']) and indefinite(negative)
    with (out/'receipt.json').open('x') as f:json.dump(dict(status='PASS',rows=records,
        original_and_diagonal_only_negative_rejected=True,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scope='Actual helper extracted unchanged; analytic singular/zero/range/scaled controls and optional captured synthetic matrices; no performance certification'),f,indent=2)
    print('INTERCEPT_PSD_CONTRACT_PASS',len(records),'matrices',flush=True)

if __name__=='__main__':main()
