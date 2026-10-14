"""Reproduce the large synthetic Walsh residual diagnostic; keep its raw FAIL."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess

p=argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True)
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['cpu','cuda'],required=True)
p.add_argument('--threads',type=int,nargs='+',default=[1,2,8,16,24,48])
p.add_argument('--version',default='2.29.0 06oct2026')
a=p.parse_args();a.package=a.package.resolve(strict=True);a.fixture=a.fixture.resolve(strict=True)
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
source=Path(__file__).with_suffix('.do');shutil.copy2(source,a.output/'tested.do')
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith('XHDFE_') and k!='LD_LIBRARY_PATH'}
env.update(OPENBLAS_NUM_THREADS='1',OMP_DYNAMIC='FALSE')
rows=[]
for team in a.threads:
    run=a.output/f't{team}';run.mkdir();(run/'tmp').mkdir()
    args=[a.package,run/'metrics.txt',a.fixture,team,a.backend,a.version]
    driver=run/'driver.do'
    driver.write_text('capture noisily do "'+str(source)+'" '+' '.join('"'+str(x)+'"' for x in args)+'\n'
                      'local diagnostic_rc=_rc\ndi "WALSH_RESULT rc=" `diagnostic_rc\'\nexit\n')
    process=subprocess.run(['/usr/local/stata/stata-mp','-q','-b','do',str(driver)],cwd=run,
         env=dict(env,STATATMP=str(run/'tmp'),TMPDIR=str(run/'tmp'),CUDA_CACHE_PATH=str(run/'tmp/cuda')),
         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=420)
    (run/'process.txt').write_bytes(process.stdout)
    log=(run/'driver.log').read_text(errors='replace') if (run/'driver.log').exists() else ''
    codes=re.findall(r'^WALSH_RESULT rc=(\d+)$',log,re.M)
    rc=int(codes[-1]) if codes else None
    metrics={}
    if (run/'metrics.txt').exists():
        for line in (run/'metrics.txt').read_text().splitlines():
            key,value=line.split();metrics[key]=None if value=='.' else float(value)
    passed=process.returncode==0 and rc==0 and 'WALSH_PUBLISHED_CONTRACT_PASS' in log and metrics.get('core_pass')==1
    row=dict(team=team,backend=a.backend,process_rc=process.returncode,stata_rc=rc,
             status='PASS' if passed else 'FAIL',metrics=metrics)
    rows.append(row)
    (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(team,row['status'],json.dumps(metrics),flush=True)
receipt=dict(rows=rows,package=str(a.package),fixture=str(a.fixture),
    fixture_sha256=sha(a.fixture),script_sha256=sha(source),version=a.version,
    plugin_sha256=sha(a.package/'stata/xhdfe.plugin'),ado_sha256=sha(a.package/'stata/xhdfe.ado'),
    scope='Published b/V/FE/normal-moment contract; the original pointwise diagnostic is retained separately',
    status='PASS_CORE_CONTRACT' if all(r['status']=='PASS' for r in rows) else 'FAIL')
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(0 if receipt['status']=='PASS_CORE_CONTRACT' else 1)
