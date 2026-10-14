"""Prove context checks and injected-failure rejection with and without NDEBUG."""
from pathlib import Path
import argparse, hashlib, json, subprocess

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,default=Path(__file__).with_name('n1_continuation_context.cpp'))
p.add_argument('--include',type=Path,required=True)
p.add_argument('--eigen',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[]
for label,flags in [('checks',['-O0']),('release',['-O3','-DNDEBUG'])]:
    binary=a.output/label
    command=['g++','-std=c++17','-pthread',*flags,'-I'+str(a.include),'-I'+str(a.eigen),str(a.source),'-o',str(binary)]
    compile=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (a.output/(label+'_build.log')).write_text(compile.stdout)
    assert compile.returncode==0,compile.stdout
    for injected in [False,True]:
        run=subprocess.run([str(binary),*(['--inject-context-error'] if injected else [])],
                           text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        key=label+('_injected' if injected else '_positive')
        (a.output/(key+'.log')).write_text(run.stdout)
        expected=1 if injected else 0
        marker='N1_CONTINUATION_CONTEXT_FAIL' if injected else 'N1_CONTINUATION_CONTEXT_PASS checks=9'
        passed=run.returncode==expected and marker in run.stdout
        rows.append(dict(case=key,returncode=run.returncode,expected_returncode=expected,
                         status='PASS' if passed else 'FAIL',command=command,
                         binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
receipt=dict(rows=rows,source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
             status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL')
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if receipt['status']=='PASS' else 1)
