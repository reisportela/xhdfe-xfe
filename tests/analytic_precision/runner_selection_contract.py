"""Exercise empty selection, resumption and interrupted attempts through the CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RUNNER=Path(__file__).with_name('run.py')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--module',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();module=args.module.resolve(strict=True)
    assert out.is_relative_to(ROOT) and module.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    records=[]
    def call(label,action,campaign,*extra,expected=0):
        cmd=[sys.executable,'-B',str(RUNNER),action,'--output',str(campaign),*extra]
        proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,timeout=120)
        with (out/(label+'.log')).open('x') as f:f.write(proc.stdout+proc.stderr)
        records.append(dict(label=label,returncode=proc.returncode,command=cmd))
        assert proc.returncode==expected,(label,proc.returncode,proc.stderr)
        return proc.stdout+proc.stderr
    def prepare(label):
        campaign=out/label
        call(label+'_prepare','prepare',campaign,'--cases','ols','--interfaces','native',
             '--backends','cpu','--native-cpu-module',str(module))
        manifest=json.loads((campaign/'manifest.json').read_text())
        assert len(manifest['jobs'])==3
        return campaign,manifest
    campaign,manifest=prepare('partial')
    text=call('empty','run',campaign,'--filter','__NO_SUCH_CASE__',expected=1)
    assert 'No jobs match this selection' in text and not list((campaign/'attempts').iterdir())
    text=call('empty_backend','run',campaign,'--backend','cuda',expected=1)
    assert 'No jobs match this selection' in text and not list((campaign/'attempts').iterdir())
    first=manifest['jobs'][0]['id']
    text=call('one_selected','run',campaign,'--filter',first)
    assert 'RUN_SCOPE SELECTED_COMPLETE SELECTED 1 PENDING 0' in text
    assessment=campaign/'attempts'/first/'assessment.json'
    original=assessment.read_bytes()
    text=call('resume','run',campaign,'--filter',first)
    assert 'RUN_SCOPE ALREADY_COMPLETE' in text and 'RECORDED_ATTEMPTS 0' in text
    assert assessment.read_bytes()==original
    call('report_missing','report',campaign,expected=1)
    report=json.loads((campaign/'report_001.json').read_text())
    assert report['counts']=={'PASS':1,'NOT_RUN':2} and not report['xhdfe_scoped_certified']
    text=call('limited','run',campaign,'--limit','1')
    assert 'RUN_SCOPE PARTIAL SELECTED 3 PENDING 1' in text
    last=manifest['jobs'][-1]
    interrupted=campaign/'attempts'/last['id'];interrupted.mkdir()
    with (interrupted/'job.json').open('x') as f:json.dump(last,f)
    before={str(p.relative_to(campaign)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (campaign/'attempts').rglob('*') if p.is_file()}
    text=call('interrupted','run',campaign,expected=1)
    assert 'Incomplete attempts preserved' in text
    after={str(p.relative_to(campaign)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (campaign/'attempts').rglob('*') if p.is_file()}
    assert before==after
    call('report_interrupted','report',campaign,expected=1)
    report=json.loads((campaign/'report_002.json').read_text())
    assert report['counts']=={'PASS':2,'NOT_RUN':1} and not report['xhdfe_scoped_certified']
    complete,_=prepare('complete')
    text=call('all','run',complete)
    assert 'RUN_SCOPE SELECTED_COMPLETE SELECTED 3 PENDING 0' in text
    call('report_complete','report',complete)
    report=json.loads((complete/'report_001.json').read_text())
    assert report['counts']=={'PASS':3} and report['xhdfe_scoped_certified'] and report['t01_scoped_pass']
    with (out/'receipt.json').open('x') as f:
        json.dump(dict(status='PASS',checks=records,runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
            completed_and_interrupted_receipts_preserved=True,
            scope='Real CLI; synthetic OLS; selection/resumption/report protocol, not general estimator certification'),f,indent=2)
    print('RUNNER_SELECTION_CONTRACT_PASS',len(records),'CLI checks',flush=True)

if __name__=='__main__':main()
