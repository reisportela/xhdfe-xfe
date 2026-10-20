"""Negative controls for matrix receipts: missing data must never pass."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(output):
    output = output.resolve()
    assert output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('core_matrix', HERE/'performance/run_matrix.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    reference = dict(success=1, converged=1, precision_certified=1, gpu_used=0,
        terms=['x'], nobs=100, nobs_full=100, num_singletons=0, df_a=4,
        df_m=1, df_resid=95, sample_sha256='same-sample', input_sha256='same-input',
        coef=[2.0], covariance=[[1.0]], elapsed_seconds=1.0, iterations=5)
    rows = []
    for name in ('valid_subset', 'nan_covariance', 'gpu_on_cpu', 'changed_sample',
                 'missing_context', 'failed_process', 'zero_time', 'missing_dof'):
        folder = output/name
        receipts = folder/'receipts/synthetic/native_cpu_comparable'
        receipts.mkdir(parents=True)
        (folder/'MANIFEST.json').write_text(json.dumps(dict(inputs={'synthetic':{}}, artifacts={})))
        candidate = deepcopy(reference)
        if name == 'nan_covariance': candidate['covariance'] = [[float('nan')]]
        if name == 'gpu_on_cpu': candidate['gpu_used'] = 1
        if name == 'changed_sample': candidate['sample_sha256'] = 'different'
        if name == 'zero_time': candidate['elapsed_seconds'] = 0
        if name == 'missing_dof': del candidate['df_resid']
        for arm, value in (('baseline',reference), ('candidate',candidate)):
            (receipts/f'p1_{arm}.json').write_text(json.dumps(value))
            if not (name == 'missing_context' and arm == 'candidate'):
                (receipts/f'p1_{arm}.context.json').write_text(json.dumps(dict(
                    returncode=7 if name == 'failed_process' and arm == 'candidate' else 0)))
        runner.ROOT = folder
        runner.report()
        report = json.loads((folder/'REPORT_001.json').read_text())
        cell = next(r for r in report['cells'] if r['interface']=='native'
                    and r['backend']=='cpu' and r['mode']=='comparable')
        expected = 'PASS' if name == 'valid_subset' else 'REVIEW'
        assert cell['pairs'][0]['status'] == expected, name
        assert report['status'] == 'INCOMPLETE', 'A subset became full certification'
        rows.append(dict(case=name, status='PASS'))
    stage = output/'driver_failure'
    stage.mkdir()
    (stage/'PLAN.json').write_text(json.dumps(dict(artifacts={},
        config=dict(candidate=str(ROOT), dependencies=str(stage)))))
    spec = importlib.util.spec_from_file_location('core_driver', HERE/'run.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    driver.execute = lambda command, path, env=None: 7
    rc = driver.run(SimpleNamespace(output=stage, stage='compatibility_cpu'))
    assert rc == 1, 'A failing required stage returned success'
    assert json.loads((stage/'compatibility_cpu.json').read_text())['status'] == 'FAIL'
    rows.append(dict(case='driver_failure_exit_status', status='PASS'))
    (output/'RESULT.json').write_text(json.dumps(dict(status='PASS', cases=rows),indent=2)+'\n')
    print('CORE_REPORTING_CONTROLS_PASS', len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
