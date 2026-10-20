"""Portable consumers must retain fixture custody and numerical rejection."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("student_t_contract", ROOT/'tests/test_student_t_precision.py')
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


class StudentTProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='xhdfe-student-t-')
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.fixture = ROOT/'tests/fixtures/student_t_stata_fixtures.txt'
        with self.fixture.open() as stream:
            reader = csv.DictReader(stream)
            self.rows = list(reader)
            self.fields = reader.fieldnames
        resource = self.out/'generator-resource'
        resource.write_text('synthetic producer toolchain identity\n')
        digest = hashlib.sha256(resource.read_bytes()).hexdigest()
        high = json.loads((ROOT/'tests/fixtures/student_t_high_precision_provenance.json').read_text())
        for key in ['boost_version_header', 'students_t_header']:
            high['arithmetic'][key] = str(resource)
            high['arithmetic'][key+'_sha256'] = digest
        high['compiler'].update(path=str(resource), sha256='0'*64)
        self.high = self.out/'high.json'
        self.high.write_text(json.dumps(high))

    def emission(self, corrupt=False):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=['name','working_critical',
            'binary64_critical','working_p','binary64_p'])
        writer.writeheader()
        for index, row in enumerate(self.rows):
            value = float(row['critical'])
            working = value + .01*max(1,abs(value)) if corrupt and index==0 else value
            writer.writerow(dict(name=row['name'], working_critical=working,
                binary64_critical=value, working_p=row['p'] or '0', binary64_p=row['p'] or '0'))
        return stream.getvalue()

    def consume(self, *, strict=False, fixture=None, corrupt=False):
        arguments = ['contract','--executable','synthetic-probe','--fixtures',str(fixture or self.fixture),
            '--provenance',str(ROOT/'tests/fixtures/student_t_stata_provenance.json'),
            '--high-precision-provenance',str(self.high)]
        if strict:
            arguments.append('--verify-generator-host')
        result = subprocess.CompletedProcess(['synthetic-probe','--emit'],0,
                                             stdout=self.emission(corrupt),stderr='')
        with mock.patch.object(sys,'argv',arguments), mock.patch.object(CONTRACT.subprocess,'run',return_value=result), contextlib.redirect_stdout(io.StringIO()):
            return CONTRACT.main()

    def test_different_consumer_host_keeps_numeric_checks(self):
        self.assertEqual(self.consume(),0)

    def test_explicit_generator_reproduction_requires_host_identity(self):
        with self.assertRaisesRegex(AssertionError,'generator host differs'):
            self.consume(strict=True)

    def test_actual_cli_fixture_is_bound_to_provenance(self):
        changed = self.out/'changed.csv'
        with changed.open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=self.fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(dict(row,p=''))
        with self.assertRaisesRegex(AssertionError,'consumed fixture hash differs'):
            self.consume(fixture=changed)

    def test_wrong_numeric_output_still_fails_on_another_host(self):
        with self.assertRaisesRegex(AssertionError,'working_critical differs'):
            self.consume(corrupt=True)


if __name__=='__main__':
    unittest.main()
