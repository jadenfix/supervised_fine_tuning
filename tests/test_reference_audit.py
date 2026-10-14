import json
import unittest
from audit_code_data import audit

class ReferenceAuditTests(unittest.TestCase):
    def test_syntax_check_does_not_execute_valid_code(self):
        rows = {'rows':[{'row':{'output':'raise RuntimeError("must not execute")'}}, {'row':{'output':'features = . . .'}}]}
        result = audit(json.dumps(rows).encode())
        self.assertEqual(result['records'], 2)
        self.assertEqual(result['parseable_references'], 1)
        self.assertIsNotNone(result['checks'][1]['syntax_error'])
