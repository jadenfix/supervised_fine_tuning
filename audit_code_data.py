"""Inspect reference syntax without executing public or generated code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def audit(payload):
    rows = [entry['row'] for entry in json.loads(payload)['rows']]
    checks = []
    for index, row in enumerate(rows):
        try:
            ast.parse(row['output'])
            error = None
        except (SyntaxError, ValueError) as exc:
            error = str(exc)
        checks.append({'source_index': index, 'reference_parseable_python': error is None, 'syntax_error': error})
    return {'source_sha256': hashlib.sha256(payload).hexdigest(), 'records': len(checks), 'parseable_references': sum(c['reference_parseable_python'] for c in checks), 'checks': checks, 'limits': 'Syntax validity alone does not establish semantic correctness, security or suitability as a supervised target. No reference code is executed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_rows')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(Path(args.source_rows).read_bytes())
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ('source_sha256','records','parseable_references','limits')}, indent=2))
