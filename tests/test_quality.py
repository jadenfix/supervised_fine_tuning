import unittest
from quality_sft import encode, split_rows, code_quality, fingerprint

class Tokenizer:
    eos_token_id = 9
    pad_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[0]['content']))) + [8]
    def encode(self, text, **kwargs):
        return [3] * len(text)

class QualityTests(unittest.TestCase):
    def test_complete_response_and_masking(self):
        result = encode({'instruction':'abc','output':'xy'}, Tokenizer(), 7)
        self.assertEqual(result['labels'], [-100]*4 + [3,3,9])
        self.assertIsNone(encode({'instruction':'abcd','output':'xy'}, Tokenizer(), 7))
    def test_duplicates_cannot_cross_split(self):
        rows = [{'instruction':f'task {i}','output':f'answer {i}'} for i in range(6)]
        rows += [dict(rows[0], output='different answer'), dict(rows[1], instruction='new task')]
        train, test, skipped = split_rows(rows, Tokenizer(), 4, 2)
        self.assertEqual(skipped['duplicate_prompt'], 1)
        self.assertEqual(skipped['duplicate_answer'], 1)
        self.assertFalse({fingerprint(r) for r in train} & {fingerprint(r) for r in test})
        self.assertEqual((len(train),len(test)), (4,2))
    def test_parseable_is_not_claimed_correct(self):
        self.assertTrue(code_quality('```python\ndef add(a,b):\n    return a-b\n```')['parseable_python'])
        self.assertFalse(code_quality('   ')['nonblank'])
        self.assertFalse(code_quality('def broken(')['parseable_python'])
