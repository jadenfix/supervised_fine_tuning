import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from transformers import AutoTokenizer
from cpu_sft import MODEL,REVISION,encode,batch

class MaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        cls.tokenizer.pad_token=cls.tokenizer.eos_token
    def test_long_instruction_keeps_supervised_response(self):
        row={"instruction":"long instruction "*1000,"input":"","output":"def add(a,b): return a+b"}
        result=encode(row,self.tokenizer)
        self.assertEqual(len(result['input_ids']),96)
        self.assertTrue(any(x!=-100 for x in result['labels']))
        self.assertTrue(all(x==-100 for x in result['labels'][:40]))
        self.assertTrue(all(label==-100 for label,mask in zip(result['labels'],result['attention_mask']) if not mask))
    def test_batch_shapes_and_targets(self):
        row={"instruction":"Explain","input":"addition","output":"add two values"}
        result=batch([row,row],self.tokenizer)
        self.assertEqual(tuple(result['labels'].shape),(2,96))
        self.assertTrue((result['labels']!=-100).any())

if __name__=='__main__':unittest.main()
