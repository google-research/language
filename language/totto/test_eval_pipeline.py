# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Test the eval pipeline defined in totto_eval.sh.

Run this script to test that the libraries called in the totto_eval script are
returning the correct output.
"""

import json
import prepare_references_for_eval
import sacrebleu
import six


class TestEval:
  """Test class for reference formatting and BLEU scoring."""

  def _format_preds(self, input_path):
    """helper function to get the multi_reference for a file."""
    references = []
    with open(input_path, "r") as input_file:
      for line in input_file:
        line = six.ensure_text(line, "utf-8")
        json_example = json.loads(line)
        multi_reference, multi_overlap_reference, multi_nonoverlap_reference = (
            prepare_references_for_eval.get_references(json_example, "dev"))
        del multi_overlap_reference, multi_nonoverlap_reference
        references.append([r.lower() for r in multi_reference])
    return references

  def test_ref_format(self):
    """Tests whether the references are returned as expected."""
    input_path = "sample/dev_sample.jsonl"
    references = self._format_preds(input_path)
    final_example = ("the nashville (2012 tv series) premiered on october 10, "
                     "2012 had 8.93 million viewers.")
    # Ensure that the final example is correct.
    assert references[-1][-1] == final_example

  def test_bleu_eval(self):
    """Tests whether we are seeing the expected BLEU score."""
    input_path = "sample/dev_sample.jsonl"
    references = self._format_preds(input_path)
    # Sacrebleu expects dimension transpose (1 list per reference count).
    references_sacrebleu = [list(x) for x in zip(*references)]
    output_path = "sample/output_sample.txt"
    with open(output_path, "r") as f:
      predictions = [p.strip().lower() for p in f]

    expected_bleu = 45.5
    bleu = sacrebleu.corpus_bleu(predictions, references_sacrebleu)
    assert round(bleu.score, 1) == expected_bleu
