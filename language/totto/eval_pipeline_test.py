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
import os

from absl.testing import absltest
from language.totto import prepare_references_for_eval
from language.totto import totto_parent_eval
import sacrebleu
import six


class TestEval(absltest.TestCase):
  """Test class for reference formatting and BLEU scoring."""

  def _get_abs_path(self, relative_path):
    """Get absolute path."""
    curr_path = os.path.abspath(__file__)
    curr_dir, _ = os.path.split(curr_path)
    return os.path.join(curr_dir, relative_path)

  def _format_refs(self, input_path):
    """helper function to get the multi_reference for a file."""
    references = []
    with open(input_path, "r", encoding="utf-8") as input_file:
      for line in input_file:
        line = six.ensure_text(line, "utf-8")
        json_example = json.loads(line)
        multi_reference, multi_overlap_reference, multi_nonoverlap_reference = (
            prepare_references_for_eval.get_references(json_example, "dev"))
        del multi_overlap_reference, multi_nonoverlap_reference
        references.append([r.lower() for r in multi_reference])
    return references

  def _format_tables(self, input_path):
    """helper function to get the tables in parent format for a file."""
    prec_tables = []
    rec_tables = []
    with open(input_path, "r", encoding="utf-8") as input_file:
      for line in input_file:
        line = six.ensure_text(line, "utf-8")
        json_example = json.loads(line)
        (table_prec, table_rec, overlap_table_prec, overlap_table_rec,
         nonoverlap_table_prec, nonoverlap_table_rec) = (
             prepare_references_for_eval.get_parent_tables(json_example, "dev"))
        del (overlap_table_prec, overlap_table_rec, nonoverlap_table_prec,
             nonoverlap_table_rec)

        prec_entries = table_prec.lower().split("\t")
        rec_entries = table_rec.lower().split("\t")

        # pylint: disable=g-complex-comprehension
        prec_table_tokens = [[
            totto_parent_eval._normalize_text(member).split()
            for member in entry.split("|||")
        ]
                             for entry in prec_entries]

        rec_table_tokens = [[
            totto_parent_eval._normalize_text(member).split()
            for member in entry.split("|||")
        ]
                            for entry in rec_entries]

        prec_tables.append(prec_table_tokens)
        rec_tables.append(rec_table_tokens)
    return prec_tables, rec_tables

  def test_ref_format(self):
    """Tests whether the references are returned as expected."""
    input_path = self._get_abs_path("sample/dev_sample.jsonl")
    references = self._format_refs(input_path)
    final_example = ("the nashville (2012 tv series) premiered on october 10, "
                     "2012 had 8.93 million viewers.")
    # Ensure that the final example is correct.
    assert references[-1][-1] == final_example

  def test_bleu_eval(self):
    """Tests whether we are seeing the expected BLEU score."""
    input_path = self._get_abs_path("sample/dev_sample.jsonl")
    references = self._format_refs(input_path)
    # Sacrebleu expects dimension transpose (1 list per reference count).
    references_sacrebleu = [list(x) for x in zip(*references)]
    output_path = self._get_abs_path("sample/output_sample.txt")
    with open(output_path, "r", encoding="utf-8") as f:
      predictions = [p.strip().lower() for p in f]

    expected_bleu = 45.5
    bleu = sacrebleu.corpus_bleu(predictions, references_sacrebleu)
    assert round(bleu.score, 1) == expected_bleu

  def test_parent_eval(self):
    input_path = self._get_abs_path("sample/dev_sample.jsonl")
    references = self._format_refs(input_path)
    reference_tokens = []
    for multi_ref in references:
      multi_ref_tokens = [r.split() for r in multi_ref]
      reference_tokens.append(multi_ref_tokens)
    prec_tables, rec_tables = self._format_tables(input_path)

    output_path = self._get_abs_path("sample/output_sample.txt")
    with open(output_path, "r", encoding="utf-8") as f:
      prediction_tokens = [p.strip().lower().split() for p in f]

    precision, recall, f_score, all_f_scores = (
        totto_parent_eval.parent(
            predictions=prediction_tokens,
            references=reference_tokens,
            precision_tables=prec_tables,
            recall_tables=rec_tables,
            lambda_weight=None))

    assert round(precision, 1) == 66.1
    assert round(recall, 1) == 40.4
    assert round(f_score, 1) == 48.1

    all_f_scores = [round(f, 1) for f in all_f_scores]
    assert all_f_scores == [68.4, 63.2, 51.9, 7.8, 48.9]


if __name__ == "__main__":
  absltest.main()
