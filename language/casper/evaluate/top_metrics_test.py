# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for complete, aligned TOP metric inputs."""

import os
import pathlib
import subprocess
import sys
import tempfile

from absl.testing import absltest, parameterized

from language.casper.evaluate import top_metrics
from language.casper.utils import top_utils

_PARSE = '[IN:SET_ALARM [SL:DATE_TIME seven ] ]'
_OTHER_VALUE = '[IN:SET_ALARM [SL:DATE_TIME eight ] ]'
_OTHER_INTENT = '[IN:GET_WEATHER [SL:DATE_TIME tomorrow ] ]'


class TopMetricsTest(parameterized.TestCase):

  @parameterized.product(
      sizes=[(2, 1), (1, 2), (1, 0), (0, 1)],
      error_on_invalid_target=[False, True],
  )
  def test_unequal_lengths_raise(self, sizes, error_on_invalid_target):
    targets = [_PARSE] * sizes[0]
    predictions = [_PARSE] * sizes[1]
    with self.assertRaisesRegex(ValueError, 'Unequal number of parses'):
      top_metrics.top_metrics(targets, predictions, error_on_invalid_target)
    self.assertEqual(targets, [_PARSE] * sizes[0])
    self.assertEqual(predictions, [_PARSE] * sizes[1])

  def test_mixed_predictions_keep_existing_metrics(self):
    actual = top_metrics.top_metrics(
        [_PARSE] * 4, [_PARSE, _OTHER_VALUE, _OTHER_INTENT, 'invalid']
    )
    self.assertEqual(actual, {
        'num_total': 4, 'full_accuracy': 0.25, 'intent_accuracy': 0.5,
        'intent_arg_accuracy': 0.5, 'invalid_predictions': 0.25,
    })

  def test_matching_empty_inputs(self):
    self.assertEqual(top_metrics.top_metrics([], []), {
        'num_total': 0, 'full_accuracy': 0.0, 'intent_accuracy': 0.0,
        'intent_arg_accuracy': 0.0, 'invalid_predictions': 0.0,
    })

  @parameterized.parameters(list, tuple)
  def test_aligned_sequences(self, sequence_type):
    values = sequence_type([_PARSE, _OTHER_INTENT])
    result = top_metrics.top_metrics(values, values)
    self.assertEqual(result, {
        'num_total': 2, 'full_accuracy': 1.0, 'intent_accuracy': 1.0,
        'intent_arg_accuracy': 1.0, 'invalid_predictions': 0.0,
    })

  def test_formatted_prediction_has_the_same_frame(self):
    result = top_metrics.top_metrics(
        [_PARSE], [top_utils.format_serialized(_PARSE)]
    )
    self.assertEqual(result['full_accuracy'], 0.0)
    self.assertEqual(result['intent_accuracy'], 1.0)
    self.assertEqual(result['intent_arg_accuracy'], 1.0)

  def test_invalid_target_still_raises_when_requested(self):
    with self.assertRaises(AssertionError):
      top_metrics.top_metrics(['invalid'], [_PARSE])

  def test_invalid_target_policy_is_unchanged(self):
    result = top_metrics.top_metrics(
        ['invalid', _PARSE], [_PARSE, _PARSE], error_on_invalid_target=False
    )
    self.assertEqual(result['num_total'], 2)
    self.assertEqual(result['full_accuracy'], 0.5)
    self.assertEqual(result['intent_accuracy'], 0.5)
    self.assertEqual(result['intent_arg_accuracy'], 0.5)

  @parameterized.parameters(True, False)
  def test_existing_cli_alignment_check(self, aligned):
    with tempfile.TemporaryDirectory() as directory:
      gold = pathlib.Path(directory) / 'gold.tsv'
      pred = pathlib.Path(directory) / 'pred.tsv'
      gold.write_text('query\t' + _PARSE + '\n', encoding='utf-8')
      pred.write_text((_PARSE + '\n') if aligned else '', encoding='utf-8')
      proc = subprocess.run([
          sys.executable, '-m',
          'language.casper.evaluate.evaluate_mtop_predictions',
          '--gold_file=' + str(gold), '--pred_file=' + str(pred),
          '--alsologtostderr',
      ], capture_output=True, text=True, timeout=60, check=False,
          env=os.environ.copy())
    if aligned:
      self.assertEqual(proc.returncode, 0, proc.stderr)
      self.assertIn('Exact match accuracy: 100.00', proc.stderr)
    else:
      self.assertNotEqual(proc.returncode, 0)
      self.assertIn('Unequal number of parses', proc.stderr)
      self.assertNotIn('Exact match accuracy:', proc.stderr)


if __name__ == '__main__':
  absltest.main()
