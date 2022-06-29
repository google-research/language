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
from absl.testing import absltest
from language.fruit import metrics
from language.fruit import rendering_utils
from t5.evaluation import test_utils


class EditRougeTest(test_utils.BaseMetricsTest):

  def test_completely_correct(self):
    targets = [{
        "inputs": "[0] Foo [1] Bar [2] Baz",
        "targets": "Completely replaced",
        "normalized_inputs": "Foo Bar Baz",
        "normalized_targets": "Completely replaced",
    }]
    predictions = [{
        "targets": "Completely replaced",
        "normalized_targets": "Completely replaced",
    }]
    observed = metrics.edit_rouge(targets, predictions)
    expected = {
        "update_rouge1": 100.0,
        "update_rouge2": 100.0,
        "update_rougeLsum": 100.0,
    }
    self.assertDictClose(observed, expected)

  def test_no_source_diff(self):
    targets = [{
        "inputs": "[0] Foo [1] Bar [2] Baz",
        "targets": "[0] [1] [2]",
        "normalized_inputs": "Foo Bar Baz",
        "normalized_targets": "Foo Bar Baz a",
    }]
    predictions = [{
        "targets": "[0] [1] [2]",
        "normalized_targets": "Foo Bar Baz a",
    }]
    observed = metrics.edit_rouge(targets, predictions)
    print(observed)
    expected = {
        "update_rouge1": 100.0,
        "update_rouge2": 100.0,
        "update_rougeLsum": 100.0,
    }
    self.assertDictClose(observed, expected)

  def test_completely_wrong(self):
    targets = [{
        "inputs": "[0] Foo [1] Bar [2] Baz",
        "targets": "replace everything.",
        "normalized_inputs": "Foo Bar Baz",
        "normalized_targets": "replace everything",
    }]
    predictions = [{
        "targets": "[0] [1] [2] ",
        "normalized_targets": "Foo Bar Baz",
    }]
    observed = metrics.edit_rouge(targets, predictions)
    expected = {
        "update_rouge1": 0.0,
        "update_rouge2": 0.0,
        "update_rougeLsum": 0.0,
    }
    self.assertDictClose(observed, expected)


class ExactMatchTest(test_utils.BaseMetricsTest):

  def test_completely_correct(self):
    targets = [{
        "normalized_targets": "hello world",
    }]
    predictions = [{
        "normalized_targets": "hello world",
    }]
    observed = metrics.exact_match(targets, predictions)
    expected = {
        "exact_match": 1.0,
    }
    self.assertDictClose(observed, expected)

  def test_completely_wrong(self):
    targets = [{
        "normalized_targets": "hello world",
    }]
    predictions = [{
        "normalized_targets": "goodbye world",
    }]
    observed = metrics.exact_match(targets, predictions)
    expected = {
        "exact_match": 0.0,
    }
    self.assertDictClose(observed, expected)


class SurfaceRecallTest(test_utils.BaseMetricsTest):

  def test_completely_correct(self):
    # Test that recall is 100% even if surface form of Rush is different in the
    # input and prediction.
    targets = [{
        "inputs": "[0] Foo [Context] (1) Yes rush",
        "generatable_surfaces": {
            "e0": ["Yes"],
            "e1": ["rush", "Rush"]
        },
    }]
    predictions = [{
        "targets": "I like Yes and Rush.",
    }]
    observed = metrics.surface_recall(targets, predictions)
    expected = {
        "surface_recall": 1.0,
        "filtered_surface_recall": 1.0,
    }
    self.assertDictClose(observed, expected)

  def test_filtered_correct(self):
    # Simulate the case where Rush mention gets truncated. In this case surface
    # recall will decrease but filtered surface should still be 100%.
    targets = [{
        "inputs": "[0] Foo [CONTEXT] (0) Yes",
        "generatable_surfaces": {
            "e0": ["Yes"],
            "e1": ["rush", "Rush"]
        },
    }]
    predictions = [{
        "targets": "I like Yes.",
    }]
    observed = metrics.surface_recall(targets, predictions)
    expected = {
        "surface_recall": 0.5,
        "filtered_surface_recall": 1.0,
    }
    self.assertDictClose(observed, expected)


class DelimiterF1Test(test_utils.BaseMetricsTest):

  def setUp(self):
    super().setUp()
    self.delimiter_range_pair = rendering_utils.get_default_delimiter_range_pair(
        task=rendering_utils.Task.diff,
        delimiter_type=rendering_utils.DelimiterType.text,
    )

  def test_completely_correct(self):
    targets = [{"targets": "[0] (0) Replace 1 [2]"}]
    predictions = [{"targets": "[0] (0) Does not affect score [2]"}]
    observed = metrics.delimiter_f1(
        targets,
        predictions,
        self.delimiter_range_pair,
    )
    expected = {
        "retention_p": 1.0,
        "retention_r": 1.0,
        "retention_f1": 1.0,
        "reference_p": 1.0,
        "reference_r": 1.0,
        "reference_f1": 1.0,
    }
    self.assertDictClose(observed, expected)

  def test_completely_wrong(self):
    targets = [{"targets": "[0] (0) Replace 1 [2]"}]
    predictions = [{"targets": "Filler [1] (1) Filler"}]
    observed = metrics.delimiter_f1(
        targets,
        predictions,
        self.delimiter_range_pair,
    )
    expected = {
        "retention_p": 0.0,
        "retention_r": 0.0,
        "retention_f1": 0.0,
        "reference_p": 0.0,
        "reference_r": 0.0,
        "reference_f1": 0.0,
    }
    self.assertDictClose(observed, expected)

  def test_somewhere_in_between(self):
    targets = [{"targets": "[0] (0) Replace 1 [2] Replace 3"}]
    predictions = [{"targets": "[0] (0) (1) Replace everything else"}]
    observed = metrics.delimiter_f1(
        targets,
        predictions,
        self.delimiter_range_pair,
    )
    expected = {
        "retention_p": 1 / 1,
        "retention_r": 1 / 2,
        "retention_f1": 1 / 1.5,
        "reference_p": 1 / 2,
        "reference_r": 1 / 1,
        "reference_f1": 1 / 1.5,
    }
    self.assertDictClose(observed, expected)

  def test_works_on_delimiterless_text(self):
    targets = [{"targets": "Foo Bar Baz"}]
    predictions = [{"targets": "Baz Bar Foo"}]
    observed = metrics.delimiter_f1(
        targets,
        predictions,
        self.delimiter_range_pair,
    )
    expected = {
        "retention_p": 0.0,
        "retention_r": 0.0,
        "retention_f1": 0.0,
        "reference_p": 0.0,
        "reference_r": 0.0,
        "reference_f1": 0.0,
    }
    self.assertDictClose(observed, expected)


class EscapeMdTest(absltest.TestCase):

  def test_escape_md(self):
    all_special_chars = r"\\`*_{}[]()#+-.!"
    observed = metrics._escape_md(all_special_chars)
    expected = r"\\\\\`\*\_\{\}\[\]\(\)\#\+\-\.\!"
    self.assertEqual(observed, expected)


class PrintPredictionsTest(test_utils.BaseMetricsTest):

  def test_print_predictions(self):
    targets = [{"targets": "example", "inputs": "hi"}]
    predictions = [{"targets": "example"}]
    observed = metrics.print_predictions(targets, predictions)
    expected_correct_predictions = (
        "## Example 0\n\n**Input**\n\nhi\n\n**Target**\n\nexample\n\n"
        "**Prediction**\n\nexample\n\n")
    expected_incorrect_predictions = ""
    self.assertEqual(observed["correct_predictions"].textdata,
                     expected_correct_predictions)
    self.assertEqual(observed["incorrect_predictions"].textdata,
                     expected_incorrect_predictions)


if __name__ == "__main__":
  absltest.main()
