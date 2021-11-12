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
"""Tests for casper_formatters."""

from absl.testing import absltest
from language.casper.augment import casper_formatters

_EXAMPLES = [
    {
        "hashed_id": "aaa",
        "input_str": "give me 30 seconds",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 30 seconds ] ]",
        "_serialized": "[IN snooze alarm = [SL snooze duration = 30 seconds]]",
    },
    {
        "hashed_id": "bbb",
        "input_str": "give me 10 minutes",
        "output_str": "[IN:CREATE_TIMER [SL:DURATION 10 minutes ] ]",
        "_serialized": "[IN create timer = [SL duration = 10 minutes]]",
    },
    {
        "hashed_id": "ccc",
        "input_str": "sleep 7 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 7 minutes ] ]",
        "_serialized": "[IN snooze alarm = [SL snooze duration = 7 minutes]]",
    },
    {
        "hashed_id": "ddd",
        "input_str": "snooze",
        "output_str": "[IN:SNOOZE_ALARM ]",
        "_serialized": "[IN snooze alarm =]",
    },
    {
        "hashed_id": "eee",
        "input_str": "set a timer for 1 hour",
        "output_str": "[IN:CREATE_TIMER [SL:DURATION 1 hour ] ]",
        "_serialized": "[IN create timer = [SL duration = 1 hour]]",
    },
]


class FormattersTest(absltest.TestCase):

  def test_punc_prompt_formatter(self):
    """Tests the punc prompt format."""
    config = casper_formatters.FormatterConfig.from_dict({"presets": ["punc"]})
    input_str, output_str = casper_formatters.augment_exemplars(
        _EXAMPLES[0], [_EXAMPLES[1], _EXAMPLES[3]], "top", config)
    expected_input_str = "{} @@ {} ## {} @@ {} ## {}".format(
        _EXAMPLES[0]["input_str"],
        _EXAMPLES[1]["input_str"],
        _EXAMPLES[1]["_serialized"],
        _EXAMPLES[3]["input_str"],
        _EXAMPLES[3]["_serialized"],
    )
    expected_output_str = _EXAMPLES[0]["_serialized"]
    self.assertEqual(input_str, expected_input_str)
    self.assertEqual(output_str, expected_output_str)

  def test_punc_inv_prompt_formatter(self):
    """Tests the punc prompt format with original input at the end."""
    config = casper_formatters.FormatterConfig.from_dict({
        "presets": ["punc"],
        "inv": True
    })
    input_str, output_str = casper_formatters.augment_exemplars(
        _EXAMPLES[0], [_EXAMPLES[1], _EXAMPLES[3]], "top", config)
    expected_input_str = "{} ## {} @@ {} ## {} @@ {}".format(
        _EXAMPLES[1]["input_str"],
        _EXAMPLES[1]["_serialized"],
        _EXAMPLES[3]["input_str"],
        _EXAMPLES[3]["_serialized"],
        _EXAMPLES[0]["input_str"],
    )
    expected_output_str = _EXAMPLES[0]["_serialized"]
    self.assertEqual(input_str, expected_input_str)
    self.assertEqual(output_str, expected_output_str)

  def test_verbal_prompt_formatter(self):
    """Tests the verbal prompt format."""
    config = casper_formatters.FormatterConfig.from_dict(
        {"presets": ["verbal"]})
    input_str, output_str = casper_formatters.augment_exemplars(
        _EXAMPLES[0], [_EXAMPLES[1], _EXAMPLES[3]], "top", config)
    expected_input_str = ("input: {} @@ example 1: {} ## output 1: {} "
                          "@@ example 2: {} ## output 2: {}").format(
                              _EXAMPLES[0]["input_str"],
                              _EXAMPLES[1]["input_str"],
                              _EXAMPLES[1]["_serialized"],
                              _EXAMPLES[3]["input_str"],
                              _EXAMPLES[3]["_serialized"],
                          )
    expected_output_str = _EXAMPLES[0]["_serialized"]
    self.assertEqual(input_str, expected_input_str)
    self.assertEqual(output_str, expected_output_str)

  def test_platinum_prompt_formatter(self):
    """Tests the adding of PLATINUM tokens."""
    config = casper_formatters.FormatterConfig.from_dict(
        {"presets": ["verbal", "plat"]})
    input_str, output_str = casper_formatters.augment_exemplars(
        _EXAMPLES[0], [_EXAMPLES[1], _EXAMPLES[3]], "top", config)
    expected_input_str = ("input: {} @@ PLATINUM example 1: {} ## output 1: {} "
                          "@@ PLATINUM example 2: {} ## output 2: {}").format(
                              _EXAMPLES[0]["input_str"],
                              _EXAMPLES[1]["input_str"],
                              _EXAMPLES[1]["_serialized"],
                              _EXAMPLES[3]["input_str"],
                              _EXAMPLES[3]["_serialized"],
                          )
    expected_output_str = _EXAMPLES[0]["_serialized"]
    self.assertEqual(input_str, expected_input_str)
    self.assertEqual(output_str, expected_output_str)


_MUSIC_EXAMPLES = [
    "[IN add to playlist = [SL music item = track] [SL playlist = metal " +
    "talks metallica]]",
    "[IN add to playlist = [SL artist = dean martin] [SL music item = track] " +
    "[SL playlist = metal xplorer]]",
    "[IN add to playlist = [SL playlist owner = my] [SL artist = norma jean] " +
    "[SL music item = tune]]",
]


class TopFuncallProcessorTest(absltest.TestCase):

  def test_rename_labels(self):
    """Tests rename on the labels."""
    # Rename the MUSIC_ITEM slot and drop the PLAYLIST slot.
    config = casper_formatters.FormatterConfig.from_dict(
        {"rename_labels": {
            "SL:MUSIC_ITEM": "SL:THING",
            "SL:PLAYLIST": "",
        }})
    exemplar_outputs, orig_output = casper_formatters.top_funcall_processor(
        [_MUSIC_EXAMPLES[0], _MUSIC_EXAMPLES[1]], _MUSIC_EXAMPLES[2], config)
    self.assertEqual(exemplar_outputs, [
        "[IN add to playlist = [SL thing = track]]",
        "[IN add to playlist = [SL artist = dean martin] [SL thing = track]]"
    ])
    self.assertEqual(
        orig_output,
        "[IN add to playlist = [SL playlist owner = my] [SL artist = norma jean] [SL thing = tune]]"
    )

  def test_anonymize(self):
    """Tests anonymization."""
    config = casper_formatters.FormatterConfig.from_dict({"anonymize": True})
    exemplar_outputs, orig_output = casper_formatters.top_funcall_processor(
        [_MUSIC_EXAMPLES[0], _MUSIC_EXAMPLES[1]], _MUSIC_EXAMPLES[2], config)
    self.assertLen(exemplar_outputs, 2)
    results_str = exemplar_outputs[0] + exemplar_outputs[1] + orig_output
    self.assertRegex(
        results_str,
        r"\[IN (\d+) = \[SL (\d+) = track\] \[SL (\d+) = metal talks metallica\]\]"
        r"\[IN \1 = \[SL (\d+) = dean martin\] \[SL \2 = track\] \[SL \3 = metal xplorer\]\]"
        r"\[IN \1 = \[SL (\d+) = my\] \[SL \4 = norma jean\] \[SL \2 = tune\]\]"
    )


if __name__ == "__main__":
  absltest.main()
