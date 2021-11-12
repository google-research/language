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
"""Tests for casper_converters."""

from absl.testing import absltest
from language.casper.augment import casper_converters

_RETRIEVAL_INDEX = [
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

_INPUT_OUTPUT_STRS = {
    entry["hashed_id"]: entry["input_str"] + " ## " + entry["_serialized"]
    for entry in _RETRIEVAL_INDEX
}

_FORMATTER_KWARGS = {"presets": ["punc"]}


class ExampleConvertersTest(absltest.TestCase):

  def test_invalid_arguments(self):
    """Tests if get_converter throws an error on invalid arguments."""
    with self.assertRaises(KeyError):
      casper_converters.get_converter("unknown_converter", _RETRIEVAL_INDEX,
                                      "top", {}, _FORMATTER_KWARGS)
    with self.assertRaises(TypeError):
      casper_converters.get_converter("add_top", _RETRIEVAL_INDEX, "top",
                                      {"unknown_arg": 42}, _FORMATTER_KWARGS)
    with self.assertRaises(ValueError):
      casper_converters.get_converter("add_top", _RETRIEVAL_INDEX,
                                      "unknown_funcall_format", {"k": 3},
                                      _FORMATTER_KWARGS)

  def test_query_only_converter(self):
    """Tests the query_only converter."""
    converter = casper_converters.get_converter("query_only", _RETRIEVAL_INDEX,
                                                "top", {}, _FORMATTER_KWARGS)

    raw_ex = {
        "input_str": "snooze 5 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 5 minutes ] ]"
    }
    results = list(converter.convert(raw_ex))
    self.assertEqual(results,
                     [("snooze 5 minutes",
                       "[IN snooze alarm = [SL snooze duration = 5 minutes]]")])

  def test_add_top_1_converter(self):
    """Tests the add_top converter."""
    converter = casper_converters.get_converter("add_top", _RETRIEVAL_INDEX,
                                                "top", {}, _FORMATTER_KWARGS)

    raw_ex = {
        "input_str": "snooze 5 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 5 minutes ] ]",
        "exemplars": {
            "hashed_ids": ["aaa", "ccc"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertEqual(results,
                     [("snooze 5 minutes @@ " + _INPUT_OUTPUT_STRS["aaa"],
                       "[IN snooze alarm = [SL snooze duration = 5 minutes]]")])

  def test_add_top_3_converter(self):
    """Tests the add_top converter."""
    converter = casper_converters.get_converter("add_top", _RETRIEVAL_INDEX,
                                                "top", {"k": 3},
                                                _FORMATTER_KWARGS)

    raw_ex = {
        "input_str": "snooze 5 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 5 minutes ] ]",
        "exemplars": {
            "hashed_ids": ["aaa", "ddd", "eee", "ccc", "bbb"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertEqual(
        results,
        [("snooze 5 minutes @@ " + _INPUT_OUTPUT_STRS["aaa"] + " @@ " +
          _INPUT_OUTPUT_STRS["ddd"] + " @@ " + _INPUT_OUTPUT_STRS["eee"],
          "[IN snooze alarm = [SL snooze duration = 5 minutes]]")])

  def test_add_sampled_3_converter(self):
    """Tests the add_sampled converter."""
    # When p = 1.0, the behavior is identical to add_top_k
    converter = casper_converters.get_converter("add_samp", _RETRIEVAL_INDEX,
                                                "top", {
                                                    "n": 2,
                                                    "k": 3,
                                                    "p": 1.0
                                                }, _FORMATTER_KWARGS)

    raw_ex = {
        "input_str": "snooze 5 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 5 minutes ] ]",
        "exemplars": {
            "hashed_ids": ["aaa", "ddd", "eee", "ccc", "bbb"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertLen(results, 2)
    self.assertEqual(
        results[0],
        ("snooze 5 minutes @@ " + _INPUT_OUTPUT_STRS["aaa"] + " @@ " +
         _INPUT_OUTPUT_STRS["ddd"] + " @@ " + _INPUT_OUTPUT_STRS["eee"],
         "[IN snooze alarm = [SL snooze duration = 5 minutes]]"))
    self.assertEqual(results[1], results[0])

  def test_add_oracle_converter(self):
    """Tests the add_oracle converter."""
    # When p = 1.0, the behavior is identical to add_top_k. But with add_oracle,
    # only the exemplars with the same frame as the target should be selected.
    converter = casper_converters.get_converter("add_oracle", _RETRIEVAL_INDEX,
                                                "top", {
                                                    "n": 2,
                                                    "k": 2,
                                                    "p": 1.0,
                                                }, _FORMATTER_KWARGS)

    raw_ex = {
        "hashed_id": "qqq",
        "input_str": "snooze 5 minutes",
        "output_str": "[IN:SNOOZE_ALARM [SL:SNOOZE_DURATION 5 minutes ] ]",
        "exemplars": {
            "hashed_ids": ["aaa", "ddd", "eee", "ccc", "bbb"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertLen(results, 2)
    self.assertEqual(results[0],
                     ("snooze 5 minutes @@ " + _INPUT_OUTPUT_STRS["aaa"] +
                      " @@ " + _INPUT_OUTPUT_STRS["ccc"],
                      "[IN snooze alarm = [SL snooze duration = 5 minutes]]"))
    self.assertEqual(results[1], results[0])

  def test_add_oracle_special_cases_converter(self):
    """Tests the add_oracle converter on special cases."""
    converter = casper_converters.get_converter("add_oracle", _RETRIEVAL_INDEX,
                                                "top", {
                                                    "n": 2,
                                                    "k": 2,
                                                    "p": 1.0,
                                                }, _FORMATTER_KWARGS)

    # raw_ex is the index entry "aaa". Even when `max_exemplars` is 2, "aaa"
    # itself should not be chosen for augmentation. Note that "ccc" will be
    # chosen even though it is not among the retrievals.
    raw_ex = dict(_RETRIEVAL_INDEX[0])
    raw_ex["exemplars"] = {"hashed_ids": ["bbb", "ddd"]}
    results = list(converter.convert(raw_ex))
    self.assertLen(results, 2)
    self.assertEqual(results[0],
                     ("give me 30 seconds @@ " + _INPUT_OUTPUT_STRS["ccc"],
                      "[IN snooze alarm = [SL snooze duration = 30 seconds]]"))
    self.assertEqual(results[1], results[0])

  def test_add_oracle_fail_converter(self):
    """Tests the add_oracle converter when there are no good index entries."""
    converter = casper_converters.get_converter("add_oracle", _RETRIEVAL_INDEX,
                                                "top", {
                                                    "n": 2,
                                                    "k": 2,
                                                    "p": 1.0,
                                                }, _FORMATTER_KWARGS)

    # The index only has Create_timer with duration.
    raw_ex = {
        "hashed_id": "qqq",
        "input_str": "create a timer",
        "output_str": "[IN:CREATE_TIMER ]",
        "exemplars": {
            "hashed_ids": ["aaa", "ddd", "eee", "ccc", "bbb"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertEmpty(results)

  def test_add_adversarial_converter(self):
    """Tests the add_adversarial converter."""
    converter = casper_converters.get_converter("add_adversarial",
                                                _RETRIEVAL_INDEX, "top", {
                                                    "n": 2,
                                                    "k": 2,
                                                    "sampler": "uniform",
                                                }, _FORMATTER_KWARGS)

    # "aaa" and "ccc" should be chosen, since their frame is the only frame
    # with >= 2 index entries that is different from the gold frame.
    raw_ex = {
        "input_str": "create a timer for 4 minutes",
        "output_str": "[IN:CREATE_TIMER [SL:DURATION 4 minutes ] ]",
        "exemplars": {
            "hashed_ids": ["eee", "ccc", "aaa", "ddd", "bbb"]
        },
    }
    results = list(converter.convert(raw_ex))
    self.assertLen(results, 2)
    for input_str, output_str in results:
      self.assertSetEqual(
          set(input_str.split(" @@ ")), {
              "create a timer for 4 minutes", _INPUT_OUTPUT_STRS["aaa"],
              _INPUT_OUTPUT_STRS["ccc"]
          })
      self.assertEqual(output_str,
                       "[IN create timer = [SL duration = 4 minutes]]")


if __name__ == "__main__":
  absltest.main()
