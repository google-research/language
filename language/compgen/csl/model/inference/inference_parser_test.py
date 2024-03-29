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
"""Tests for inference_parser."""

from language.compgen.csl.model import test_utils
from language.compgen.csl.model.inference import inference_parser
from language.compgen.csl.model.inference import inference_wrapper
from language.compgen.csl.qcfg import qcfg_rule
import tensorflow as tf


class InferenceParserTest(tf.test.TestCase):

  def test_get_outputs(self):
    rules = [
        qcfg_rule.rule_from_string("foo NT_1 ### foo NT_1"),
        qcfg_rule.rule_from_string("bar ### bar"),
        qcfg_rule.rule_from_string("foo bar ### foo bar"),
    ]
    config = test_utils.get_test_config()

    wrapper = inference_wrapper.InferenceWrapper(rules, config)
    wrapper.compute_application_scores()

    source = "foo bar"
    outputs = inference_parser.run_inference(source, wrapper)
    print("outputs: %s" % outputs)

    self.assertIsNotNone(outputs)

if __name__ == "__main__":
  tf.test.main()
