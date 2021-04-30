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
"""Utilties for testing."""

from official.nlp.bert import configs

# Tokens used in tests.
_TOKENS = [
    "[PAD]", "[CLS]", "[SEP]", "[unused0]", "[unused1]", "foo", "bar", "what",
    "river", "traverses", "the", "most", "states", "and"
]


class MockTokenizer(object):
  """Mock tokenizer to replace `tokenization.FullTokenizer` in tests."""

  def __init__(self, **kwargs):
    del kwargs
    self.tokens_to_ids = {
        token: token_id for token_id, token in enumerate(_TOKENS)
    }

  def tokenize(self, input_str):
    return input_str.split()

  def convert_tokens_to_ids(self, tokens):
    return [self.tokens_to_ids[token] for token in tokens]


def get_test_config():
  return {
      "batch_size": 4,
      "learning_rate": 0.001,
      "training_steps": 10000,
      "warmup_steps": 100,
      "steps_per_iteration": 8,
      "model_dims": 16,
      "max_num_wordpieces": 8,
      "max_num_applications": 8,
      "max_num_numerator_nodes": 8,
      "max_num_denominator_nodes": 8,
      "max_num_rules": 8,
  }


def get_test_bert_config():
  return configs.BertConfig(
      vocab_size=32,
      hidden_size=8,
      intermediate_size=8,
      num_attention_heads=2,
      num_hidden_layers=2)
