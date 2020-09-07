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
"""BERT utils."""
from bert import tokenization
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
from tensorflow_text.python.ops import bert_tokenizer


def get_tokenization_info(module_handle):
  with tf.Graph().as_default():
    bert_module = hub.Module(module_handle)
    with tf.Session() as sess:
      return sess.run(bert_module(signature="tokenization_info", as_dict=True))


def get_tokenizer(module_handle):
  tokenization_info = get_tokenization_info(module_handle)
  return tokenization.FullTokenizer(
      vocab_file=tokenization_info["vocab_file"],
      do_lower_case=tokenization_info["do_lower_case"])


def get_tf_tokenizer(module_handle):
  """Creates a preprocessing function."""
  tokenization_info = get_tokenization_info(module_handle)

  table_initializer = tf.lookup.TextFileInitializer(
      filename=tokenization_info["vocab_file"],
      key_dtype=tf.string,
      key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
      value_dtype=tf.int64,
      value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  vocab_lookup_table = tf.lookup.StaticVocabularyTable(
      initializer=table_initializer,
      num_oov_buckets=1,
      lookup_key_dtype=tf.string)

  tokenizer = tf_text.BertTokenizer(
      vocab_lookup_table=vocab_lookup_table,
      lower_case=tokenization_info["do_lower_case"])

  return tokenizer, vocab_lookup_table


def tokenize_with_original_mapping(text_input, tokenizer):
  """Tokenize with original mapping."""
  # pylint:disable=protected-access
  text_input = tf.regex_replace(text_input, r"\p{Cc}|\p{Cf}", " ")
  orig_tokens = tf_text.regex_split(
      text_input, bert_tokenizer._DELIM_REGEX_PATTERN,
      tokenizer._basic_tokenizer._keep_delim_regex_pattern,
      "BertBasicTokenizer")
  normalized_tokens = orig_tokens
  normalized_text = text_input
  if tokenizer._basic_tokenizer._lower_case:

    def _do_lower_case(t):
      t = tf_text.case_fold_utf8(t)
      t = tf_text.normalize_utf8(t, "NFD")
      t = tf.regex_replace(t, r"\p{Mn}", "")
      return t

    normalized_tokens = _do_lower_case(normalized_tokens)
    normalized_text = _do_lower_case(normalized_text)

  wordpieces = tokenizer._wordpiece_tokenizer.tokenize(normalized_tokens)

  orig_token_map = tf.ragged.range(orig_tokens.row_lengths())
  orig_token_map = tf.expand_dims(orig_token_map, 2) + tf.zeros_like(wordpieces)
  # pylint:enable=protected-access

  wordpieces = wordpieces.merge_dims(1, 2)
  orig_token_map = orig_token_map.merge_dims(1, 2)

  return orig_tokens, orig_token_map, wordpieces, normalized_text


def pad_or_truncate(token_ids, sequence_length, cls_id, sep_id):
  token_ids = token_ids[:sequence_length - 2]
  truncated_len = tf.size(token_ids)
  padding = tf.zeros([sequence_length - 2 - truncated_len], tf.int32)
  token_ids = tf.concat([[cls_id], token_ids, [sep_id], padding], 0)
  mask = tf.concat([tf.ones([truncated_len + 2], tf.int32), padding], 0)
  token_ids = tf.ensure_shape(token_ids, [sequence_length])
  mask = tf.ensure_shape(mask, [sequence_length])
  return token_ids, mask


def pad_or_truncate_pair(token_ids_a, token_ids_b, sequence_length, cls_id,
                         sep_id):
  """Pad or truncate pair."""
  token_ids_a = token_ids_a[:sequence_length - 3]
  truncated_len_a = tf.size(token_ids_a)
  maximum_len_b = tf.maximum(sequence_length - 3 - truncated_len_a, 0)
  token_ids_b = token_ids_b[:maximum_len_b]
  truncated_len_b = tf.size(token_ids_b)
  truncated_len_pair = truncated_len_a + truncated_len_b
  padding = tf.zeros([sequence_length - 3 - truncated_len_pair], tf.int32)
  token_ids = tf.concat([
      [cls_id],
      token_ids_a,
      [sep_id],
      token_ids_b,
      [sep_id],
      padding], 0)
  mask = tf.concat([
      tf.ones([truncated_len_pair + 3], tf.int32),
      padding], 0)
  segment_ids = tf.concat([
      tf.zeros([truncated_len_a + 2], tf.int32),
      tf.ones([truncated_len_b + 1], tf.int32),
      padding], 0)
  token_ids = tf.ensure_shape(token_ids, [sequence_length])
  mask = tf.ensure_shape(mask, [sequence_length])
  segment_ids = tf.ensure_shape(segment_ids, [sequence_length])

  return token_ids, mask, segment_ids
