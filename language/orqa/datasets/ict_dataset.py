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
"""Inverse Cloze Task dataset."""
import functools
from language.orqa.utils import bert_utils
import numpy as np
import tensorflow.compat.v1 as tf


def get_retrieval_examples(serialized_example, mask_rate, bert_hub_module_path,
                           query_seq_len, block_seq_len):
  """Make retrieval examples."""
  feature_spec = dict(
      title_ids=tf.FixedLenSequenceFeature([], tf.int64, True),
      token_ids=tf.FixedLenSequenceFeature([], tf.int64, True),
      sentence_starts=tf.FixedLenSequenceFeature([], tf.int64, True))
  features = tf.parse_single_example(serialized_example, feature_spec)
  features = {k: tf.cast(v, tf.int32) for k, v in features.items()}

  title_ids = features["title_ids"]
  token_ids = features["token_ids"]
  sentence_starts = features["sentence_starts"]
  sentence_ends = tf.concat([sentence_starts[1:], [tf.size(token_ids)]], 0)

  tokenizer = bert_utils.get_tokenizer(bert_hub_module_path)
  cls_id, sep_id = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

  # Randomly choose a sentence and pretend that it is a query.
  query_index = tf.random.uniform(
      shape=[],
      minval=0,
      maxval=tf.size(sentence_starts),
      dtype=tf.int32)
  query_start = sentence_starts[query_index]
  query_end = sentence_ends[query_index]

  query_ids = token_ids[query_start:query_end]

  mask_query = tf.less(tf.random.uniform([]), mask_rate)

  def _apply_mask():
    return tf.concat([token_ids[:query_start], token_ids[query_end:]], 0)

  block_ids = tf.cond(
      pred=mask_query,
      true_fn=_apply_mask,
      false_fn=lambda: token_ids)

  query_ids, query_mask = bert_utils.pad_or_truncate(
      token_ids=query_ids,
      sequence_length=query_seq_len,
      cls_id=cls_id,
      sep_id=sep_id)
  block_ids, block_mask, block_segment_ids = bert_utils.pad_or_truncate_pair(
      token_ids_a=title_ids,
      token_ids_b=block_ids,
      sequence_length=block_seq_len,
      cls_id=cls_id,
      sep_id=sep_id)

  # Masked examples for single-sentence blocks don't make any sense.
  keep_example = tf.logical_or(
      tf.logical_not(mask_query),
      tf.greater(tf.size(sentence_starts), 1))

  return dict(
      keep_example=keep_example,
      mask_query=mask_query,
      query_ids=query_ids,
      query_mask=query_mask,
      block_ids=block_ids,
      block_mask=block_mask,
      block_segment_ids=block_segment_ids)


def perturbed_chunks(max_val, num_chunks):
  """Perturbed chunks."""
  indices, chunk_size = np.linspace(
      start=0,
      stop=max_val,
      num=num_chunks,
      endpoint=False,
      retstep=True,
      dtype=np.int64)
  perturbation = np.random.randint(chunk_size, size=indices.shape)
  return indices + perturbation


def get_dataset(examples_path, mask_rate, bert_hub_module_path, query_seq_len,
                block_seq_len, num_block_records, num_input_threads):
  """An input function satisfying the tf.estimator API."""
  # The input file is not sharded. We can still get the randomization and
  # efficiency benefits of sharded inputs by doing multiple reads concurrently
  # but starting at different points.
  skips = perturbed_chunks(num_block_records, num_input_threads)
  tf.logging.info("Concurrent reads of %d records: %s",
                  num_block_records, skips)
  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(skips, tf.int64))

  def _skipped_dataset(skip):
    """Get skipped dataset."""
    dataset = tf.data.TFRecordDataset(
        examples_path, buffer_size=16 * 1024 * 1024)
    dataset = dataset.repeat()
    dataset = dataset.skip(skip)
    dataset = dataset.map(
        functools.partial(
            get_retrieval_examples,
            mask_rate=mask_rate,
            bert_hub_module_path=bert_hub_module_path,
            query_seq_len=query_seq_len,
            block_seq_len=block_seq_len))
    dataset = dataset.filter(lambda d: d.pop("keep_example"))
    return dataset

  dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      _skipped_dataset,
      sloppy=True,
      cycle_length=num_input_threads))
  return dataset
