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
"""Tests for data utils."""

import itertools
from typing import Dict, List

from absl.testing import absltest
from absl.testing import parameterized
from language.mentionmemory.utils import mention_preprocess_utils
from language.mentionmemory.utils import test_utils
import numpy as np
import tensorflow.compat.v2 as tf


def _get_test_mask_mentions_and_tokens_iterable():

  mask_rates = [0.0, 0.2, 1.0]
  mention_mask_rates = [0.0, 0.2, 1.0]
  text_lens_0 = [80, 100]
  text_lens_1 = [0, 1]
  n_mentions_list_0 = [10]
  n_mentions_list_1 = [0]

  return itertools.chain(
      itertools.product(mask_rates, mention_mask_rates, text_lens_0,
                        n_mentions_list_0),
      itertools.product(mask_rates, mention_mask_rates, text_lens_1,
                        n_mentions_list_1))


def _get_test_preprocess_mention_targets_iterable():

  max_mention_targets_list = [0, 5, 10]
  text_lens_0 = [80, 100]
  text_lens_1 = [0, 1]
  n_mentions_list_0 = [5, 10]
  n_mentions_list_1 = [0]

  return itertools.chain(
      itertools.product(max_mention_targets_list, text_lens_0,
                        n_mentions_list_0),
      itertools.product(max_mention_targets_list, text_lens_1,
                        n_mentions_list_1))


def _get_test_process_batchwise_mention_targets_iterable():

  batch_size_list = [1, 2, 3]
  max_mention_targets_list = [0, 5, 10]
  text_lens_0 = [80, 100]
  text_lens_1 = [0, 1]
  n_mentions_list_0 = [5, 10]
  n_mentions_list_1 = [0]
  max_mentions_list = [1, 5, 10]

  return itertools.chain(
      itertools.product(batch_size_list, text_lens_0, n_mentions_list_0,
                        max_mentions_list, max_mention_targets_list),
      itertools.product(batch_size_list, text_lens_1, n_mentions_list_1,
                        max_mentions_list, max_mention_targets_list))


def _get_test_add_entity_tokens():
  batch_size_list = [1, 2, 10]
  new_length_list = [50, 75, 100, 125, 150, 200]
  p_mention_to_mask_list = [0, 0.5, 1.0]

  test_list = itertools.product(batch_size_list, new_length_list,
                                p_mention_to_mask_list)
  test_list = [tuple([index] + list(x)) for index, x in enumerate(test_list)]
  return test_list


class MentionPreprocessUtilsTest(tf.test.TestCase, test_utils.TestCase):
  """Tests for mention-specific preprocessing."""

  max_len = 150
  max_mlm_targets = 20
  mention_size = 5
  max_mentions = 10
  mask_token_id = 200
  vocab_size = 100

  def _gen_input(self, text_len: int, n_mentions: int):
    """Generate synthetic inputs to test masking."""
    text_ids = np.pad(
        np.random.randint(1, self.vocab_size, size=text_len),
        (0, self.max_len - text_len),
        mode='constant')
    text_ids = text_ids.astype(np.int64)
    text_mask = np.pad(
        np.ones(text_len, dtype=np.int64), (0, self.max_len - text_len),
        mode='constant')
    mention_start_positions = np.random.choice(
        max((text_len - self.mention_size) // self.mention_size, 0),
        size=n_mentions,
        replace=False) * self.mention_size
    mention_start_positions.sort()
    mention_end_positions = mention_start_positions + self.mention_size - 1
    mention_end_positions = np.minimum(mention_end_positions, self.max_len - 1)
    dense_span_starts = np.zeros_like(text_ids)
    dense_span_starts[mention_start_positions] = 1
    dense_span_ends = np.zeros_like(text_ids)
    dense_span_ends[mention_end_positions] = 1
    dense_mention_mask = np.zeros(self.max_len, dtype=np.int64)
    dense_mention_ids = np.zeros(self.max_len, dtype=np.int64)
    for idx in range(n_mentions):
      mention_slice = slice(mention_start_positions[idx],
                            mention_end_positions[idx] + 1)
      dense_mention_mask[mention_slice] = 1
      dense_mention_ids[mention_slice] = np.random.randint(1, 1000)

    # This feature is only for `test_process_batchwise_mention_targets`
    dense_is_masked = np.random.randint(2, size=self.max_len)
    return {
        'text_ids': text_ids,
        'text_mask': text_mask,
        'dense_span_starts': dense_span_starts,
        'dense_span_ends': dense_span_ends,
        'mention_start_positions': mention_start_positions,
        'mention_end_positions': mention_end_positions,
        'dense_mention_ids': dense_mention_ids,
        'dense_mention_mask': dense_mention_mask,
        'dense_is_masked': dense_is_masked,
    }

  @parameterized.parameters((1, 1), (5, 5), (10, 1), (10, 2), (20, 3), (20, 5),
                            (20, 10), (100, 10), (100, 20), (100, 40))
  def test_get_dense_is_inside_for_dense_spans(self, seq_length, num_spans):
    sparse_start_positions = np.random.randint(seq_length, size=(num_spans,))
    sparse_end_positions = np.random.randint(seq_length, size=(num_spans,))
    sparse_end_positions = np.maximum(sparse_start_positions,
                                      sparse_end_positions)

    dense_start_positions = np.zeros(seq_length, dtype=np.int32)
    dense_end_positions = np.zeros(seq_length, dtype=np.int32)
    for i in range(num_spans):
      dense_start_positions[sparse_start_positions[i]] += 1
      dense_end_positions[sparse_end_positions[i]] += 1

    start_positions_tf = tf.compat.v1.placeholder_with_default(
        dense_start_positions, shape=[None])
    end_positions_tf = tf.compat.v1.placeholder_with_default(
        dense_end_positions, shape=[None])
    actual_tf = mention_preprocess_utils.get_dense_is_inside_for_dense_spans(
        start_positions_tf, end_positions_tf)
    actual_np = self.evaluate(actual_tf)
    for i in range(seq_length):
      is_inside_span = False
      for j in range(num_spans):
        if sparse_start_positions[j] <= i and i <= sparse_end_positions[j]:
          is_inside_span = True
          break
      self.assertEqual(actual_np[i], int(is_inside_span))

  @parameterized.parameters((1, 1), (5, 5), (10, 1), (10, 2), (20, 3), (20, 5),
                            (20, 10), (100, 10), (100, 20), (100, 40))
  def test_get_dense_is_inside_for_sparse_spans(self, seq_length, num_spans):
    sparse_start_positions = np.random.randint(seq_length, size=(num_spans,))
    sparse_end_positions = np.random.randint(seq_length, size=(num_spans,))
    sparse_end_positions = np.maximum(sparse_start_positions,
                                      sparse_end_positions)

    start_positions_tf = tf.compat.v1.placeholder_with_default(
        sparse_start_positions, shape=[None])
    end_positions_tf = tf.compat.v1.placeholder_with_default(
        sparse_end_positions, shape=[None])
    actual_tf = mention_preprocess_utils.get_dense_is_inside_for_sparse_spans(
        start_positions_tf, end_positions_tf, seq_length)
    actual_np = self.evaluate(actual_tf)
    for i in range(seq_length):
      is_inside_span = False
      for j in range(num_spans):
        if sparse_start_positions[j] <= i and i <= sparse_end_positions[j]:
          is_inside_span = True
          break
      self.assertEqual(actual_np[i], int(is_inside_span))

  def test_dynamic_padding_1d(self):

    def pad(values, length):
      tensor = tf.convert_to_tensor(values, dtype=tf.int32)
      return mention_preprocess_utils.dynamic_padding_1d(
          tensor, length, padding_token_id=0)

    self.assertAllEqual([1, 2], pad([1, 2, 3], 2))
    self.assertAllEqual([1, 2, 3], pad([1, 2, 3], 3))
    self.assertAllEqual([1, 2, 3, 0, 0], pad([1, 2, 3], 5))

  def test_mask_tokens_by_spans_uniform_distribution(self):
    n_repeats = 2000
    seq_length = 50
    mask_rate = 0.4
    max_mlm_targets = 10
    text_ids = np.arange(seq_length)
    start_positions = [0, 10, 20, 30, 40]
    end_positions = [0, 11, 22, 34, 49]

    n_times_position_was_masked = np.zeros(seq_length)

    text_ids_tf = tf.compat.v1.placeholder_with_default(text_ids, shape=[None])
    start_positions_tf = tf.compat.v1.placeholder_with_default(
        start_positions, shape=[None])
    end_positions_tf = tf.compat.v1.placeholder_with_default(
        end_positions, shape=[None])

    for _ in range(n_repeats):
      positions_to_mask_tf = mention_preprocess_utils.mask_tokens_by_spans(
          text_ids_tf, start_positions_tf, end_positions_tf, mask_rate,
          max_mlm_targets)
      positions_to_mask = self.evaluate(positions_to_mask_tf)
      n_times_position_was_masked[positions_to_mask] += 1

    n_times_position_was_masked /= n_repeats

    for i in range(seq_length):
      is_inside_span = False
      for j in range(len(start_positions)):
        if start_positions[j] <= i and i <= end_positions[j]:
          is_inside_span = True
          break
      if is_inside_span:
        self.assertNear(n_times_position_was_masked[i], mask_rate, 0.1)
      else:
        self.assertEqual(n_times_position_was_masked[i], 0)

  @parameterized.parameters(_get_test_mask_mentions_and_tokens_iterable())
  def test_mask_mentions_and_tokens(
      self,
      non_mention_mask_rate: float,
      mention_mask_rate: float,
      text_len: int,
      n_mentions: int,
  ):
    """Test masking for correctness and consistency."""

    input_features = self._gen_input(
        text_len=text_len,
        n_mentions=n_mentions,
    )

    masked_dict_tf = mention_preprocess_utils.mask_mentions_and_tokens_tf(
        text_ids=tf.convert_to_tensor(input_features['text_ids']),
        text_mask=tf.convert_to_tensor(input_features['text_mask']),
        dense_span_starts=tf.convert_to_tensor(
            input_features['dense_span_starts']),
        dense_span_ends=tf.convert_to_tensor(input_features['dense_span_ends']),
        non_mention_mask_rate=non_mention_mask_rate,
        mention_mask_rate=mention_mask_rate,
        max_mlm_targets=self.max_mlm_targets,
        mask_token_id=self.mask_token_id,
        vocab_size=self.vocab_size,
        random_replacement_prob=0,
        identity_replacement_prob=0,
    )
    masked_dict = self.evaluate(masked_dict_tf)

    # Check if arrays have correct length
    self.assertLen(masked_dict['masked_text_ids'], self.max_len)
    self.assertLen(masked_dict['mlm_target_positions'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_ids'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_weights'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_is_mention'], self.max_mlm_targets)

    # Check if number of masked tokens corresponds to nonzero weights
    self.assertEqual(
        (masked_dict['masked_text_ids'] == self.mask_token_id).sum(),
        int(masked_dict['mlm_target_weights'].sum()))

    # Check if target ids consistent with token ids at target positions
    self.assertTrue(
        np.all((masked_dict['mlm_target_ids'] == input_features['text_ids'][
            masked_dict['mlm_target_positions']]
               )[:int(masked_dict['mlm_target_weights'].sum())]))

  @parameterized.parameters(_get_test_mask_mentions_and_tokens_iterable())
  def test_mask_mentions_and_tokens_numpy(
      self,
      mask_rate: float,
      mention_mask_rate: float,
      text_len: int,
      n_mentions: int,
  ):
    """Test masking for correctness and consistency."""

    input_features = self._gen_input(
        text_len=text_len,
        n_mentions=n_mentions,
    )

    masked_dict = mention_preprocess_utils.mask_mentions_and_tokens(
        text_ids=input_features['text_ids'],
        text_mask=input_features['text_mask'],
        mention_start_positions=input_features['mention_start_positions'],
        mention_end_positions=input_features['mention_end_positions'],
        mask_rate=mask_rate,
        mention_mask_rate=mention_mask_rate,
        max_mlm_targets=self.max_mlm_targets,
        mask_token_id=self.mask_token_id,
    )

    # Check if arrays have correct length
    self.assertLen(masked_dict['masked_text_ids'], self.max_len)
    self.assertLen(masked_dict['mlm_target_positions'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_ids'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_weights'], self.max_mlm_targets)
    self.assertLen(masked_dict['mlm_target_is_mention'], self.max_mlm_targets)

    # Check if number of masked tokens corresponds to nonzero weights
    self.assertEqual(
        (masked_dict['masked_text_ids'] == self.mask_token_id).sum(),
        int(masked_dict['mlm_target_weights'].sum()))

    # Check if target ids consistent with token ids at target positions
    self.assertTrue(
        np.all((masked_dict['mlm_target_ids'] == input_features['text_ids'][
            masked_dict['mlm_target_positions']]
               )[:int(masked_dict['mlm_target_weights'].sum())]))

  @parameterized.parameters(
      ([1, 1, 0], [1, 0, 1], [0, 2, 0]),
      ([1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0
                                    ], [0, 2, 0, 5, 0, 0, 7, 0, 0]))
  def test_get_dense_span_ends_from_starts(self, dense_span_starts: List[int],
                                           dense_span_ends: List[int],
                                           expected: List[int]):
    dense_span_starts_tf = tf.compat.v1.placeholder_with_default(
        np.array(dense_span_starts).astype(np.int32), shape=[None])
    dense_span_ends_tf = tf.compat.v1.placeholder_with_default(
        np.array(dense_span_ends).astype(np.int32), shape=[None])
    actual_tf = mention_preprocess_utils.get_dense_span_ends_from_starts(
        dense_span_starts_tf, dense_span_ends_tf)
    actual_np = self.evaluate(actual_tf)
    expected_np = np.array(expected, dtype=np.int32)
    self.assertAllEqual(actual_np, expected_np)

  @parameterized.parameters(
      _get_test_process_batchwise_mention_targets_iterable())
  def test_process_batchwise_mention_targets(
      self,
      batch_size: int,
      text_len: int,
      n_mentions: int,
      max_mentions: int,
      max_mention_targets: int,
  ):
    """Test batchwise mention processing for correctness and consistency."""
    max_mention_targets = min(max_mention_targets, max_mentions)
    dense_span_starts = np.zeros((batch_size, self.max_len), dtype=np.int32)
    dense_span_ends = np.zeros((batch_size, self.max_len), dtype=np.int32)
    dense_mention_ids_list = []
    dense_mention_mask_list = []
    dense_is_masked_list = []

    for index in range(batch_size):
      input_features = self._gen_input(
          text_len=text_len,
          n_mentions=n_mentions,
      )
      dense_span_starts[index, input_features['mention_start_positions']] = 1
      dense_span_ends[index, input_features['mention_end_positions']] = 1
      dense_mention_ids_list.append(input_features['dense_mention_ids'])
      dense_mention_mask_list.append(input_features['dense_mention_mask'])
      dense_is_masked_list.append(input_features['dense_is_masked'])

    def placeholder(array: np.ndarray) -> tf.Tensor:
      return tf.compat.v1.placeholder_with_default(
          array.astype(np.int32), shape=[None])

    dense_mention_ids = np.stack(dense_mention_ids_list)
    dense_mention_mask = np.stack(dense_mention_mask_list)
    dense_is_masked = np.stack(dense_is_masked_list)

    actual_dict_tf = mention_preprocess_utils.process_batchwise_mention_targets(
        dense_span_starts=placeholder(dense_span_starts),
        dense_span_ends=placeholder(dense_span_ends),
        dense_mention_ids=placeholder(dense_mention_ids),
        dense_linked_mention_mask=placeholder(dense_mention_mask),
        dense_is_masked=placeholder(dense_is_masked),
        max_mentions=max_mentions,
        max_mention_targets=max_mention_targets,
    )

    actual_dict_np = self.evaluate(actual_dict_tf)

    # Check if all arrays are the right length
    self.assertLen(actual_dict_np['mention_batch_positions'], max_mentions)
    self.assertLen(actual_dict_np['mention_start_positions'], max_mentions)
    self.assertLen(actual_dict_np['mention_end_positions'], max_mentions)
    self.assertLen(actual_dict_np['mention_mask'], max_mentions)
    self.assertLen(actual_dict_np['mention_target_weights'],
                   max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_indices'],
                   max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_ids'], max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_is_masked'],
                   max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_batch_positions'],
                   max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_start_positions'],
                   max_mention_targets)
    self.assertLen(actual_dict_np['mention_target_end_positions'],
                   max_mention_targets)

    # Check every individual mention extracted
    num_sampled_mentions = 0
    for i in range(max_mentions):
      if actual_dict_np['mention_mask'][i] == 0:
        self.assertEqual(actual_dict_np['mention_batch_positions'][i], 0)
        self.assertEqual(actual_dict_np['mention_start_positions'][i], 0)
        self.assertEqual(actual_dict_np['mention_end_positions'][i], 0)
      else:
        num_sampled_mentions += 1
        batch_index = actual_dict_np['mention_batch_positions'][i]
        start_pos = actual_dict_np['mention_start_positions'][i]
        end_pos = actual_dict_np['mention_end_positions'][i]
        self.assertEqual(dense_span_starts[batch_index, start_pos], 1)
        self.assertEqual(dense_span_ends[batch_index, end_pos], 1)
        # Check start_pos and end_pos actually correspond to the same mention
        if start_pos < end_pos:
          self.assertEqual(
              dense_span_starts[batch_index, start_pos + 1:end_pos + 1].sum(),
              0)
          self.assertEqual(
              dense_span_ends[batch_index, start_pos + 1:end_pos].sum(), 0)

    total_mentions = dense_span_starts.sum()
    if max_mentions < total_mentions:
      self.assertEqual(num_sampled_mentions, max_mentions)
    else:
      self.assertEqual(num_sampled_mentions, total_mentions)

    # Check every individual linked mention extracted
    num_sampled_target_mentions = 0
    for j in range(max_mention_targets):
      if actual_dict_np['mention_target_weights'][j] == 0:
        self.assertEqual(actual_dict_np['mention_target_indices'][j], 0)
        self.assertEqual(actual_dict_np['mention_target_ids'][j], 0)
      else:
        num_sampled_target_mentions += 1
        i = actual_dict_np['mention_target_indices'][j]
        batch_index = actual_dict_np['mention_batch_positions'][i]
        start_pos = actual_dict_np['mention_start_positions'][i]
        end_pos = actual_dict_np['mention_end_positions'][i]
        self.assertEqual(dense_mention_mask[batch_index, start_pos], 1)
        self.assertEqual(dense_mention_ids[batch_index, start_pos],
                         actual_dict_np['mention_target_ids'][j])

    total_target_mentions = (dense_span_starts * dense_mention_mask).sum()
    if max_mention_targets < total_target_mentions:
      self.assertEqual(num_sampled_target_mentions, max_mention_targets)
    else:
      self.assertEqual(num_sampled_target_mentions, total_target_mentions)

    # Check mention masking computed correctly
    expected_value = dense_is_masked[
        actual_dict_np['mention_batch_positions'],
        actual_dict_np['mention_start_positions']][:max_mention_targets]
    self.assertTrue(
        np.all(actual_dict_np['mention_target_is_masked'] == expected_value))

    # Check mention_x_target_positions
    mention_target_indices = actual_dict_np['mention_target_indices']
    expected = actual_dict_np['mention_batch_positions'][mention_target_indices]
    self.assertTrue(
        np.all(actual_dict_np['mention_target_batch_positions'] == expected))
    expected = actual_dict_np['mention_start_positions'][mention_target_indices]
    self.assertTrue(
        np.all(actual_dict_np['mention_target_start_positions'] == expected))
    expected = actual_dict_np['mention_end_positions'][mention_target_indices]
    self.assertTrue(
        np.all(actual_dict_np['mention_target_end_positions'] == expected))

  def _gen_batch_for_test_add_entity_tokens(
      self, batch_size: int, min_length: int, max_length: int, vocab_size: int,
      max_mentions_per_sample: int, n_mentions: int, p_mention_to_pad: float,
      min_mlm_targets: int, max_mlm_targets: int,
      masked_token_id: int) -> Dict[str, np.ndarray]:
    """Generate batch with batchwise mentions to test `add_entity_tokens`."""

    features = {}

    # Generate text and mentions per sample
    features['text_ids'] = np.zeros((batch_size, max_length), dtype=np.int64)
    features['text_mask'] = np.zeros((batch_size, max_length), dtype=np.int64)
    features['mlm_target_positions'] = np.zeros((batch_size, max_mlm_targets),
                                                dtype=np.int64)
    features['mlm_target_weights'] = np.zeros((batch_size, max_mlm_targets),
                                              dtype=np.int64)
    all_mentions_in_batch = []

    for batch_index in range(batch_size):
      text_length = np.random.randint(low=min_length, high=max_length)
      text_ids = np.random.randint(
          low=1, high=vocab_size, size=text_length, dtype=np.int64)
      features['text_ids'][batch_index, :text_length] = text_ids
      features['text_mask'][batch_index, :text_length] = np.ones_like(text_ids)

      n_mlm_targets = np.random.randint(
          low=min_mlm_targets, high=max_mlm_targets)
      n_mlm_targets = min(n_mlm_targets, text_length)

      mlm_target_positions = np.random.choice(
          text_length, size=(n_mlm_targets), replace=False)
      features['mlm_target_positions'][
          batch_index, :n_mlm_targets] = mlm_target_positions
      features['mlm_target_weights'][batch_index, :n_mlm_targets] = 1

      features['text_ids'][batch_index, mlm_target_positions] = masked_token_id

      mention_positions = np.random.choice(
          text_length, size=(2 * max_mentions_per_sample), replace=False)
      mention_positions.sort()
      for index in range(max_mentions_per_sample):
        all_mentions_in_batch.append((batch_index, mention_positions[2 * index],
                                      mention_positions[2 * index + 1]))

    # Subsample at most `n_mentions` for the entire batch.
    np.random.shuffle(all_mentions_in_batch)

    def get_array_from_tuple(array_of_tuples, index):
      return np.array([x[index] for x in array_of_tuples], dtype=np.int64)

    all_mentions_in_batch = all_mentions_in_batch[:n_mentions]
    features['mention_batch_positions'] = get_array_from_tuple(
        all_mentions_in_batch, 0)
    self.assertLen(features['mention_batch_positions'], n_mentions)
    features['mention_start_positions'] = get_array_from_tuple(
        all_mentions_in_batch, 1)
    self.assertLen(features['mention_start_positions'], n_mentions)
    features['mention_end_positions'] = get_array_from_tuple(
        all_mentions_in_batch, 2)
    self.assertLen(features['mention_end_positions'], n_mentions)
    features['mention_mask'] = np.ones_like(features['mention_batch_positions'])

    # Mask fraction of `p_mention_to_pad` mentions.
    n_mentions_to_mask = int(p_mention_to_pad * n_mentions)
    if n_mentions_to_mask > 0:
      mentions_to_mask = np.random.choice(
          n_mentions, size=n_mentions_to_mask, replace=False)
      features['mention_mask'][mentions_to_mask] = 0
    self.assertEqual(features['mention_mask'].sum(),
                     n_mentions - n_mentions_to_mask)

    return features

  @parameterized.parameters(_get_test_add_entity_tokens())
  def test_add_entity_tokens(self, seed, batch_size, new_length,
                             p_mention_to_mask):
    np.random.seed(seed)
    min_length = 100
    max_length = 128
    vocab_size = 100
    max_mentions_per_sample = 32
    n_mentions = max_mentions_per_sample * batch_size
    p_mention_to_mask = 0
    entity_start_token_id = -1
    entity_end_token_id = -2
    masked_token_id = -3
    min_mlm_targets = 20
    max_mlm_targets = 30

    batch = self._gen_batch_for_test_add_entity_tokens(
        batch_size, min_length, max_length, vocab_size, max_mentions_per_sample,
        n_mentions, p_mention_to_mask, min_mlm_targets, max_mlm_targets,
        masked_token_id)

    def to_tensor(array):
      return tf.convert_to_tensor(array, dtype=tf.int64)

    actual_batch_tf = mention_preprocess_utils.add_entity_tokens(
        text_ids=to_tensor(batch['text_ids']),
        text_mask=to_tensor(batch['text_mask']),
        mention_mask=to_tensor(batch['mention_mask']),
        mention_batch_positions=to_tensor(batch['mention_batch_positions']),
        mention_start_positions=to_tensor(batch['mention_start_positions']),
        mention_end_positions=to_tensor(batch['mention_end_positions']),
        new_length=new_length,
        mlm_target_positions=to_tensor(batch['mlm_target_positions']),
        mlm_target_weights=to_tensor(batch['mlm_target_weights']),
        entity_start_token_id=entity_start_token_id,
        entity_end_token_id=entity_end_token_id,
    )
    actual_batch = self.evaluate(actual_batch_tf)

    expected_text_ids = np.zeros((batch_size, new_length), dtype=np.int64)
    expected_text_mask = np.zeros((batch_size, new_length), dtype=np.int64)
    expected_mlm_target_positions = np.zeros((batch_size, max_mlm_targets),
                                             dtype=np.int64)
    expected_mlm_target_weights = np.zeros((batch_size, max_mlm_targets),
                                           dtype=np.int64)

    mention_start_positions = batch['mention_start_positions'].copy()
    mention_end_positions = batch['mention_end_positions'].copy()
    mention_mask = batch['mention_mask'].copy()
    for batch_index in range(batch_size):
      # Subselect mentions from the current sample and insert entity tokens
      # into the corresponding sample.
      # pylint:disable=g-complex-comprehension
      current_indices = [
          (batch['mention_start_positions'][index], index)
          for index in range(n_mentions)
          if batch['mention_batch_positions'][index] == batch_index and
          batch['mention_mask'][index] == 1
      ]
      # Insert entity tokens for mentions starting from left to right.
      current_indices.sort()
      current_indices = [x[1] for x in current_indices]

      current_text_ids = batch['text_ids'][batch_index].tolist()
      current_text_mask = batch['text_mask'][batch_index].tolist()
      for index, m_index in enumerate(current_indices):
        if mention_end_positions[m_index] + 2 >= new_length:
          # Cannot insert entity tokens anymore
          for m_index_2 in current_indices[index:]:
            mention_end_positions[m_index_2] = 0
            mention_start_positions[m_index_2] = 0
            mention_mask[m_index_2] = 0
          break
        current_text_ids.insert(mention_start_positions[m_index],
                                entity_start_token_id)
        current_text_mask.insert(mention_start_positions[m_index], 1)
        mention_end_positions[m_index] += 2
        current_text_ids.insert(mention_end_positions[m_index],
                                entity_end_token_id)
        current_text_mask.insert(mention_end_positions[m_index], 1)

        for m_index_2 in current_indices[index + 1:]:
          mention_start_positions[m_index_2] += 2
          mention_end_positions[m_index_2] += 2

      current_text_ids = np.array(current_text_ids, dtype=np.int64)
      current_text_mask = np.array(current_text_mask, dtype=np.int64)

      # pylint:disable=g-complex-comprehension
      current_mlm_positions = [(batch['mlm_target_positions'][batch_index,
                                                              index], index)
                               for index in range(max_mlm_targets)
                               if batch['mlm_target_weights'][batch_index,
                                                              index] == 1]
      current_mlm_positions.sort()
      current_mlm_positions = [x[1] for x in current_mlm_positions]
      expected_mlm_target_positions[batch_index,
                                    current_mlm_positions] = np.nonzero(
                                        current_text_ids == masked_token_id)[0]
      expected_mlm_target_weights[batch_index, current_mlm_positions] = 1

      current_text_ids = current_text_ids[:new_length]
      expected_text_ids[batch_index, :len(current_text_ids)] = current_text_ids
      current_text_mask = current_text_mask[:new_length]
      expected_text_mask[
          batch_index, :len(current_text_mask)] = current_text_mask

    mlm_positions_within_new_length = expected_mlm_target_positions < new_length
    mlm_positions_within_new_length = mlm_positions_within_new_length.astype(
        np.int64)
    expected_mlm_target_weights *= mlm_positions_within_new_length
    expected_mlm_target_positions *= expected_mlm_target_weights

    self.assertArrayEqual(expected_text_ids, actual_batch['text_ids'])
    self.assertArrayEqual(expected_text_mask, actual_batch['text_mask'])

    self.assertArrayEqual(mention_start_positions,
                          actual_batch['mention_start_positions'])
    self.assertArrayEqual(mention_end_positions,
                          actual_batch['mention_end_positions'])
    self.assertArrayEqual(mention_mask, actual_batch['mention_mask'])

    self.assertArrayEqual(expected_mlm_target_positions,
                          actual_batch['mlm_target_positions'])
    self.assertArrayEqual(expected_mlm_target_weights,
                          actual_batch['mlm_target_weights'])


class HashTest(tf.test.TestCase, absltest.TestCase):
  max_int_value = 1000
  tensor_size = 1000
  batch_size = 1000

  def test_text_hash(self):
    hashes = set()
    for _ in range(self.batch_size):
      text = np.random.randint(self.max_int_value, size=self.tensor_size)
      hashes.add(mention_preprocess_utils.text_hash(text))

    duplicate_ratio = (self.batch_size - len(hashes)) / self.batch_size
    self.assertLessEqual(duplicate_ratio, 2 / self.batch_size)

  def test_text_hash_tf(self):
    hashes = set()
    for _ in range(self.batch_size):
      text = np.random.randint(self.max_int_value, size=self.tensor_size)
      text_tf = tf.convert_to_tensor(text)
      hash_tf = mention_preprocess_utils.text_hash_tf(text_tf, self.tensor_size)
      hash_np = self.evaluate(hash_tf)
      expected_hash_np = mention_preprocess_utils.text_hash(text)
      self.assertEqual(hash_np, expected_hash_np)
      hashes.add(hash_np)

    duplicate_ratio = (self.batch_size - len(hashes)) / self.batch_size
    self.assertLessEqual(duplicate_ratio, 2 / self.batch_size)

  def test_modified_cantor_pairing(self):
    a = np.random.randint(self.max_int_value, size=self.tensor_size)
    b = np.random.randint(self.max_int_value, size=self.tensor_size)

    hash_values = mention_preprocess_utils.modified_cantor_pairing(
        tf.constant(a), tf.constant(b))
    n_unique_hashes = len(np.unique(hash_values))
    duplicate_ratio = (self.tensor_size - n_unique_hashes) / self.tensor_size
    self.assertLessEqual(duplicate_ratio, 1 / 50)


if __name__ == '__main__':
  tf.test.main()
