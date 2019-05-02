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
import tempfile

from language.nql import dataset
from language.nql import nql
import numpy as np
import tensorflow as tf

NP_NONE = np.array([0., 0., 0., 0., 0.])
NP_A = np.array([1., 0., 0., 0., 0.])
NP_B = np.array([0., 1., 0., 0., 0.])
NP_C = np.array([0., 0., 1., 0., 0.])
NP_D = np.array([0., 0., 0., 1., 0.])
NP_UNK = np.array([0., 0., 0., 0., 1.])


class TestTFDataset(tf.test.TestCase):

  def setUp(self):
    super(TestTFDataset, self).setUp()
    self.clean_examples = ['a|A', 'b|B', 'c|C,D']
    self.noisy_examples = ['a|A', 'b|Beta', 'c|C,noise', 'd|D']
    self.empty_examples = ['a|A,', 'b|Beta,', 'c|']
    self.noisy_examples_good_count = 4
    self.empty_examples_good_count = 3
    self.context = nql.NeuralQueryContext()
    self.context.extend_type('uc_t', ['A', 'B', 'C', 'D'])
    self.context.freeze('uc_t')
    with tf.Session() as session:
      s_const = tf.constant('hello world', dtype=tf.string)
      s_eval = session.run(s_const)
      self.tf_string_type = type(s_eval)
      self.tf_string_type = bytes

  def as_list(self, dset):
    it = tf.data.make_one_shot_iterator(dset).get_next()
    buf = []
    with tf.Session() as session:
      session.run([
          tf.global_variables_initializer(),
          tf.local_variables_initializer(),
          tf.initializers.tables_initializer()
      ])
      try:
        while True:
          buf.append(session.run(it))
      except tf.errors.OutOfRangeError:
        pass
    return buf

  def test_size(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.clean_examples, [str, str],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    self.assertEqual(len(self.as_list(dset)), len(self.clean_examples))

  def test_repeat(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.clean_examples, [str, str],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    dset = dset.repeat(2)
    self.assertEqual(len(self.as_list(dset)), 2 * len(self.clean_examples))

  def test_lookup(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.clean_examples, [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), len(self.clean_examples))
    exp_values = {
        b'a': NP_A,
        b'b': NP_B,
        b'c': NP_C + NP_D,
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_normalize(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.clean_examples, [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=True)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), 3)
    exp_values = {
        b'a': NP_A,
        b'b': NP_B,
        b'c': 0.5 * (NP_C + NP_D),
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_normalize_empty(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.empty_examples, [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=True)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), 3)
    exp_values = {
        b'a': NP_A,
        b'b': NP_UNK,
        b'c': NP_NONE,
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_handle_unk_entity(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.clean_examples + ['d|UNK,D'], [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), len(self.clean_examples) + 1)
    exp_values = {
        b'a': NP_A,
        b'b': NP_B,
        b'c': NP_C + NP_D,
        b'd': NP_D + NP_UNK,
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_error_recovery(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.noisy_examples, [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), self.noisy_examples_good_count)
    exp_values = {
        b'a': NP_A,
        b'b': NP_UNK,
        b'c': NP_C + NP_UNK,
        b'd': NP_D,
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_empty_recovery(self):
    dset = dataset.tuple_dataset(
        self.context,
        self.empty_examples, [str, 'uc_t'],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    instances = self.as_list(dset)
    self.assertEqual(len(instances), self.empty_examples_good_count)
    exp_values = {
        b'a': NP_A,
        b'b': NP_UNK,
        b'c': NP_NONE,
    }
    for (s, a) in instances:
      self.assertEqual(type(s), self.tf_string_type)
      np.testing.assert_array_equal(a, exp_values[s])

  def test_idempotent_file_usage(self):
    data_tempfile = tempfile.mktemp('tuple_data')
    with open(data_tempfile, 'w') as fh:
      fh.write('\n'.join(self.clean_examples))
    dset = dataset.tuple_dataset(
        self.context,
        data_tempfile, [str, str],
        field_separator='|',
        entity_separator=',',
        normalize_outputs=False)
    dset = dset.repeat(2)
    self.assertEqual(len(self.as_list(dset)), 2 * len(self.clean_examples))


class TestKhotOverFrozenWithNone(tf.test.TestCase):

  def setUp(self):
    self.context = nql.NeuralQueryContext()
    self.context.declare_entity_type(
        'uc_t', fixed_vocab=['A', 'B', 'C', 'D'], unknown_marker=None)

  # For fixed-vocab types having no unknown_marker, unknown values are ignored.
  def testSafe(self):
    khot = dataset.k_hot_array_from_string_list(self.context, 'uc_t',
                                                ['A', 'B', 'C', 'D', 'Z'])
    np.testing.assert_array_equal([1., 1., 1., 1.], khot)


if __name__ == '__main__':
  tf.test.main()
