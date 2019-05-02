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
"""Tests for nql."""

import pprint
import tempfile

from language.nql import nql
from language.nql import nql_test_lib
import mock
import numpy as np
import tensorflow as tf

cell = nql_test_lib.cell


class TestDeclaredTypes(tf.test.TestCase):

  def setUp(self):
    super(TestDeclaredTypes, self).setUp()
    self.context = nql.NeuralQueryContext()

  def test_fixed_vocab(self):
    self.context.declare_entity_type('vowel_t', fixed_vocab='aeiou')
    self.context.extend_type('vowel_t', ['y', 'oo'])
    instances = self.get_instances('vowel_t')
    for v in 'aeiou':
      self.assertIn(v, instances)
    # +1 for UNK
    self.assertEqual(len(instances), len('aeiou') + 1)

  def test_fixed_vocab_size(self):
    self.context.declare_entity_type('id_t', fixed_vocab_size=10)
    self.context.extend_type('id_t', '0123456')
    instances = self.get_instances('id_t')
    for v in '0123456':
      self.assertIn(v, instances)
    self.assertEqual(len(instances), 10)
    self.assertEqual(self.context.get_max_id('id_t'), 10)
    # now reset and reload
    self.context.clear_entity_type_vocabulary('id_t')
    self.context.extend_type('id_t', 'abcd')
    instances = self.get_instances('id_t')
    for v in 'abcd':
      self.assertIn(v, instances)
    for v in '0123456':
      self.assertNotIn(v, instances)
    self.assertEqual(len(instances), 10)
    self.assertEqual(self.context.get_max_id('id_t'), 10)

  def get_instances(self, type_name):
    result = []
    for i in range(self.context.get_max_id(type_name)):
      result.append(self.context.get_entity_name(i, type_name))
    return result


class TestLoad(tf.test.TestCase):

  def setUp(self):
    super(TestLoad, self).setUp()
    self.context = nql.NeuralQueryContext()
    self.context.declare_relation('foo', 'foo_d', 'foo_r')
    self.context.declare_relation('bat', 'bat_d', 'bat_r')

  def _tabify(self, lines):
    return ['\t'.join(parts) + '\n' for parts in lines]

  def test_schema(self):
    self.assertTrue(self.context.is_relation('foo'))
    self.assertFalse(self.context.is_relation('bar'))
    self.assertEqual(self.context.get_domain('foo'), 'foo_d')
    self.assertEqual(self.context.get_range('foo'), 'foo_r')
    self.assertEqual(self.context.get_domain('bat'), 'bat_d')

  def test_load_schema(self):
    filename = tempfile.mktemp()
    with open(filename, 'w') as fh:
      fh.write('mentions_entity\tquestion_t\tentity_t\n'
               'has_feature\tquestion_t\tword_t\n'
               'answer\tquestion_t\tentity_t\n')
    self.context.load_schema(filename)
    self.assertEqual(self.context.get_domain('answer'), 'question_t')
    self.assertEqual(self.context.get_range('answer'), 'entity_t')

  def test_load_kg(self):
    kg_lines = [['foo', 'a', 'x', '1.0'], ['foo', 'a', 'y'],
                ['foo', 'b', 'z', '2.0'], ['bat', 'dracula', 'true'],
                ['bat', 'louisville slugger', 'true', '0.1']]
    self.context.load_kg(lines=self._tabify(kg_lines))
    self.assertEqual(
        list(self.context.query_kg('foo', 'a')), [('x', 1.0), ('y', 1.0)])
    self.assertEqual(
        list(self.context.query_kg('foo', 'z', as_object=True)), [('b', 2.0)])
    tmp = sorted(list(self.context.query_kg('bat', 'true', as_object=True)))
    self.assertEqual(tmp[0][0], 'dracula')
    self.assertEqual(tmp[1][0], 'louisville slugger')
    # small confidences maybe slightly changed in in float conversion?
    self.assertAlmostEqual(tmp[1][1], 0.1, delta=0.00001)
    self.assertEqual((2, 3), self.context.get_shape('foo'))
    self.assertEqual((2, 1), self.context.get_shape('bat'))
    self.assertEqual(self.context.get_unk_id('foo_d'), None)
    self.assertEqual(self.context.get_unk_id('foo_r'), None)
    self.assertEqual(self.context.get_unk_id('bat_d'), None)
    self.assertEqual(self.context.get_unk_id('bat_r'), None)

  def test_load_kg_and_freeze(self):
    kg_lines = [['foo', 'a', 'x', '1.0'], ['foo', 'a', 'y'],
                ['foo', 'b', 'z', '2.0'], ['bat', 'dracula', 'true'],
                ['bat', 'louisville slugger', 'true', '0.1']]
    self.context.load_kg(lines=self._tabify(kg_lines), freeze=True)
    self.assertEqual((3, 4), self.context.get_shape('foo'))
    self.assertEqual((3, 2), self.context.get_shape('bat'))
    self.assertEqual(self.context.get_unk_id('foo_d'), 2)
    self.assertEqual(self.context.get_unk_id('foo_r'), 3)
    self.assertEqual(self.context.get_unk_id('bat_d'), 2)
    self.assertEqual(self.context.get_unk_id('bat_r'), 1)

  def test_load_kg_from_files(self):
    kg_lines1 = [['foo', 'a', 'x', '1.0'], ['foo', 'a', 'y'],
                 ['foo', 'b', 'z', '2.0']]
    kg_lines2 = [['bat', 'dracula', 'true'],
                 ['bat', 'louisville slugger', 'true', '0.1']]
    filename1 = tempfile.mktemp()
    filename2 = tempfile.mktemp()
    with open(filename1, 'w') as fh:
      for line in self._tabify(kg_lines1):
        fh.write(line)
    with open(filename2, 'w') as fh:
      for line in self._tabify(kg_lines2):
        fh.write(line)
    self.context.load_kg(files=[open(filename1, 'r'), open(filename2, 'r')])
    self.assertEqual(
        list(self.context.query_kg('foo', 'a')), [('x', 1.0), ('y', 1.0)])
    self.assertEqual(
        list(self.context.query_kg('foo', 'z', as_object=True)), [('b', 2.0)])
    tmp = sorted(list(self.context.query_kg('bat', 'true', as_object=True)))
    self.assertEqual(tmp[0][0], 'dracula')
    self.assertEqual(tmp[1][0], 'louisville slugger')
    # small confidences maybe slightly changed in in float conversion?
    self.assertAlmostEqual(tmp[1][1], 0.1, delta=0.00001)

  def test_load_kg_and_ignore_undef_rels(self):
    kg_lines = [['foo', 'a', 'x', '1.0'], ['undef_rel', 'a', 'y'],
                ['foo', 'b', 'z', '2.0']]
    self.context.load_kg(lines=self._tabify(kg_lines), ignore_undef_rels=True)

  def test_load_kg_and_crash_undef_rels(self):
    kg_lines = [['foo', 'a', 'x', '1.0'], ['undef_rel', 'a', 'y'],
                ['foo', 'b', 'z', '2.0']]
    with self.assertRaises(nql.RelationNameError):
      self.context.load_kg(
          lines=self._tabify(kg_lines), ignore_undef_rels=False)

  def test_extend_type(self):
    self.assertFalse(self.context.is_type('vowel_t'))
    self.context.extend_type('vowel_t', 'aeiou')
    self.assertTrue(self.context.is_type('vowel_t'))
    self.assertEqual(self.context.get_max_id('vowel_t'), 5)
    for i, v in enumerate('aeiou'):
      self.assertEqual(self.context.get_id(v, 'vowel_t'), i)


class TestOnGrid(tf.test.TestCase):

  def setUp(self):
    super(TestOnGrid, self).setUp()
    self.context = nql_test_lib.make_grid()
    self.session = tf.Session()

  def tearDown(self):
    super(TestOnGrid, self).tearDown()
    tf.reset_default_graph()

  def test_problematic_relation_name(self):
    with mock.patch.object(tf.logging, 'warn') as mocked_log:
      self.context.declare_relation('follow', 'place_t', 'place_t')
      self.context.declare_relation('tf', 'place_t', 'place_t')
      mocked_log.assert_called()

  def test_grid_np(self):
    # check the numpy computations
    x_vec = self.context.one_hot_numpy_array(cell(2, 2), 'place_t')
    for (d, (i, j)) in zip('nsew', [(1, 2), (3, 2), (2, 3), (2, 1)]):
      mat = self.context._np_initval[d]
      y_vec = mat.dot(x_vec.transpose()).transpose()
      self.assertEqual(self._vec2cell(y_vec), cell(i, j))

  def test_grid_tf(self):
    # check the tensorflow computations
    x = self.context.one(cell(2, 2), 'place_t')
    for (d, (i, j)) in zip('nsew', [(1, 2), (3, 2), (2, 3), (2, 1)]):
      mat = self.context.get_tf_tensor(d)
      y_vec = tf.transpose(
          a=tf.sparse.sparse_dense_matmul(mat, tf.transpose(a=x.tf))).eval(
              session=self.session)
      self.assertEqual(self._vec2cell(y_vec), cell(i, j))

  def test_grid_context(self):
    # check the context computations
    x = self.context.one(cell(2, 2), 'place_t')
    for (d, (i, j)) in zip('nsew', [(1, 2), (3, 2), (2, 3), (2, 1)]):
      y = x.follow(d)
      y_vec = self.session.run(y.tf)
      self.assertEqual(self._vec2cell(y_vec), cell(i, j))

  def test_or(self):
    x = self.context.one(cell(2, 2), 'place_t')

    def near(x):
      return x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')

    y_vec = self.session.run(near(x).tf)
    self._check_2_2_neighbors(y_vec)

  def _check_2_2_neighbors(self, y_vec):
    entity_ids = np.nonzero(y_vec)[1].tolist()
    entity_names = set(
        [self.context.get_entity_name(i, 'place_t') for i in entity_ids])
    for i, j in [(1, 2), (3, 2), (2, 3), (2, 1)]:
      self.assertIn(cell(i, j), entity_names)
    self.assertEqual(len(entity_names), 4)

  def test_eval(self):
    x = self.context.one(cell(2, 2), 'place_t')

    def near(x):
      return x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')

    y_vec = near(x).eval(session=self.session, as_dicts=False)
    with self.session.as_default():
      y_vec = near(x).eval(as_dicts=False)
    self._check_2_2_neighbors(y_vec)
    # test the dictionary bit
    y_dict = near(x).eval(self.session)
    for i, j in [(1, 2), (3, 2), (2, 3), (2, 1)]:
      self.assertIn(cell(i, j), y_dict)
    self.assertEqual(len(y_dict), 4)

    x.eval(self.session)
    near(x).eval(self.session)

    w = self.context.placeholder('w', 'place_t')
    near_w = (w.n() | w.s())
    two_hop_w = (near_w.e() | near_w.w())
    val = x.eval(self.session, as_dicts=False)
    near_w.eval(self.session, feed_dict={w.name: val})
    two_hop_w.eval(self.session, feed_dict={w.name: val})

    # make sure weights work

    def lr_near(x):
      return near(x).weighted_by('distance_to', 'ul')

    y_dict = lr_near(x).eval(session=self.session)
    self._check_lr_near_2_2(y_dict)

  def test_weighted_by(self):
    x = self.context.one(cell(2, 2), 'place_t')
    near_x = x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')
    lr_near_x = near_x.weighted_by('distance_to', 'ul')
    y_vec = self.session.run(lr_near_x.tf)
    y_dict = self._as_dict(y_vec, 'place_t')
    self._check_lr_near_2_2(y_dict)

  def test_provenance(self):
    x = self.context.one(cell(2, 2), 'place_t')
    near_x = x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')
    ul_near_x = near_x.weighted_by('distance_to', 'ul')
    pprinted = pprint.pformat(ul_near_x.provenance.pprintable())
    lines = pprinted.split('\n')
    self.assertEqual(len(lines), 25)
    self.assertEqual(
        lines[0], "{'inner': {'inner': {'inner': {'inner': {'args': ('n', 1),")
    self.assertEqual(lines[-1], "           'operation': 'follow'}}")

  def test_tf_utilities(self):
    # make a minibatch of 2 copies of cell_2_2, to
    # test broadcasting
    row = self.context.one_hot_numpy_array(cell(2, 2), 'place_t')
    x = self.context.constant(np.vstack([row, row]), 'place_t')

    def near(x):
      return x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')

    def lr_near(x):
      return near(x).weighted_by('distance_to', 'ul')

    # softmax should split most of the weight between 2, 3 and 3,2

    sm = nql.nonneg_softmax(lr_near(x).tf)
    sm_list = self.context.as_dicts(self.session.run(sm), 'place_t')
    for i in range(len(sm_list)):
      self.assertAllInRange(sm_list[i][cell(2, 3)], 0.4, 0.5)
      self.assertAllInRange(sm_list[i][cell(3, 2)], 0.4, 0.5)
    # construct a target
    target_nq = (self.context.one(cell(2, 3), 'place_t')
                 | self.context.one(cell(3, 2), 'place_t')) * 0.5
    low_loss = nql.nonneg_crossentropy(sm, target_nq.tf)
    offtarget_nq = (self.context.one(cell(2, 1), 'place_t')
                    | self.context.one(cell(2, 2), 'place_t')) * 0.5
    high_loss = nql.nonneg_crossentropy(sm, offtarget_nq.tf)
    lo = self.session.run(low_loss)
    hi = self.session.run(high_loss)
    self.assertAllInRange(lo, 0.5, 2.0)
    self.assertAllInRange(hi, 5.0, 15.0)

  def _check_lr_near_2_2(self, y_dict):
    self.assertAlmostEqual(y_dict['cell_1_2'], 3.0, delta=0.0001)
    self.assertAlmostEqual(y_dict['cell_2_1'], 3.0, delta=0.0001)
    self.assertAlmostEqual(y_dict['cell_3_2'], 5.0, delta=0.0001)
    self.assertAlmostEqual(y_dict['cell_2_3'], 5.0, delta=0.0001)
    self.assertEqual(len(y_dict), 4)

  def test_grid_multihop(self):
    x = self.context.one(cell(2, 2), 'place_t')
    for (d1, d2) in zip('nsew', 'snwe'):
      y = x.follow(d1).follow(d2)
      y_vec = self.session.run(y.tf)
      self.assertEqual(self._vec2cell(y_vec), cell(2, 2))
    for d in 'nsew':
      y = x.follow(d).follow(d, -1)
      y_vec = self.session.run(y.tf)
      self.assertEqual(self._vec2cell(y_vec), cell(2, 2))
    y = x.follow('n').follow('e').follow('s').follow('w')
    y_vec = self.session.run(y.tf)
    self.assertEqual(self._vec2cell(y_vec), cell(2, 2))

  def test_grid_reflection(self):
    x = self.context.one(cell(2, 2), 'place_t')
    y = x.n().s()
    y_vec = self.session.run(y.tf)
    self.assertEqual(self._vec2cell(y_vec), cell(2, 2))
    y = x.e().w()
    y_vec = self.session.run(y.tf)
    self.assertEqual(self._vec2cell(y_vec), cell(2, 2))
    y = x.n().n(-1)
    y_vec = self.session.run(y.tf)
    self.assertEqual(self._vec2cell(y_vec), cell(2, 2))

  def test_factory(self):

    class MyExpression(nql.NeuralQueryExpression):

      def ne(self):
        return self.n().e()

    old_class = self.context.expression_factory_class
    self.context.expression_factory_class = MyExpression
    x = self.context.one(cell(2, 2), 'place_t')
    y1 = x.n().e().eval(session=self.session)
    y2 = x.ne().eval(session=self.session)
    self.context.expression_factory_class = old_class
    self.assertEqual(y1, y2)

  def test_grid_colors(self):
    black_color = self.context.one('black', 'color_t')
    black_cells = black_color.follow('color', -1)
    all_cells = self.context.all('place_t')
    num_cells = tf.reduce_sum(input_tensor=all_cells.tf)
    num_black_cells = tf.reduce_sum(input_tensor=black_cells.tf)
    self.assertEqual(self.session.run(num_cells), 16.0)
    self.assertEqual(self.session.run(num_black_cells), 8.0)

  def test_grid_conditionals(self):
    go_north = lambda x: x.follow('n')
    go_east = lambda x: x.follow('e')

    def conditional_move1(c):
      return go_north(c).weighted_by_sum(c.filtered_by(
          'color', 'black')) + go_east(c).weighted_by_sum(
              c.filtered_by('color', 'white'))

    def conditional_move2(c):
      return go_north(c).if_any(c.filtered_by(
          'color', 'black')) + go_east(c).if_any(
              c.filtered_by('color', 'white'))

    x1 = self.context.one(cell(2, 2), 'place_t')
    x2 = self.context.one(cell(3, 2), 'place_t')
    y1_vec = self.session.run(conditional_move1(x1).tf)
    y2_vec = self.session.run(conditional_move1(x2).tf)
    d1 = self._as_dict(y1_vec, 'place_t')
    d2 = self._as_dict(y2_vec, 'place_t')

    def check_dicts(d1, d2):
      self.assertEqual(len(d1), 1)
      self.assertEqual(len(d2), 1)
      self.assertIn(cell(1, 2), d1)
      self.assertIn(cell(3, 3), d2)

    check_dicts(d1, d2)
    # make sure the syntactic variant works
    z1_vec = self.session.run(conditional_move2(x1).tf)
    z2_vec = self.session.run(conditional_move2(x2).tf)
    e1 = self._as_dict(z1_vec, 'place_t')
    e2 = self._as_dict(z2_vec, 'place_t')
    check_dicts(e1, e2)

  def test_grid_recursion(self):

    def at_most_k_cells_north(k, c):
      if k <= 0:
        return c
      else:
        return c + at_most_k_cells_north(k - 1, c.follow('n'))

    x = self.context.one(cell(3, 2), 'place_t')
    for k in range(3):
      d = at_most_k_cells_north(k, x).eval(session=self.session, as_dicts=True)
      self.assertIn(cell(3 - k, 2), d)
      self.assertEqual(len(d), k + 1)

  def test_one_hot_numpy_with_none(self):
    self.context.declare_entity_type(
        'fixed_none_t', fixed_vocab='aeiou', unknown_marker=None)
    with self.assertRaises(nql.EntityNameError):
      self.context.one_hot_numpy_array('Z', 'fixed_none_t')

  # a few helper functions

  def _vec2cell(self, onehot_vec):
    # map a one-hot vector to a cell name
    entity_id = np.nonzero(onehot_vec)[1][0]
    return self.context.get_entity_name(entity_id, 'place_t')

  def _as_dict(self, vec, type_name):
    result = {}
    entity_ids = np.nonzero(vec)[1].tolist()
    for i in entity_ids:
      result[self.context.get_entity_name(i, type_name)] = vec[0, i]
    return result


class TestOnDenseGrid(TestOnGrid):

  def setUp(self):
    super(TestOnDenseGrid, self).setUp()
    sparse_grid_context = nql_test_lib.make_grid()
    context = nql.NeuralQueryContext()
    # copy the grid but densify some of it
    context.declare_relation('n', 'place_t', 'place_t')
    context.declare_relation('s', 'place_t', 'place_t')
    context.declare_relation('e', 'place_t', 'place_t')
    context.declare_relation('w', 'place_t', 'place_t')
    context.declare_relation('color', 'place_t', 'color_t', dense=True)
    context.declare_relation('distance_to', 'place_t', 'corner_t')
    # copy the type definitions
    for type_name in sparse_grid_context.get_type_names():
      entity_list = [
          sparse_grid_context.get_entity_name(i, type_name)
          for i in range(sparse_grid_context.get_max_id(type_name))
      ]
      context.extend_type(type_name, entity_list)
    # copy the data over
    for r in sparse_grid_context.get_relation_names():
      m = sparse_grid_context.get_initial_value(r)
      if context.is_dense(r):
        context.set_initial_value(r, m.todense())
      else:
        context.set_initial_value(r, m)
    self.context = context
    self.session = tf.Session()


class TestTrainable(TestOnGrid):

  def setUp(self):
    super(TestTrainable, self).setUp()
    self.context.declare_relation(
        'trained_distance_to', 'place_t', 'corner_t', trainable=True)
    self.context._np_initval['trained_distance_to'] = self.context._np_initval[
        'distance_to']

  def test_trainable_weighted_by(self):
    x = self.context.one(cell(2, 2), 'place_t')
    near_x = x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')
    lr_near_x = near_x.weighted_by('trained_distance_to', 'ul')
    # need to initialize if there are any trainable relations
    self.session.run(self.context.get_initializers())
    y_vec = self.session.run(lr_near_x.tf)
    y_dict = self._as_dict(y_vec, 'place_t')
    self._check_lr_near_2_2(y_dict)

    # Must be run on an operation after training.
    filename = tempfile.mktemp()
    iv = self.context.get_initial_value('trained_distance_to')
    self.context.serialize_trained(filename, self.session)
    # Clear out the initial values.
    self.context.set_initial_value('trained_distance_to',
                                   123. + np.zeros(iv.shape))

    # Confirm we have done that.
    iv0 = self.context.get_initial_value('trained_distance_to')
    self.assertFalse(np.any(iv.todense().ravel() == iv0.todense().ravel()))

    # Check the deserialized values as ewqual
    self.context.deserialize_trained(filename)
    io_iv = self.context.get_initial_value('trained_distance_to')
    np.testing.assert_array_equal(iv.todense().ravel(), io_iv.todense().ravel())

  def test_gradients(self):
    x = self.context.one(cell(2, 2), 'place_t')
    near_x = x.follow('n') + x.follow('s') + x.follow('e') + x.follow('w')
    lr_near_x = near_x.weighted_by('trained_distance_to', 'ul')
    expected_y = self.context.one(cell(1, 2), 'place_t') * 3 + self.context.one(
        cell(2, 1), 'place_t') * 3 + self.context.one(cell(
            3, 2), 'place_t') * 5 + self.context.one(cell(2, 3), 'place_t') * 5
    almost_y = self.context.one(cell(1, 2), 'place_t') * 2 + self.context.one(
        cell(2, 1), 'place_t') * 3 + self.context.one(cell(
            3, 2), 'place_t') * 4 + self.context.one(cell(2, 3), 'place_t') * 5
    # need to initialize if there are any trainable relations
    self.session.run(self.context.get_initializers())
    # compute some gradients
    loss_1 = tf.reduce_sum(
        input_tensor=tf.multiply(lr_near_x.tf - expected_y.tf, lr_near_x.tf -
                                 expected_y.tf))
    loss_2 = tf.reduce_sum(
        input_tensor=tf.multiply(lr_near_x.tf - almost_y.tf, lr_near_x.tf -
                                 almost_y.tf))
    self.assertEqual(loss_1.eval(session=self.session), 0.0)
    self.assertEqual(loss_2.eval(session=self.session), 2.0)
    grad_1 = tf.gradients(
        ys=loss_1,
        xs=self.context.get_underlying_parameter('trained_distance_to'))
    grad_2 = tf.gradients(
        ys=loss_2,
        xs=self.context.get_underlying_parameter('trained_distance_to'))
    self.assertEqual(len(grad_1), 1)
    self.assertEqual(len(grad_2), 1)
    sum_grad_1 = tf.reduce_sum(input_tensor=grad_1[0])
    sum_grad_2 = tf.reduce_sum(input_tensor=grad_2[0])
    self.assertEqual(sum_grad_1.eval(session=self.session), 0.0)
    self.assertEqual(sum_grad_2.eval(session=self.session), 4.0)


class TestFollowGroup(TestOnGrid):

  def setUp(self):
    super(TestFollowGroup, self).setUp()
    self.context.construct_relation_group('dir_g', 'place_t', 'place_t')
    self.context.construct_relation_group('vdir_g', 'place_t', 'place_t',
                                          ['n', 's'])

  def test_declare_group(self):
    self.assertEqual(len(self.context._group['dir_g'].members), 4)
    for d in 'nsew':
      self.assertIn(d, self.context._group['dir_g'].members)
    # Implicitly computed group will force a specific ordering.
    for t, i in {'e': 0, 'n': 1, 's': 2, 'w': 3}.items():
      self.assertEqual(i, self.context.get_id(t, 'dir_g'))
    g = self.context._group['dir_g']
    for rel in [g.subject_rel, g.object_rel]:
      self.assertEqual(self.context.get_domain(rel), g.triple_type)
      self.assertEqual(self.context.get_range(rel), 'place_t')
    self.assertEqual(self.context.get_domain(g.relation_rel), g.triple_type)
    self.assertEqual(self.context.get_range(g.relation_rel), g.name)

    self.assertEqual(len(self.context._group['vdir_g'].members), 2)
    for d in 'ns':
      self.assertIn(d, self.context._group['vdir_g'].members)

  def test_group_rels(self):
    # going north
    x = self.context.one(cell(2, 2), 'place_t')
    rel_n = self.context.one('n', 'dir_g')
    y1 = x.n().eval(self.session)

    def follow_thru(x, r):
      return (x.dir_g_subj(-1).dir_g_weight() * r.dir_g_rel(-1)).dir_g_obj()

    y2 = follow_thru(x, rel_n).eval(self.session)
    self.assertEqual(y1, y2)
    rel_all = self.context.all('dir_g')
    z1 = (x.n() | x.s() | x.e() | x.w()).eval(self.session)
    z2 = follow_thru(x, rel_all).eval(self.session)
    self.assertEqual(z1, z2)

    def follow_thru_triple(group_name, x, r):
      g = self.context._group[group_name]
      return (x.follow(g.subject_rel, -1).follow(g.weight_rel) * r.follow(
          g.relation_rel, -1)).follow(g.object_rel)

    y3 = follow_thru_triple('dir_g', x, rel_n).eval(self.session)
    self.assertEqual(y1, y3)
    z3 = follow_thru_triple('dir_g', x, rel_all).eval(self.session)
    self.assertEqual(z1, z3)
    w1 = (x.n() + x.s()).eval(self.session)
    vrel_all = self.context.all('vdir_g')
    w2 = follow_thru_triple('vdir_g', x, vrel_all).eval(self.session)
    self.assertEqual(w1, w2)

    y4 = x.follow(rel_n).eval(self.session)
    self.assertEqual(y1, y4)
    z4 = x.follow(rel_all).eval(self.session)
    self.assertEqual(z1, z4)

    v1 = x.n(-1).eval(self.session)
    v2 = x.follow(rel_n, -1).eval(self.session)
    self.assertEqual(v1, v2)


class TestMinibatchOnGrid(TestOnGrid):

  def test_grid_tf(self):
    x_np = self.minibatch_np([cell(2, 2), cell(3, 3)])
    x = self.context.as_nql(x_np, 'place_t')
    for (d, ij_list) in zip(
        'nsew', [[(1, 2), (2, 3)], [(3, 2)], [(2, 3)], [(2, 1), (3, 2)]]):
      mat = self.context.get_tf_tensor(d)
      y_mat = tf.transpose(
          a=tf.sparse.sparse_dense_matmul(mat, tf.transpose(a=x.tf))).eval(
              session=self.session)
      y_dict = self.context.as_dicts(y_mat, 'place_t')
      for k, (i, j) in enumerate(ij_list):
        self.assertIn(cell(i, j), y_dict[k])

  def test_context_eval_asdict(self):
    x = self.minibatch([cell(2, 2), cell(3, 3)], 'place_t')
    y = x.follow('n')
    y_list = y.eval(session=self.session, as_dicts=True)
    self.assertIsInstance(y_list, list)
    self.assertEqual(len(y_list), 2)
    self.confirm_dict(y_list[0], {cell(1, 2): 1.0})
    self.confirm_dict(y_list[1], {cell(2, 3): 1.0})

  # helpers

  def minibatch_np(self, cell_names):
    rows = [self.context.one_hot_numpy_array(c, 'place_t') for c in cell_names]
    result = np.vstack(rows)
    return result

  def minibatch(self, cell_names, typename):
    return self.context.as_nql(self.minibatch_np(cell_names), typename)

  def confirm_dict(self, actual, expected):
    for key in expected:
      self.assertAlmostEqual(actual[key], expected[key], delta=0.00001)
      self.assertEqual(len(actual), len(expected))

  # tests

  def test_weighted_by(self):
    x = self.minibatch([cell(2, 2), cell(3, 3)], 'place_t')
    y = (x.n() + x.s() + x.e() + x.w()).weighted_by('distance_to', 'ul')
    y_dict = y.eval(session=self.session)
    self.confirm_dict(y_dict[0], {
        cell(1, 2): 3,
        cell(2, 1): 3,
        cell(3, 2): 5,
        cell(2, 3): 5
    })
    self.confirm_dict(y_dict[1], {cell(3, 2): 5, cell(2, 3): 5})
    # test the as_top option
    y_top = y.eval(session=self.session, as_top=2)
    self.assertEqual(len(y_top), 2)
    for i in range(2):
      self.assertEqual(len(y_top[i]), 2)
      for cell_name, score in y_top[i]:
        self.assertEqual(score, 5.0)
        self.assertTrue(cell_name == cell(3, 2) or cell_name == cell(2, 3))

  def test_tf_op(self):
    x = self.minibatch([cell(2, 2), cell(3, 3)], 'place_t')
    z = (x.n() + x.s() + x.e() + x.w()).weighted_by('distance_to', 'ul')
    y = z.tf_op(lambda t: tf.clip_by_value(t, 0.0, 1.0))
    y_dict = y.eval(session=self.session)
    self.confirm_dict(y_dict[0], {
        cell(1, 2): 1,
        cell(2, 1): 1,
        cell(3, 2): 1,
        cell(2, 3): 1
    })
    self.confirm_dict(y_dict[1], {cell(3, 2): 1, cell(2, 3): 1})

  def test_multihop(self):
    x = self.minibatch([cell(2, 2), cell(3, 3)], 'place_t')
    y = x.n().w()
    y_dict = y.eval(session=self.session)
    self.confirm_dict(y_dict[0], {cell(1, 1): 1.0})
    self.confirm_dict(y_dict[1], {cell(2, 2): 1.0})
    z = y.n().w()
    z_dict = z.eval(session=self.session)
    self.confirm_dict(z_dict[0], {cell(0, 0): 1.0})
    self.confirm_dict(z_dict[1], {cell(1, 1): 1.0})
    w = z.n().w()
    w_dict = w.eval(session=self.session)
    self.confirm_dict(w_dict[0], {})
    self.confirm_dict(w_dict[1], {cell(0, 0): 1.0})

  def test_grid_conditionals(self):
    go_north = lambda x: x.follow('n')
    go_east = lambda x: x.follow('e')

    def conditional_move(c):
      return go_north(c).weighted_by_sum(c.filtered_by(
          'color', 'black')) + go_east(c).weighted_by_sum(
              c.filtered_by('color', 'white'))

    x = self.minibatch([cell(2, 2), cell(3, 2)], 'place_t')
    y = conditional_move(x)
    y_dict = y.eval(session=self.session)
    self.confirm_dict(y_dict[0], {cell(1, 2): 1.0})
    self.confirm_dict(y_dict[1], {cell(3, 3): 1.0})

  def test_and_or_jump_notation(self):
    x = self.context.one(cell(2, 2), 'place_t')

    def neighbors1(c):
      return c.n() + c.s() + c.e() + c.w()

    def neighbors2(c):
      return c.n() | c.s() | c.e() | c.w()

    d1 = neighbors1(x).eval(session=self.session)
    self.confirm_dict(d1, {
        cell(1, 2): 1.0,
        cell(3, 2): 1.0,
        cell(2, 1): 1.0,
        cell(2, 3): 1.0
    })
    d2 = neighbors2(x).eval(session=self.session)
    self.confirm_dict(d1, d2)
    y1 = neighbors1(x.jump_to(cell(1, 1), 'place_t')) * neighbors2(x)
    y2 = neighbors2(x.jump_to(cell(1, 1), 'place_t')) & neighbors1(x)
    e1 = y1.eval(session=self.session)
    e2 = y2.eval(session=self.session)
    self.confirm_dict(e1, {cell(1, 2): 1.0, cell(2, 1): 1.0})
    self.confirm_dict(e1, e2)

  def test_jump_all(self):
    all1 = self.context.all('place_t')
    all2 = self.context.one('black', 'color_t').jump_to_all('place_t')
    self.confirm_dict(
        all1.eval(session=self.session), all2.eval(session=self.session))

  def test_jump_none(self):
    all1 = self.context.none('place_t')
    all2 = self.context.one('black', 'color_t').jump_to_none('place_t')
    self.confirm_dict(
        all1.eval(session=self.session), all2.eval(session=self.session))


if __name__ == '__main__':
  tf.test.main()
