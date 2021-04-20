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


from language.canine import tensor_contracts as tc
import tensorflow.compat.v1 as tf


class TensorContractsTest(tf.test.TestCase):

  def test_static_rank(self):
    @tc.contract(tc.Require("t", rank=1))
    def my_func(t):
      del t  # Unused.

    # Passes:
    my_func(tf.constant([0]))

    # Fails:
    with self.assertRaises(ValueError):
      my_func(tf.constant([[0]]))

  def test_static_shape(self):
    @tc.contract(tc.Require("t", shape=[1, 2]))
    def my_func(t):
      del t  # Unused.

    # Passes:
    my_func(tf.constant([[0, 1]]))

    # Fails:
    with self.assertRaises(ValueError):
      my_func(tf.constant([[0]]))

  def test_dynamic_shape(self):
    @tc.contract(tc.Require("t", shape=[1, 2]))
    def my_func(t):
      return t

    input_tensor = tf.placeholder(tf.float32, shape=[None, None])
    result = my_func(input_tensor)
    with self.session() as sess:
      # Passes:
      sess.run(result, feed_dict={input_tensor: [[0, 1]]})

    input_tensor = tf.placeholder(tf.float32)
    with self.assertRaises(ValueError):
      result = my_func(input_tensor)
      with self.session() as sess:
        # Fails:
        sess.run(result, feed_dict={input_tensor: [[0, 1, 2]]})

  def test_named_dim(self):
    @tc.contract(
        tc.Require("a", shape=["seq_length"]),
        tc.Require("b", shape=["seq_length"]),
        tc.NamedDim("seq_length", "a", 0))
    def my_func(a, b):
      del a  # Unused.
      del b  # Unused.

    # Passes:
    my_func(tf.constant([0]), tf.constant([1]))

    # Fails:
    with self.assertRaises(ValueError):
      my_func(tf.constant([0]), tf.constant([1, 2]))

  def test_dotted_name(self):
    """Tests that we can extract `obj.t`."""

    class Obj(object):

      def __init__(self):
        self.t = tf.constant([[0, 1]])

    @tc.contract(tc.Require("obj.t", shape=[1, 2]))
    def my_func(obj):
      del obj  # Unused.

    my_func(Obj())

  def test_require_true(self):
    """Tests that we can check arbitrary functions in `RequireTrue`."""

    def _greater(a, b):
      return tf.greater(a, b)

    @tc.contract(tc.RequireTrue(_greater, tensors=["a", "b"], error="Fail."))
    def my_func(a, b):
      return a + b

    small = tf.constant([1, 2])
    big = tf.constant([3, 4])

    result = my_func(big, small)
    with self.session() as sess:
      # Passes:
      sess.run(result)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = my_func(small, big)
      with self.session() as sess:
        # Fails:
        sess.run(result)

  def test_ensure_true(self):
    """Tests that we can check arbitrary functions in `EnsureTrue`."""

    def _nonzero(a):
      return tf.not_equal(a, 0)

    # TODO(jhclark): Notice that the syntax is currently insufficient for
    # handling tuple return types with `EnsureTrue`.
    @tc.contract(tc.EnsureTrue(_nonzero, tensors=["RESULT"], error="Fail."))
    def my_func(a):
      return a

    a = tf.constant([1, 2])
    result = my_func(a)
    with self.session() as sess:
      # Passes:
      sess.run(result)

    a = tf.constant([0, 1])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = my_func(a)
      with self.session() as sess:
        # Fails:
        sess.run(result)

  def test_varargs(self):
    @tc.contract(tc.Require("t", shape=[1, 2]))
    def my_func(t, *args):
      del t  # Unused.
      del args  # Unused.
    my_func(tf.constant([[0, 1]]))

  def test_kwargs(self):
    @tc.contract(tc.Require("t", shape=[1, 2]))
    def my_func(t, **kwargs):
      del t  # Unused.
      del kwargs  # Unused.
    my_func(tf.constant([[0, 1]]))

  def test_local_invariant(self):
    t = tf.constant([1])

    # Passes:
    with tc.local_invariant(tc.Require("t", rank=1)):
      _ = tf.identity(t)

    # Fails:
    with self.assertRaises(ValueError):
      with tc.local_invariant(tc.Require("t", rank=2)):
        _ = tf.identity(t)

  def test_dynamic_condition(self):

    def _condition_generator(a, b):
      result = []
      for i, (a_item, b_item) in enumerate(zip(a, b)):
        # Test NamedDims.
        result.append(tc.NamedDim("length", var=a, var_name="a", dim=0))
        # Test static Require.
        result.append(tc.Require(var=a, var_name="a", shape=["length"]))
        result.append(tc.Require(var=b, var_name="b", shape=["length"]))
        # Test RequireTrue conditions.
        result.append(
            tc.RequireTrue(
                tf.greater,
                tensors=[("a[{}]".format(i), a_item),
                         ("b[{}]".format(i), b_item)],
                error="a_item > b_item."))
      return result

    @tc.contract(tc.Dynamic(_condition_generator, "a", "b"))
    def my_func(a, b):
      return a[0] + b[0]

    small = [tf.constant([1, 2])]
    big = [tf.constant([3, 4])]

    result = my_func(big, small)
    with self.session() as sess:
      # Passes:
      sess.run(result)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = my_func(small, big)
      with self.session() as sess:
        # Fails:
        sess.run(result)

  def test_constructor_static_rank(self):
    class MyClass(object):

      @tc.contract(tc.Require("t", rank=1))
      def __init__(self, t):
        del t  # Unused.

    # Passes:
    MyClass(tf.constant([0]))

    # Fails:
    with self.assertRaises(ValueError):
      MyClass(tf.constant([[0]]))

  def test_static_rank_optional(self):
    @tc.contract(tc.Require("t", rank=1, optional=True))
    def my_func(t):
      del t  # Unused.

    # Passes:
    my_func(tf.constant([0]))
    my_func(None)

    # Fails:
    with self.assertRaises(ValueError):
      my_func(tf.constant([[0]]))

  def test_self_property_access(self):
    class MyClass(object):

      def __init__(self, t, desired_dim):
        self.t = t
        self.desired_dim = desired_dim

      @tc.contract(
          tc.Require("self.t", shape=["dim1"]),
          tc.Ensure(tc.RESULT, shape=["dim2"]),
          tc.NamedDim("dim1", "self.t", 0),
          tc.NamedDim("dim2", value_of="self.desired_dim"))
      def my_func(self):
        return self.t

    # Passes:
    c = MyClass(tf.constant([0]), desired_dim=tf.constant(1))
    result = c.my_func()
    with self.session() as sess:
      sess.run(result)

    # Fails:
    with self.assertRaises(tf.errors.InvalidArgumentError):
      c = MyClass(tf.constant([0, 1]), desired_dim=tf.constant(1))
      result = c.my_func()
      with self.session() as sess:
        sess.run(result)

  def test_static_dims(self):
    @tc.contract(tc.Require("t", static_dims=[0]))
    def my_func(t):
      return t

    # Passes:
    static_dim = tf.constant([0])
    result = my_func(static_dim)
    with self.session() as sess:
      sess.run(result)

    # Fails:
    dynamic_dim = tf.placeholder(tf.float32, shape=[None])
    with self.assertRaises(ValueError):
      result = my_func(dynamic_dim)
      with self.session() as sess:
        sess.run(result)


if __name__ == "__main__":
  tf.test.main()
