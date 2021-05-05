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
import collections
import copy
import itertools
import json
import random
import re
from typing import Dict, Text

from absl.testing import parameterized
from language.canine import modeling
import tensorflow.compat.v1 as tf


class CanineModelTest(parameterized.TestCase, tf.test.TestCase):

  class CanineModelTester(object):

    def __init__(self,
                 parent,
                 batch_size=13,
                 dynamic_batch_size=False,
                 seq_length=16,
                 is_training=True,
                 vocab_size=99,
                 hidden_size=32,
                 num_hidden_layers=2,
                 downsampling_rate=4,
                 num_attention_heads=4,
                 intermediate_size=37,
                 max_position_embeddings=64,
                 type_vocab_size=16):
      self.parent = parent
      self.batch_size = batch_size
      self.dynamic_batch_size = dynamic_batch_size
      self.seq_length = seq_length
      self.is_training = is_training
      self.vocab_size = vocab_size
      self.hidden_size = hidden_size
      self.num_hidden_layers = num_hidden_layers
      self.downsampling_rate = downsampling_rate
      self.num_attention_heads = num_attention_heads
      self.intermediate_size = intermediate_size
      self.max_position_embeddings = max_position_embeddings
      self.type_vocab_size = type_vocab_size

    def create_model(self, seed=None):
      input_ids = ids_tensor(
          [self.batch_size, self.seq_length],
          self.vocab_size,
          seed=seed,
          dynamic_batch_size=self.dynamic_batch_size)

      if seed is not None:
        seed *= 7

      input_mask = ids_tensor(
          [self.batch_size, self.seq_length],
          vocab_size=2,
          seed=seed,
          dynamic_batch_size=self.dynamic_batch_size)

      if seed is not None:
        seed *= 5

      token_type_ids = ids_tensor(
          [self.batch_size, self.seq_length],
          self.type_vocab_size,
          seed=seed,
          dynamic_batch_size=self.dynamic_batch_size)

      config = modeling.CanineModelConfig(
          hidden_size=self.hidden_size,
          num_hidden_layers=self.num_hidden_layers,
          num_attention_heads=self.num_attention_heads,
          intermediate_size=self.intermediate_size,
          max_positions=self.max_position_embeddings,
          downsampling_rate=self.downsampling_rate)

      model = modeling.CanineModel(
          config=config,
          atom_input_ids=input_ids,
          atom_input_mask=input_mask,
          atom_segment_ids=token_type_ids,
          is_training=self.is_training)

      outputs = {
          "pooled_output": model.get_pooled_output(),
          "sequence_output": model.get_sequence_output(),
          "downsampled_layers": model.get_downsampled_layers(),
      }
      return outputs

    def check_output(self, result):
      self.parent.assertAllEqual(result["pooled_output"].shape,
                                 [self.batch_size, self.hidden_size])
      self.parent.assertAllEqual(
          result["sequence_output"].shape,
          [self.batch_size, self.seq_length, self.hidden_size])
      for layer in result["downsampled_layers"]:
        self.parent.assertEqual(layer.shape[0], self.batch_size)
        # NOTE: Not checking sequence molecule length.
        self.parent.assertEqual(layer.shape[2], self.hidden_size)

  def test_model_static_batch_size(self):
    self.run_tester(
        CanineModelTest.CanineModelTester(self), check_reachable=False)

  def test_model_dynamic_batch_size(self):
    self.run_tester(
        CanineModelTest.CanineModelTester(
            self, dynamic_batch_size=True),
        check_reachable=False)

  @parameterized.named_parameters(
      dict(testcase_name="5", rate=5),
      dict(testcase_name="6", rate=6),
      dict(testcase_name="7", rate=7))
  def test_model_downsampling_rate(self, rate):
    self.run_tester(
        CanineModelTest.CanineModelTester(self, downsampling_rate=rate),
        check_reachable=False)

  def test_config_to_json_string(self):
    config = modeling.CanineModelConfig(hidden_size=37)
    obj = json.loads(config.to_json_string())
    tf.logging.info(str(obj))
    self.assertEqual(obj["hidden_size"], 37)

  def test_determinism_same_graph(self):
    # Deterministic only at inference (training has dropout)
    tester = CanineModelTest.CanineModelTester(
        self, is_training=False)
    with self.session() as sess:
      ops = tester.create_model()
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      sess.run(init_op)
      run1 = sess.run(ops)
      run2 = sess.run(ops)
    tester.check_output(run1)
    tester.check_output(run2)
    self.assertAllClose(run1, run2)

  def run_tester(self, tester, check_reachable=True):
    with self.session() as sess:
      ops: Dict[Text, tf.Tensor] = tester.create_model()
      init_op: tf.Operation = tf.group(tf.global_variables_initializer(),
                                       tf.local_variables_initializer())
      sess.run(init_op)
      output_result = sess.run(ops)
      tester.check_output(output_result)
      if check_reachable:
        self.assert_all_tensors_reachable(sess, [init_op, ops])

  def assert_all_tensors_reachable(self, sess, outputs):
    """Checks that all the tensors in the graph are reachable from outputs."""

    ignore_strings = [
        "^.*/assert_less_equal/.*$",
        "^.*/dilation_rate$",
        "^.*/Tensordot/concat$",
        "^.*/Tensordot/concat/axis$",
        "^testing/.*$",
        # TensorContracts:
        "^Require.*$",
        "^Ensure.*$",
        "^Assert.*$",
        "^Identity.*$",
        "^ExpandDims.*$",
        "^Squeeze.*$",
        "^.*Const.*$",
        "^MaxPool2d.*$",
        "^.*Repeat.*shape$",
        "^ones.*$",
        "^concat.*$",
    ]
    ignore_pattern = "|".join(ignore_strings)
    ignore_regex = re.compile(ignore_pattern)
    unreachable = get_unreachable_ops(sess.graph, outputs)
    unreachable = [x for x in unreachable if not ignore_regex.match(x.name)]
    self.assertEmpty(
        unreachable, "The following ops are unreachable: {}".format(" ".join(
            x.name for x in unreachable)))


def ids_tensor(shape,
               vocab_size,
               dynamic_batch_size,
               rng=None,
               name=None,
               seed=None):
  """Creates a random int32 tensor of the shape within the vocab size."""
  if rng is None:
    rng = random.Random()
  if seed is not None:
    rng.seed(seed)

  total_dims = 1
  for dim in shape:
    total_dims *= dim

  values = []
  for _ in range(total_dims):
    values.append(rng.randint(0, vocab_size - 1))

  const_tensor = tf.constant(
      value=values, dtype=tf.int32, shape=shape, name=name)

  if dynamic_batch_size:
    placeholder_shape = copy.deepcopy(shape)
    if dynamic_batch_size:
      placeholder_shape[0] = None
    # Rather than having to pass back out values for the feed_dict all over
    # the place, we use placholder_with_default and always use those defaults
    # during testing.
    return tf.placeholder_with_default(const_tensor, shape=placeholder_shape)
  else:
    return const_tensor


def get_unreachable_ops(graph, outputs):
  """Finds all of the tensors in graph that are unreachable from outputs."""
  outputs = flatten_recursive(outputs)
  output_to_op = collections.defaultdict(list)
  op_to_all = collections.defaultdict(list)
  assign_out_to_in = collections.defaultdict(list)

  for op in graph.get_operations():
    for x in op.inputs:
      op_to_all[op.name].append(x.name)
    for y in op.outputs:
      output_to_op[y.name].append(op.name)
      op_to_all[op.name].append(y.name)
    if str(op.type) == "Assign":
      for y in op.outputs:
        for x in op.inputs:
          assign_out_to_in[y.name].append(x.name)

  assign_groups = collections.defaultdict(list)
  for out_name in assign_out_to_in.keys():
    name_group = assign_out_to_in[out_name]
    for n1 in name_group:
      assign_groups[n1].append(out_name)
      for n2 in name_group:
        if n1 != n2:
          assign_groups[n1].append(n2)

  seen_tensors = set()
  stack = [x.name for x in outputs]
  while stack:
    name = stack.pop()
    if name in seen_tensors:
      continue
    seen_tensors.add(name)

    if name in output_to_op:
      for op_name in output_to_op[name]:
        if op_name in op_to_all:
          for input_name in op_to_all[op_name]:
            if input_name not in stack:
              stack.append(input_name)

    stack.extend([n for n in assign_groups[name] if n not in stack])

  unreachable_ops = []
  for op in graph.get_operations():
    all_names = [x.name for x in list(op.inputs) + op.outputs]
    is_unreachable = any(name for name in all_names if name not in seen_tensors)
    if is_unreachable:
      unreachable_ops.append(op)
  return unreachable_ops


def flatten_recursive(item):
  """Flattens (potentially nested) a tuple/dictionary/list to a list."""
  output = []
  if isinstance(item, (list, tuple)):
    output.extend(item)
  elif isinstance(item, collections.Mapping):
    output.extend(item.values())
  else:
    return [item]

  return itertools.chain.from_iterable(map(flatten_recursive, output))


if __name__ == "__main__":
  tf.test.main()
