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
"""A decoder for tf.SequenceExample."""

import tensorflow.compat.v1 as tf
import tf_slim as slim


class TFSequenceExampleDecoder(slim.data_decoder.DataDecoder):
  """A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  and (2) tensor manipulation.
  In the first stage, the tf.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors.
  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.
  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, context_keys_to_features, sequence_keys_to_features,
               items_to_handlers):
    """Constructs the TFSequenceExampleDecoder decoder."""
    self._context_keys_to_features = context_keys_to_features
    self._sequence_keys_to_features = sequence_keys_to_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return list(self._items_to_handlers.keys())

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-example."""
    context, sequence = tf.parse_single_sequence_example(
        serialized_example, self._context_keys_to_features,
        self._sequence_keys_to_features)

    # Merge context and sequence features
    example = {}
    example.update(context)
    example.update(sequence)

    all_features = {}
    all_features.update(self._context_keys_to_features)
    all_features.update(self._sequence_keys_to_features)

    # Reshape non-sparse elements just once:
    for k, value in all_features.items():
      if isinstance(value, tf.FixedLenFeature):
        example[k] = tf.reshape(example[k], value.shape)

    if not items:
      items = list(self._items_to_handlers.keys())

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {key: example[key] for key in handler.keys}
      outputs.append(handler.tensors_to_item(keys_to_tensors))

    return outputs
