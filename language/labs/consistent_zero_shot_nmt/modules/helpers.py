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
"""A library of helpers for custom decoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    "ContinuousEmbeddingTrainingHelper",
    "ScheduledContinuousEmbeddingTrainingHelper",
    "GreedyContinuousEmbeddingHelper",
    "FixedContinuousEmbeddingHelper",
]


def _unstack_ta(inp):
  return tf.TensorArray(
      dtype=inp.dtype, size=tf.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


class ContinuousEmbeddingTrainingHelper(tf.contrib.seq2seq.TrainingHelper):
  """Regards previous outputs as the next input embeddings. Avoids sampling.

  By doing so, the decoded sequences are differentiable throughout.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def sample(self, time, outputs, state, name=None):
    with tf.name_scope(name, "TrainingHelperSample", [time, outputs, state]):
      if isinstance(state, tuple):
        # TODO(alshedivat): fix the if statement as it works only with GNMT.
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
      else:
        sample_ids = tf.cast(tf.argmax(state.alignments, axis=-1), tf.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name, "TrainingHelperNextInputs", [time, outputs]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = tf.reduce_all(finished)
      next_inputs = tf.cond(
          all_finished, lambda: self._zero_inputs, lambda: outputs)
      return finished, next_inputs, state


class ScheduledContinuousEmbeddingTrainingHelper(
    tf.contrib.seq2seq.TrainingHelper):
  """Training helper that constructs next inputs using scheduled mixing.

  The hlper mixes previous outputs with the true ground truth embeddings for the
  previous time step using `sampling_probability` as the mixing weight for the
  ground truth, i.e.:

      next_inputs = weight * ground_truth + (1 - weight) * generated
  """

  def __init__(self, inputs, sequence_length, mixing_concentration=1.,
               time_major=False, seed=None, scheduling_seed=None, name=None):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      mixing_concentration: <float32> [] for the alpha parameter of the
        Dirichlet distribution used to sample mixing weights from [0, 1].
      time_major: Python bool. Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      seed: The sampling seed.
      scheduling_seed: The schedule decision rule sampling seed.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sampling_probability` is not a scalar or vector.
    """
    with tf.name_scope(name, "ScheduledContinuousEmbedding",
                       [mixing_concentration]):
      self._mixing_concentration = tf.convert_to_tensor(
          mixing_concentration, name="mixing_concentration")
      if self._mixing_concentration.get_shape().ndims == 0:
        self._mixing_concentration = tf.expand_dims(self._mixing_concentration,
                                                    0)
      if (self._mixing_concentration.get_shape().ndims != 1 or
          self._mixing_concentration.get_shape().as_list()[0] > 1):
        raise ValueError(
            "mixing_concentration must be a scalar. saw shape: %s" %
            (self._mixing_concentration.get_shape()))
      self._seed = seed
      self._scheduling_seed = scheduling_seed
      super(ScheduledContinuousEmbeddingTrainingHelper, self).__init__(
          inputs=inputs,
          sequence_length=sequence_length,
          time_major=time_major,
          name=name)

  def sample(self, time, outputs, state, name=None):
    with tf.name_scope(name, "TrainingHelperSample", [time, outputs, state]):
      if isinstance(state, tuple):
        # TODO(alshedivat): fix the if statement as it works only with GNMT.
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
      else:
        sample_ids = tf.cast(tf.argmax(state.alignments, axis=-1), tf.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    del sample_ids  # Unused.
    with tf.name_scope(name, "ScheduledContinuousEmbeddingNextInputs",
                       [time, outputs, state]):
      # Get ground truth next inputs.
      (finished, base_next_inputs,
       state) = tf.contrib.seq2seq.TrainingHelper.next_inputs(
           self, time, outputs, state, name=name)

      # Get generated next inputs.
      all_finished = tf.reduce_all(finished)
      generated_next_inputs = tf.cond(
          all_finished,
          # If we're finished, the next_inputs value doesn't matter
          lambda: outputs,
          lambda: outputs)

      # Sample mixing weights.
      weight_sampler = tf.distributions.Dirichlet(
          concentration=self._mixing_concentration)
      weight = weight_sampler.sample(
          sample_shape=self.batch_size, seed=self._scheduling_seed)
      alpha, beta = weight, 1 - weight

      # Mix the inputs.
      next_inputs = alpha * base_next_inputs + beta * generated_next_inputs

      return finished, next_inputs, state


class GreedyContinuousEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """Greedy decoding helper with continuous reuse of embeddings."""

  def sample(self, time, outputs, state, name=None):
    del time  # Unused.
    if isinstance(state, tuple):
      # TODO(alshedivat): fix the if statement as it works only with GNMT.
      sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
    else:
      sample_ids = tf.cast(tf.argmax(state.alignments, axis=-1), tf.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    del time  # Unused.
    finished = tf.equal(sample_ids, self._end_token)
    all_finished = tf.reduce_all(finished)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: outputs)
    return finished, next_inputs, state


class FixedContinuousEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """Decodes for a fixed number of steps and continuously reuses embeddings."""

  def __init__(self, embedding, start_tokens, end_token, num_steps):
    super(FixedContinuousEmbeddingHelper, self).__init__(
        embedding, start_tokens, end_token)
    self._num_steps = num_steps

  def sample(self, time, outputs, state, name=None):
    del time  # Unused.
    if isinstance(state, tuple):
      # TODO(alshedivat): fix the if statement as it works only with GNMT.
      sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
    else:
      sample_ids = tf.cast(tf.argmax(state.alignments, axis=-1), tf.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    next_time = time + 1
    finished = (next_time >= self._num_steps)
    all_finished = tf.reduce_all(finished)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: outputs)
    return finished, next_inputs, state
