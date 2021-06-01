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
"""BERT+LSTM network for use with MuZero."""

from typing import Optional

from language.search_agents.muzero import transformer_encoder
import tensorflow as tf

from muzero import network


class BERTandLSTM(network.AbstractEncoderandLSTM):
  """BERT and LSTM model for the Search Agent."""

  def __init__(self, bert_config, type_vocab_sizes, num_float_features,
               bert_sequence_length, bert_init_ckpt: Optional[str],
               action_encoder_hidden_size: int,
               pretraining_num_unroll_steps: Optional[int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._bert_model = transformer_encoder.get_transformer_encoder(
        bert_config,
        type_vocab_sizes=type_vocab_sizes,
        num_float_features=num_float_features,
        sequence_length=bert_sequence_length,
        bert_init_ckpt=bert_init_ckpt)

    self.action_encoder_hidden_size = action_encoder_hidden_size
    if self.action_encoder_hidden_size > 0:
      self.rnn_action_encoder = tf.keras.layers.RNN(
          tf.keras.layers.GRUCell(self.action_encoder_hidden_size))

    self.pretraining_num_unroll_steps = pretraining_num_unroll_steps

  def _encode_observation(self, observation, training=True):
    token_ids, type_ids, float_features, action_history = observation
    mask = tf.cast(tf.not_equal(token_ids, 0), tf.int32)
    _, cls_output = self._bert_model(
        (token_ids, mask, type_ids, float_features), training=training)
    encoding = cls_output
    if self.action_encoder_hidden_size > 0:
      one_hot_actions = tf.one_hot(
          action_history,
          depth=self._parametric_action_distribution.param_size + 1,
          axis=-1)
      actions_mask = tf.math.less(
          action_history, self._parametric_action_distribution.param_size)
      actions_encoding = self.rnn_action_encoder(
          one_hot_actions, mask=actions_mask)
      encoding = tf.concat([actions_encoding, encoding], axis=-1)
    return encoding

  def pretraining_loss(self, sample, training=True):
    observation, action, reward, value, _ = sample[:5]
    output = self.initial_inference(observation, training=training)
    loss = 0.
    policy_accuracy = tf.keras.metrics.categorical_accuracy(
        tf.one_hot(action, output.policy_logits.shape[-1]),
        output.policy_logits)
    value_accuracy = tf.abs(tf.squeeze(output.value) - value)
    reward_accuracy = 0.
    for i in range(self.pretraining_num_unroll_steps):
      _, action, reward, value, mask = sample[i * 5:(i + 1) * 5]
      policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=output.policy_logits, labels=action) * mask
      value_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=output.value_logits,
          labels=self.value_encoder.encode(value)) * mask

      output = self.recurrent_inference(
          output.hidden_state, action, training=training)
      reward_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=output.reward_logits,
          labels=self.reward_encoder.encode(reward)) * mask
      if i == 0:
        reward_accuracy = tf.abs(tf.squeeze(output.reward) - reward)

      loss += policy_loss + value_loss + reward_loss

    return loss, {
        'losses/policy': policy_loss,
        'losses/value': value_loss,
        'losses/reward': reward_loss,
        'accuracies/policy': policy_accuracy,
        'accuracies/value': value_accuracy,
        'accuracies/reward': reward_accuracy
    }
