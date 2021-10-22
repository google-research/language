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
"""The MuZero search agent."""

from typing import Optional

import dataclasses

from language.search_agents.muzero import common_flags
from language.search_agents.muzero import network
from seed_rl.common import parametric_distribution

from muzero import actor
from muzero import core as mzcore
from muzero import learner_flags


@dataclasses.dataclass(frozen=True)
class AgentConfig:
  value_encoder_steps: int
  reward_encoder_steps: Optional[int]

  lstm_size: int
  n_lstm_layers: int

  head_hidden_size: int
  n_head_hidden_layers: int


def agent_config_from_flags() -> AgentConfig:
  return AgentConfig(
      value_encoder_steps=common_flags.VALUE_ENCODER_STEPS.value,
      reward_encoder_steps=common_flags.REWARD_ENCODER_STEPS.value,
      lstm_size=common_flags.LSTM_SIZE.value,
      n_lstm_layers=common_flags.N_LSTM_LAYERS.value,
      head_hidden_size=common_flags.HEAD_HIDDEN_SIZE.value,
      n_head_hidden_layers=common_flags.N_HEAD_HIDDEN_LAYERS.value)


def muzeroconfig_from_flags(
    env_descriptor: mzcore.EnvironmentDescriptor) -> mzcore.MuZeroConfig:
  """Initializes a MuZeroConfig from flags and `env_descriptor`."""

  def visit_softmax_temperature(num_moves, training_steps, is_training=True):  # pylint: disable=unused-argument
    del training_steps
    if common_flags.PLAY_MAX_AFTER_MOVES.value < 0:
      return common_flags.TEMPERATURE.value
    if num_moves < common_flags.PLAY_MAX_AFTER_MOVES.value:
      return common_flags.TEMPERATURE.value
    else:
      return 0.

  # Known bounds for Q-values have to include rewards and values.
  known_bounds = mzcore.KnownBounds(
      *map(sum, zip(env_descriptor.reward_range, env_descriptor.value_range)))
  return mzcore.MuZeroConfig(
      action_space_size=env_descriptor.action_space.n,  # pytype: disable=attribute-error
      max_moves=env_descriptor.extras['max_episode_length'] * 2,
      discount=1.0 - common_flags.ONE_MINUS_DISCOUNT.value,
      dirichlet_alpha=common_flags.DIRICHLET_ALPHA.value,
      root_exploration_fraction=common_flags.ROOT_EXPLORATION_FRACTION.value,
      num_simulations=common_flags.NUM_SIMULATIONS.value,
      recurrent_inference_batch_size=(
          learner_flags.RECURRENT_INFERENCE_BATCH_SIZE.value),
      initial_inference_batch_size=(
          learner_flags.INITIAL_INFERENCE_BATCH_SIZE.value),
      train_batch_size=learner_flags.BATCH_SIZE.value,
      # Not using time difference learning.
      td_steps=-1,
      num_unroll_steps=common_flags.NUM_UNROLL_STEPS.value,
      pb_c_base=common_flags.PB_C_BASE.value,
      pb_c_init=common_flags.PB_C_INIT.value,
      known_bounds=known_bounds,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      use_softmax_for_action_selection=(
          common_flags.USE_SOFTMAX_FOR_ACTION_SELECTION.value == 1),
      max_num_action_expansion=actor.MAX_NUM_ACTION_EXPANSION.value)


def create_agent(env_descriptor: mzcore.EnvironmentDescriptor,
                 parametric_action_distribution: parametric_distribution
                 .ParametricDistribution,
                 agent_config: AgentConfig) -> network.BERTandLSTM:
  """Creates an agent."""
  if agent_config.reward_encoder_steps is None:
    reward_encoder_steps = agent_config.value_encoder_steps
  else:
    reward_encoder_steps = agent_config.reward_encoder_steps

  reward_encoder = mzcore.ValueEncoder(
      min_value=env_descriptor.reward_range.low,
      max_value=env_descriptor.reward_range.high,
      num_steps=reward_encoder_steps,
      use_contractive_mapping=False,
  )
  value_encoder = mzcore.ValueEncoder(
      min_value=env_descriptor.value_range.low,
      max_value=env_descriptor.value_range.high,
      num_steps=agent_config.value_encoder_steps,
      use_contractive_mapping=False,
  )

  return network.BERTandLSTM(
      bert_config=env_descriptor.extras['bert_config'],
      type_vocab_sizes=env_descriptor.extras['type_vocab_sizes'],
      num_float_features=env_descriptor.extras['num_float_features'],
      bert_sequence_length=env_descriptor.extras['sequence_length'],
      bert_init_ckpt=env_descriptor.extras['bert_init_ckpt'],
      action_encoder_hidden_size=env_descriptor
      .extras['action_encoder_hidden_size'],
      parametric_action_distribution=parametric_action_distribution,
      rnn_sizes=[agent_config.lstm_size] * agent_config.n_lstm_layers,
      head_hidden_sizes=[agent_config.head_hidden_size] *
      agent_config.n_head_hidden_layers,
      reward_encoder=reward_encoder,
      value_encoder=value_encoder,
      pretraining_num_unroll_steps=env_descriptor
      .extras['pretraining_num_unroll_steps'],
      head_relu_before_norm=True,
      nonlinear_to_hidden=True,
      recurrent_activation='relu',
      embed_actions=(common_flags.EMBED_ACTIONS.value == 1))
