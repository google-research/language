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
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
"""MuZero."""

import functools

from absl import app
from absl import flags
from absl import logging
from language.search_agents.muzero import agent_lib
from language.search_agents.muzero import common_flags
from language.search_agents.muzero import env
from seed_rl.common import common_flags as seed_common_flags  # pylint: disable=unused-import
import tensorflow as tf

from muzero import actor
from muzero import learner
from muzero import learner_flags


FLAGS = flags.FLAGS


def create_optimizer(unused_final_iteration):
  if common_flags.LR_WARM_RESTARTS.value:
    learning_rate_fn = tf.keras.experimental.CosineDecayRestarts(
        common_flags.LEARNING_RATE.value,
        common_flags.LR_DECAY_STEPS.value,
        t_mul=2.0,
        m_mul=1.0,
        alpha=common_flags.LR_DECAY_FRACTION.value)
  else:
    learning_rate_fn = tf.keras.experimental.CosineDecay(
        common_flags.LEARNING_RATE.value,
        common_flags.LR_DECAY_STEPS.value,
        alpha=common_flags.LR_DECAY_FRACTION.value)
  if common_flags.OPTIMIZER.value == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate_fn, momentum=common_flags.MOMENTUM.value)
  elif common_flags.OPTIMIZER.value == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
  elif common_flags.OPTIMIZER.value == 'adagrad':
    optimizer = tf.keras.optimizers.AdaGrad(learning_rate_fn)
  elif common_flags.OPTIMIZER.value == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate_fn, momentum=common_flags.MOMENTUM.value)
  else:
    raise ValueError('Unknown optimizer: {}'.format(FLAGS.optimizer))
  return optimizer, learning_rate_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  env_descriptor = env.get_descriptor()

  mzconfig = agent_lib.muzeroconfig_from_flags(env_descriptor=env_descriptor)

  create_agent_fn = functools.partial(
      agent_lib.create_agent, agent_config=agent_lib.agent_config_from_flags())

  if FLAGS.run_mode == 'actor':
    logging.info('Make actor, %s/%s', FLAGS.task, FLAGS.num_envs)
    actor.actor_loop(
        functools.partial(
            env.create_environment,
            stop_after_seeing_new_results=common_flags
            .STOP_AFTER_SEEING_NEW_RESULTS.value > 0), mzconfig)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(
        env_descriptor=env_descriptor,
        create_agent_fn=create_agent_fn,
        create_optimizer_fn=create_optimizer,
        config=learner_flags.learner_config_from_flags(),
        mzconfig=mzconfig)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
