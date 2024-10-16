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
"""Run Jax experiments."""

import functools
import json
import math
import os
import time
from typing import Any, Mapping, Sequence, Text

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import platform
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import memory_generation_task
from language.mentionmemory.utils import checkpoint_utils
from language.mentionmemory.utils import data_utils
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

# Training config.
flags.DEFINE_string('config', None, 'Train configuration serialized as JSON.')
flags.DEFINE_string(
    'config_file', None,
    'Path to file that contains JSON serialized model configuration. If this'
    "is specified, ignores 'config' parameter.")
flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')


def get_data_iterator(config):
  """Get iterator over the dataset."""
  task = memory_generation_task.MemoryGenerationTask

  # Establish host information
  local_device_count = jax.local_device_count()
  host_count = jax.process_count()
  host_id = jax.process_index()

  # Load datasets
  logging.info('Loading dataset.')
  decode_fn = data_utils.make_decode_fn(
      name_to_features=task.get_name_to_features(config),
      samples_per_example=config.samples_per_example,
  )

  preprocess_fn = task.make_preprocess_fn(config)
  collater_fn = task.make_collater_fn(config)

  data = data_utils.load_dataset(
      patterns=config.data_patterns,
      decode_fn=decode_fn,
      preprocess_fn=preprocess_fn,
      collater_fn=collater_fn,
      is_training=False,
      per_device_batch_size=config.per_device_batch_size,
      local_device_count=local_device_count,
      host_count=host_count,
      host_id=host_id,
      seed=0,
  )
  return iter(data)


def get_num_total_memories(config):
  """Computes the total number of mentions in the corpus."""
  logging.info('Estimating the total number of memories to be generated.')
  data_iter = get_data_iterator(config)
  start_time = time.time()
  num_total_memories = 0
  for batch in data_iter:
    batch_jax = jax.tree.map(np.asarray, batch)
    num_total_memories += batch_jax['mention_target_weights'].sum()
  logging.info('Found %d memories in %.2f seconds', num_total_memories,
               time.time() - start_time)
  return num_total_memories


def generate(config: ml_collections.ConfigDict):
  """Generates memories."""
  # Establish host information
  local_device_count = jax.local_device_count()
  device_count = jax.device_count()
  process_count = jax.process_count()
  process_index = jax.process_index()

  task = memory_generation_task.MemoryGenerationTask
  model_config = ml_collections.FrozenConfigDict(config.model_config)
  model = task.build_model(model_config)
  p_predict_step = jax.pmap(
      functools.partial(
          task.make_prediction_fn(config),
          model_config,
      ),
      axis_name='batch')
  rng = jax.random.PRNGKey(config.seed)

  # Initialization needs to be pmapped because models use collective ops.
  # Create dummy input
  dummy_input = {
      key: jnp.tile(value, (local_device_count,) + (1,) * value.ndim)
      for key, value in task.dummy_input(config).items()
  }

  rng, init_rng = jax.random.split(rng)
  init_rng = jax.random.split(init_rng, local_device_count)

  logging.info('Initializing model.')
  initial_variables = jax.pmap(
      model.init, 'batch', static_broadcasted_argnums=2)(init_rng, dummy_input,
                                                         True)
  logging.info('Finished initializing model.')
  initial_variables = initial_variables.unfreeze()

  if config.load_weights is not None:
    logging.info('Loading model weights from file')
    loaded_variables = task.load_weights(config)
    unexpected, missing = checkpoint_utils.merge_nested_dicts(
        initial_variables, loaded_variables)
    logging.info('*** Unexpected features: ***')
    for feature_name in unexpected:
      logging.info('\t%s', feature_name)
    # In the prediction mode we don't allow any features to be missing
    # pylint: disable=g-explicit-length-test
    if len(missing) > 0:
      raise ValueError('Missing features: %s' % ','.join(missing))

  # model_params = jax_utils.unreplicate(initial_variables['params'])
  model_params = initial_variables['params']
  model_vars = {
      key: value for key, value in initial_variables.items() if key != 'params'
  }
  # We access model params only from train state.
  del initial_variables

  writer = metric_writers.create_default_writer(
      config.output_dir, just_logging=process_index > 0)

  max_length = config.get('max_length_with_entity_tokens',
                          model_config.encoder_config.max_length)

  num_total_memories = math.ceil(config.num_total_memories / process_count)
  memory_saver = memory_generation_task.MemorySaver(
      num_total_memories=num_total_memories,
      memory_dim=config.memory_dim,
      max_length=max_length,
      max_mentions_per_sample=config.max_mentions_per_sample,
      memory_key_dim=config.get('memory_key_dim'))
  n_samples = 0
  data_iter = get_data_iterator(config)

  logging.info('Start memory generation.')
  with metric_writers.ensure_flushes(writer):
    for step, batch in enumerate(data_iter):
      batch = jax.tree.map(jnp.asarray, batch)
      predictions = p_predict_step(
          model_params,
          model_vars,
          batch,
      )
      predictions = jax.device_get(predictions)
      memory_saver.add_memories(batch, predictions)
      n_devices, batch_size, _ = batch['text_ids'].shape
      logging.log_first_n(
          logging.INFO, 'Process %d / %d: '
          'Finished generating step %d, local devices %d, batch size %d', 5,
          process_index, process_count, step, n_devices, batch_size)

      n_samples += device_count * config.per_device_batch_size
      if (step % config.log_every_steps == 0 or
          memory_saver.get_num_memories() >= num_total_memories):
        writer.write_scalars(
            step,
            dict(
                n_memories=memory_saver.get_num_memories(),
                n_samples=n_samples))

      if memory_saver.get_num_memories() >= num_total_memories:
        break

  logging.info('Process %d / %d: Finished generating memories: %d out of %d',
               process_index, process_count, memory_saver.get_num_memories(),
               num_total_memories)

  start_time = time.time()
  logging.info('Process %d / %d: Start saving generated memories to files.',
               process_index, process_count)
  memory_saver.save(
      config.output_dir,
      num_shards=config.num_shards,
      stride=process_count,
      offset=process_index,
      shard_size_divisible=config.shard_size_divisible)

  logging.info(
      'Process %d / %d: Finished saving generated memories to files in %.2f seconds',
      process_index, process_count,
      time.time() - start_time)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info('JAX total devices: %r', jax.device_count())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.output_dir, 'output_dir')

  tf.io.gfile.makedirs(FLAGS.output_dir)

  # Process config here
  if FLAGS.config_file:
    with tf.io.gfile.GFile(FLAGS.config_file, 'r') as reader:
      config = json.load(reader)
  else:
    config = json.loads(FLAGS.config)
    # # Save config to workdir if it's not yet exists
    if jax.process_index() == 0:
      config_file = os.path.join(FLAGS.output_dir, 'config.json')
      with tf.io.gfile.GFile(config_file, 'w') as writer:
        writer.write(json.dumps(config, indent=4))

  config['output_dir'] = FLAGS.output_dir

  if 'num_total_memories' not in config:
    config['num_total_memories'] = get_num_total_memories(
        ml_collections.ConfigDict(config))

  generate(ml_collections.ConfigDict(config))


def validate_config_flags(flag_dict: Mapping[Text, Any]) -> bool:
  return flag_dict['config'] is not None or flag_dict['config_file'] is not None


if __name__ == '__main__':
  flags.register_multi_flags_validator(['config', 'config_file'],
                                       validate_config_flags,
                                       'Either --config or --config_file needs '
                                       'to be set.')
  app.run(main)
