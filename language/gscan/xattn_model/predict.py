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
"""Methods for testing the model using JAX."""

import collections
import functools
import json
import os

from absl import logging
from clu import checkpoint
from clu import metric_writers

import flax.jax_utils as flax_utils
import jax
import jax.experimental.optimizers
import jax.numpy as jnp

from language.gscan.xattn_model import evaluation
from language.gscan.xattn_model import train_utils
from language.gscan.xattn_model.dataset import input_pipeline
from language.gscan.xattn_model.model import decode
from language.gscan.xattn_model.model import models

import numpy as np
import tensorflow as tf


def remove_pad(x):
  """Remove padding examples."""
  if 'mask' in x:
    ind = jnp.where(jnp.array(x.pop('mask'), dtype=jnp.int32) > 0)
    x = jax.tree_map(lambda v: v[ind], x)  # pylint: disable=cell-var-from-loop
  return x


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""

  def single_tohost(x):
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))

  return jax.tree_map(single_tohost, x)


def remove_special_tokens(tokens, eos_idx):
  if eos_idx in tokens:
    seq_len = tokens.index(eos_idx)
  else:
    seq_len = len(tokens)
  tokens = tokens[1:seq_len]
  return tokens


def array_to_sentence(array, vocab):
  return [vocab['idx_to_word'][w] for w in array]


def predict_step(batch, state, cache, eos_idx, config):
  """Compute the prediction for the given model in inference mode."""

  logging.info('predict_step(batch=%s)', batch)
  variables = {'params': state.optimizer.target}
  model = models.Model(config)
  encoded, encoded_mask = model.apply(
      variables, batch, method=models.Model.encode)

  encoded_inputs = decode.flat_batch_beam_expand(encoded, config.beam_size)
  encoded_mask = decode.flat_batch_beam_expand(encoded_mask, config.beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = model.apply(
        {
            'params': state.optimizer.target,
            'cache': flat_cache
        },
        flat_ids,
        encoded_inputs,
        flat_ids > 0,
        encoded_mask,
        mutable=['cache'],
        method=models.Model.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _, = decode.beam_search(
      batch['token'],
      cache,
      tokens_ids_to_logits,
      beam_size=config.beam_size,
      alpha=0.6,
      eos_id=eos_idx,
      max_decode_len=config.max_decode_step)
  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence.
  return beam_seqs[:, -1]


def init_cache(inputs, config):
  init_vars = models.Model(config).init(jax.random.PRNGKey(0), inputs)
  return init_vars['cache']


def evaluate_sequence_accuracy(p_pred_step,
                               p_init_cache,
                               state,
                               ds,
                               config,
                               split,
                               workdir,
                               num_eval_steps=-1):
  """Evaluate classification on the given dataset."""
  prediction_dir = os.path.join(workdir, 'predictions')
  tf.io.gfile.makedirs(prediction_dir)
  logging.info('Starting evaluating sequence accuracy on %s split.', split)
  outputs = []

  test_metrics = collections.defaultdict(list)

  data_dir = config.dataset.data_dir
  input_vocab_file = os.path.join(data_dir, 'training_input_vocab.txt')
  target_vocab_file = os.path.join(data_dir, 'training_target_vocab.txt')
  dataset_file = os.path.join(data_dir, 'dataset.txt')
  eos_idx = config.dataset.eos_idx

  with tf.io.gfile.GFile(input_vocab_file, 'r') as f:
    input_vocab = json.load(f)
  with tf.io.gfile.GFile(target_vocab_file, 'r') as f:
    target_vocab = json.load(f)
  with tf.io.gfile.GFile(dataset_file, 'r') as f:
    annotations = json.load(f)

  for step, batch in enumerate(ds):  # pytype: disable=wrong-arg-types
    batch = jax.tree_map(np.asarray, batch)
    cache = p_init_cache(batch)
    batch['predictions'] = p_pred_step(batch, state, cache, eos_idx)
    batch = remove_pad(tohost(batch))
    target_token = batch['target_token']
    predictions = batch['predictions']
    for i, (prediction, target) in enumerate(zip(predictions, target_token)):
      prediction = remove_special_tokens(prediction.tolist(), eos_idx)
      target = remove_special_tokens(target.tolist(), eos_idx)
      acc = evaluation.sequence_accuracy(prediction, target)
      test_metrics['test_accuracy'].append(acc)
      exact_match = 100 if acc == 100 else 0
      test_metrics['test_exact_match'].append(exact_match)

      input_command = remove_special_tokens(batch['token'][i].tolist(), eos_idx)
      index = int(batch['index'][i][0])
      example = annotations['examples'][split][index]
      outputs.append({
          'split': split,
          'index': index,
          'input': array_to_sentence(input_command, input_vocab),
          'prediction': array_to_sentence(prediction, target_vocab),
          'target': array_to_sentence(target, target_vocab),
          'derivation': [example['derivation']],
          'situation': [example['situation']],
          'accuracy': acc,
          'exact_match': True if acc == 100 else False,
          'attention_weights_input': [],
          'attention_weights_situation': [],
      })
    if num_eval_steps > 0 and step + 1 == num_eval_steps:
      break
  test_metrics = {k: sum(v) / len(v) for k, v in test_metrics.items()}
  step = flax_utils.unreplicate(state).step
  out_path = os.path.join(prediction_dir, f'{split}_predict_{step}.json')
  with tf.io.gfile.GFile(out_path, 'w') as f:
    json.dump(outputs, f, indent=2)
  return test_metrics


def predict_and_evaluate(config, workdir, ckpt_path=None):
  """Runs a testing and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    ckpt_path: The checkpoint to evaluate. If not specified, use the latest
      checkpoint.
  """
  logging.info('Starting testing at %s', workdir)
  tf.io.gfile.makedirs(workdir)

  rng = jax.random.PRNGKey(config.seed)
  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  test_ds = []
  for split in config.dataset.test_splits:
    ds = input_pipeline.create_val_dataset(
        config.dataset, split, config.dataset.test_per_device_batch_size,
        config.dataset.test_pad_last_batch)
    test_ds.append(ds)

  # Initialize model.
  inputs = train_utils.get_init_inputs(test_ds[0])
  rng, model_rng = jax.random.split(rng)
  predict_config = models.TransformerConfig(**config.model.to_dict())
  predict_config = predict_config.replace(decode=True)
  model = models.Model(predict_config)
  state = train_utils.create_train_state(
      model, config, model_rng, inputs=inputs)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=3)

  logging.info('Testing and evaluating checkpoint %s', ckpt_path)
  try:
    state = ckpt.restore(state, ckpt_path)
  except FileNotFoundError:
    state = ckpt.restore_or_initialize(state)
  step = int(state.step)

  p_pred_step = jax.pmap(
      functools.partial(predict_step, config=predict_config),
      axis_name='batch',
      static_broadcasted_argnums=(3,))
  p_init_cache = jax.pmap(
      functools.partial(init_cache, config=predict_config), axis_name='batch')

  # Distribute testing.
  state = flax_utils.replicate(state)
  with metric_writers.ensure_flushes(writer):
    test_metrics = {}
    for ds, split in zip(test_ds, config.dataset.test_splits):
      ds_metrics = evaluate_sequence_accuracy(p_pred_step, p_init_cache, state,
                                              ds, config, split, workdir,
                                              config.num_test_steps)
      ds_metrics = {f'{k}_{split}': v for k, v in ds_metrics.items()}
      test_metrics.update(ds_metrics)
    writer.write_scalars(step, test_metrics)
