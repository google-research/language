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
"""Input pipeline for dataset."""

from clu import deterministic_data
import jax

from language.gscan.xattn_model.dataset import gscan_dataset

import numpy as np
import tensorflow as tf


def create_val_dataset(config, split, batch_size, pad_last_batch):
  """Create validataion dataset.

  Args:
    config: ml_collections.ConfigDict to use.
    split: The validation split.
    batch_size: The batch size.
    pad_last_batch: Bool to indicate whether to pad last patch or not.

  Returns:
    The validation dataset.
  """
  dataset_builder = gscan_dataset.GSCANDataset(**config)
  num_batches = None
  cardinality = None
  if pad_last_batch:
    num_examples = dataset_builder.num_examples[split]
    val_batch_size = jax.local_device_count() * batch_size
    num_batches = int(
        np.ceil(num_examples / val_batch_size / jax.process_count()))
    cardinality = int(np.ceil(num_examples / jax.process_count()))
  ds = deterministic_data.create_dataset(
      dataset_builder,
      split=split,
      preprocess_fn=dataset_builder.preprocess,
      cache=jax.process_count() > 1,
      batch_dims=[jax.local_device_count(), batch_size],
      num_epochs=1,
      shuffle=False,
      pad_up_to_batches=num_batches,
      cardinality=cardinality)
  return ds


def create_datasets(config, data_rng):
  """Create datasets for training and evaluation.

  Args:
    config: ml_collections.ConfigDict to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    A tuple of the training dataset and the evaluation dataset.
  """
  dataset_builder = gscan_dataset.GSCANDataset(**config)
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=config.train_split,
      rng=data_rng,
      preprocess_fn=dataset_builder.preprocess,
      cache=False,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), config.train_per_device_batch_size],
      num_epochs=config.num_epochs,
      shuffle=True,
      prefetch_size=tf.data.experimental.AUTOTUNE,
  )

  eval_ds = create_val_dataset(config, config.eval_split,
                               config.eval_per_device_batch_size,
                               config.eval_pad_last_batch)
  return train_ds, eval_ds
