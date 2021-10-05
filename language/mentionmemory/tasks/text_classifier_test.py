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
"""Tests for text classifier model."""
import json


from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import text_classifier
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np
import tensorflow as tf

# easiest to define as constant here
MENTION_SIZE = 2


class TextClassifierTest(test_utils.TestCase):
  """Tests for text classifier model."""

  encoder_config = {
      'dtype': 'bfloat16',
      'vocab_size': 1000,
      'entity_vocab_size': 1000,
      'max_positions': 512,
      'max_length': 128,
      'hidden_size': 64,
      'intermediate_dim': 128,
      'entity_dim': 32,
      'num_attention_heads': 8,
      'num_initial_layers': 4,
      'num_final_layers': 8,
      'dropout_rate': 0.1,
  }

  model_config = {
      'encoder_config': encoder_config,
      'vocab_size': 3,
      'encoder_name': 'eae',
      'dtype': 'bfloat16',
  }

  config = {
      'model_config': model_config,
      'seed': 0,
      'per_device_batch_size': 2,
      'samples_per_example': 1,
      'max_sample_mentions': 24,
      'max_mentions': 10,
  }

  def setUp(self):
    super().setUp()
    self.config = ml_collections.ConfigDict(self.config)

    self.model_config = self.config.model_config
    encoder_config = self.model_config.encoder_config

    self.max_length = encoder_config.max_length
    self.max_sample_mentions = self.config.max_sample_mentions
    self.collater_fn = text_classifier.TextClassifier.make_collater_fn(
        self.config)
    self.postprocess_fn = text_classifier.TextClassifier.make_output_postprocess_fn(
        self.config)

    model = text_classifier.TextClassifier.build_model(self.model_config)
    dummy_input = text_classifier.TextClassifier.dummy_input(self.config)
    init_rng = jax.random.PRNGKey(0)
    self.init_parameters = model.init(init_rng, dummy_input, True)

  def _gen_raw_batch(
      self,
      n_mentions,
  ):
    """Generate raw example."""

    bsz = self.config.per_device_batch_size

    text_ids = np.random.randint(
        low=1,
        high=self.model_config.encoder_config.vocab_size,
        size=(bsz, self.max_length),
        dtype=np.int64)

    text_mask = np.ones_like(text_ids)

    pad_size = max(0, self.max_sample_mentions - n_mentions)
    mention_pad_shape = (0, pad_size)
    mention_start_positions = np.random.choice(
        self.max_length // MENTION_SIZE, size=n_mentions,
        replace=False) * MENTION_SIZE
    mention_start_positions.sort()
    mention_end_positions = mention_start_positions + MENTION_SIZE - 1
    mention_mask = np.ones_like(mention_start_positions)

    mention_start_positions = np.pad(
        mention_start_positions[:self.max_sample_mentions],
        pad_width=mention_pad_shape,
        mode='constant')
    mention_end_positions = np.pad(
        mention_end_positions[:self.max_sample_mentions],
        pad_width=mention_pad_shape,
        mode='constant')
    mention_mask = np.pad(
        mention_mask[:self.max_sample_mentions],
        pad_width=mention_pad_shape,
        mode='constant')

    target = np.random.randint(self.model_config.vocab_size, size=bsz)

    raw_batch = {
        'text_ids': tf.constant(text_ids),
        'text_mask': tf.constant(text_mask),
        'target': tf.constant(target),
        'mention_start_positions': tf.constant(mention_start_positions),
        'mention_end_positions': tf.constant(mention_end_positions),
        'mention_mask': tf.constant(mention_mask),
    }

    for key in [
        'mention_start_positions', 'mention_end_positions', 'mention_mask'
    ]:
      raw_batch[key] = tf.tile(tf.reshape(raw_batch[key], (1, -1)), (bsz, 1))

    return raw_batch

  @parameterized.parameters([0, 1, 10, 24, 30])
  def test_loss_fn(self, n_mentions):
    """Test loss function runs and produces expected values."""
    raw_batch = self._gen_raw_batch(n_mentions)
    batch = self.collater_fn(raw_batch)
    batch = jax.tree_map(np.asarray, batch)

    loss_fn = text_classifier.TextClassifier.make_loss_fn(self.config)
    _, metrics, auxiliary_output = loss_fn(
        model_config=self.model_config,
        model_params=self.init_parameters['params'],
        model_vars={},
        batch=batch,
        deterministic=True)

    self.assertEqual(metrics['agg']['denominator'],
                     self.config.per_device_batch_size)
    features = self.postprocess_fn(batch, auxiliary_output)
    # Check features are JSON-serializable
    json.dumps(features)
    # Check features match the original batch
    for key in batch.keys():
      self.assertArrayEqual(np.array(features[key]), batch[key])


if __name__ == '__main__':
  absltest.main()
