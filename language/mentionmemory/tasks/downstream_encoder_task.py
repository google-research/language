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
"""Contains task with base methods for downstream eval for a mention encoder."""



from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import base_task
import ml_collections


class DownstreamEncoderTask(base_task.BaseTask):
  """Task with base methods for downstream evaluations for a mention encoder."""

  @classmethod
  def load_weights(cls, config):
    """Load model weights from file.

    We assume that `encoder_name` is specified in the config.
    We use corresponding class to load encoder weights.

    Args:
      config: experiment config.

    Returns:
      Dictionary of model weights.
    """
    encoder_name = config.model_config.encoder_name
    encoder_class = encoder_registry.get_registered_encoder(encoder_name)
    encoder_variables = encoder_class.load_weights(config)
    model_variables = {}
    for group_key in encoder_variables:
      model_variables[group_key] = {'encoder': encoder_variables[group_key]}

    return model_variables

  @classmethod
  def make_output_postprocess_fn(
      cls,
      config  # pylint: disable=unused-argument
  ):
    """Postprocess task samples (input and output). See BaseTask."""

    base_postprocess_fn = base_task.BaseTask.make_output_postprocess_fn(config)

    encoder_name = config.model_config.encoder_name
    encoder_class = encoder_registry.get_registered_encoder(encoder_name)
    encoder_postprocess_fn = encoder_class.make_output_postprocess_fn(config)

    def postprocess_fn(batch,
                       auxiliary_output):
      """Function that prepares model's input and output for serialization."""

      new_auxiliary_output = {}
      new_auxiliary_output.update(auxiliary_output)
      encoder_specific_features = encoder_postprocess_fn(
          batch, new_auxiliary_output)
      new_auxiliary_output.update(encoder_specific_features)
      return base_postprocess_fn(batch, new_auxiliary_output)

    return postprocess_fn

  @classmethod
  def get_auxiliary_output(cls, loss_helpers):
    """Extract features from `loss_helpers` to be used as `auxiliary_output`."""
    auxiliary_output = {}

    # As part of auxiliary output for the downstream tasks we save retrieved
    # memory information if it exists in the `loss_helper`.
    # This extra information would allow us to manually analyze models
    # performance on the task.

    def add_memory_auxiliary_output(prefix=''):
      keys = ('top_entity_ids', 'top_memory_ids', 'memory_attention_weights')
      for key in keys:
        full_key = prefix + key
        if full_key in loss_helpers:
          auxiliary_output[full_key] = loss_helpers[full_key]

    add_memory_auxiliary_output()
    add_memory_auxiliary_output('second_')

    return auxiliary_output
