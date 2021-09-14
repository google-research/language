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
"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
# pylint: disable=g-complex-comprehension

from typing import Text, Union

from absl import logging
import tensorflow as tf

from muzero import learner_flags

from tensorflow.core.protobuf import trackable_object_graph_pb2  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations
from official.modeling import tf_utils
from official.nlp import keras_nlp
from official.nlp.bert import configs
from official.nlp.modeling import layers


def get_transformer_encoder(bert_config,
                            type_vocab_sizes,
                            num_float_features,
                            sequence_length,
                            bert_init_ckpt=None):
  """Gets a 'TransformerEncoder' object.

  Args:
    bert_config: A 'modeling.BertConfig' or 'modeling.AlbertConfig' object.
    type_vocab_sizes: Type vocab sizes.
    num_float_features: Number of float features.
    sequence_length: Maximum sequence length of the training data.
    bert_init_ckpt: str, Location of the initial checkpoint.

  Returns:
    A networks.BertEncoder object.
  """

  type_vocab_sizes = type_vocab_sizes or ()

  kwargs = dict(
      vocab_size=bert_config.vocab_size,
      hidden_size=bert_config.hidden_size,
      num_layers=bert_config.num_hidden_layers,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      sequence_length=sequence_length,
      max_sequence_length=bert_config.max_position_embeddings,
      type_vocab_sizes=type_vocab_sizes,
      num_float_features=num_float_features,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      bert_init_ckpt=bert_init_ckpt,
  )
  assert isinstance(bert_config, configs.BertConfig)
  return TransformerEncoder(**kwargs)


class TransformerEncoder(tf.keras.Model):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  Arguments:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    sequence_length: The sequence length that this encoder expects. If None, the
      sequence length is dynamic; if an integer, the encoder will require
      sequences padded to this length.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_sizes: The number of types that the 'type_ids' inputs can take.
    num_float_features: The number floating point features.
    intermediate_size: The intermediate size for the transformer layers.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    return_all_encoder_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers.
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               sequence_length=512,
               max_sequence_length=None,
               type_vocab_sizes=(16,),
               num_float_features=0,
               intermediate_size=3072,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               return_all_encoder_outputs=False,
               bert_init_ckpt=None,
               **kwargs):
    activation = tf.keras.activations.get(activation)
    initializer = tf.keras.initializers.get(initializer)

    if not max_sequence_length:
      max_sequence_length = sequence_length
    self._self_setattr_tracking = False
    num_type_features = len(type_vocab_sizes)
    self._config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'sequence_length': sequence_length,
        'max_sequence_length': max_sequence_length,
        'type_vocab_sizes': type_vocab_sizes,
        'num_type_features': num_type_features,
        'num_float_features': num_float_features,
        'intermediate_size': intermediate_size,
        'activation': tf.keras.activations.serialize(activation),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'initializer': tf.keras.initializers.serialize(initializer),
        'return_all_encoder_outputs': return_all_encoder_outputs,
    }

    word_ids = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(sequence_length,), dtype=tf.int32, name='input_mask')
    all_inputs = [word_ids, mask]
    if num_type_features:
      type_ids = tf.keras.layers.Input(
          shape=(sequence_length, num_type_features),
          dtype=tf.int32,
          name='input_type_ids')
      all_inputs.append(type_ids)
    if num_float_features:
      float_features = tf.keras.layers.Input(
          shape=(sequence_length, num_float_features),
          dtype=tf.float32,
          name='float_features')
      all_inputs.append(float_features)

    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=hidden_size,
        initializer=initializer,
        name='word_embeddings')
    word_embeddings = self._embedding_layer(word_ids)

    # Always uses dynamic slicing for simplicity.
    self._position_embedding_layer = keras_nlp.layers.PositionEmbedding(
        initializer=initializer, max_length=max_sequence_length)
    position_embeddings = self._position_embedding_layer(word_embeddings)
    all_embeddings = [word_embeddings, position_embeddings]

    if num_type_features:
      type_embeddings = [(layers.OnDeviceEmbedding(
          vocab_size=type_vocab_sizes[idx],
          embedding_width=hidden_size,
          initializer=initializer,
          use_one_hot=True,
          name='type_embeddings_{}'.format(idx))(type_ids[..., idx]))
                         for idx in range(num_type_features)]
      all_embeddings += type_embeddings

    if num_float_features:
      float_embeddings = [
          (
              tf.keras.layers.Dense(
                  hidden_size, name='float_features_{}'.format(idx))(
                      # Expanding the last dim here is important.
                      float_features[..., idx, None]))
          for idx in range(num_float_features)
      ]
      all_embeddings += float_embeddings

    embeddings = tf.keras.layers.Add()(all_embeddings)
    embeddings = (
        tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32)(embeddings))
    embeddings = (tf.keras.layers.Dropout(rate=dropout_rate)(embeddings))

    self._transformer_layers = []
    data = embeddings
    attention_mask = layers.SelfAttentionMask()([data, mask])
    encoder_outputs = []
    for i in range(num_layers):
      layer = layers.Transformer(
          num_attention_heads=num_attention_heads,
          intermediate_size=intermediate_size,
          intermediate_activation=activation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          kernel_initializer=initializer,
          name='model/layer_with_weights-%d' % (i + 4))
      self._transformer_layers.append(layer)
      data = layer([data, attention_mask])
      encoder_outputs.append(data)

    first_token_tensor = (
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
            encoder_outputs[-1]))
    cls_output = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')(
            first_token_tensor)

    if return_all_encoder_outputs:
      outputs = [encoder_outputs, cls_output]
    else:
      outputs = [encoder_outputs[-1], cls_output]
    super(TransformerEncoder, self).__init__(
        inputs=all_inputs, outputs=outputs, **kwargs)

    if bert_init_ckpt and learner_flags.INIT_CHECKPOINT.value is None:
      self.init_weights(bert_init_ckpt)

  def init_weights(self, ckpt_directory: Text):

    def assign_weights(model, checkpoint_reader,
                       variable_to_checkpoint_name_mapping):
      """Assign weights from checkpoint to model."""
      for variable_name, ckpt_variable_name in (
          variable_to_checkpoint_name_mapping.items()):
        variable = get_variable_by_name(model, variable_name)
        ckpt_weight = checkpoint_reader.get_tensor(ckpt_variable_name)
        if ckpt_weight.shape != variable.shape:
          logging.warning('Shape doesnt match for %s: %s vs %s', variable_name,
                          variable.shape, ckpt_weight.shape)
          continue
        variable.assign(ckpt_weight)

    checkpoint_reader = tf.train.load_checkpoint(ckpt_directory)
    ckpt_tensor_names = object_graph_key_mapping(ckpt_directory).values()
    tensor_names = [w.name for w in self.weights]
    mapping = {
        '/_': '/',
        'attention_layer': 'self_attention',
        'attention_norm': 'attention_layer_norm',
        'attention_output_dense': 'self_attention/attention_output',
        '_dense': '',
        '/.ATTRIBUTES/VARIABLE_VALUE': ':0'
    }
    var2ckpt = {}
    for ckpt_tensor_name in ckpt_tensor_names:
      var_name = ckpt_tensor_name
      for k, v in mapping.items():
        var_name = var_name.replace(k, v)
      if var_name in tensor_names:
        var2ckpt[var_name] = ckpt_tensor_name
    embeddings_mapping = {
        'word_embeddings/embeddings:0':
            'model/layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE',
        'position_embedding/embeddings:0':
            'model/layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE',
        'embeddings/layer_norm/gamma:0':
            'model/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE',
        'embeddings/layer_norm/beta:0':
            'model/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE',
    }
    for k, v in embeddings_mapping.items():
      var2ckpt[k] = v
    assert set(var2ckpt.keys()) - set(tensor_names) == set(
    ), 'A non-existent variable was added.'
    assert set(var2ckpt.values()) - set(ckpt_tensor_names) == set(
    ), 'Trying to load a non-existent weight.'
    missing_variables = set(tensor_names) - set(var2ckpt.keys())
    missing_ckpt_variables = set(ckpt_tensor_names) - set(var2ckpt.values())
    if missing_variables:
      logging.warning(
          'These variables cannot be found in the checkpoint directory: %s',
          '\n'.join(missing_variables))
    if missing_ckpt_variables:
      logging.warning(
          'These variables from the checkpoint cannot be found: %s',
          '\n'.join(missing_ckpt_variables))

    logging.info('Started loading weights from %s', ckpt_directory)
    assign_weights(self, checkpoint_reader, var2ckpt)
    logging.info('Finished loading weights from %s', ckpt_directory)

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_config(self):
    return self._config_dict

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def object_graph_key_mapping(checkpoint_path):
  """Return name to key mappings from the checkpoint.

  Args:
    checkpoint_path: string, path to object-based checkpoint

  Returns:
    Dictionary mapping tensor names to checkpoint keys.
  """
  reader = tf.train.load_checkpoint(checkpoint_path)
  object_graph_string = reader.get_tensor('_CHECKPOINTABLE_OBJECT_GRAPH')
  object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
  object_graph_proto.ParseFromString(object_graph_string)
  names_to_keys = {}
  for node in object_graph_proto.nodes:
    for attribute in node.attributes:
      names_to_keys[attribute.full_name] = attribute.checkpoint_key
  return names_to_keys


def get_variable_by_name(model: Union[tf.keras.Model, tf.keras.layers.Layer],
                         name: Text) -> tf.Variable:
  variables_with_name = [x for x in model.variables if x.name == name]
  if not variables_with_name:
    raise KeyError('Variable not found: %s' % (name,))

  if len(variables_with_name) != 1:
    raise RuntimeError('Found multiple variables with same name: %s' % (name,))

  return variables_with_name[0]
