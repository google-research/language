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
"""Utility funcs to operate on BERT's first layer non-contextual embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random

from bert import modeling

import numpy as np
import tensorflow.compat.v1 as tf


class BertFlexEmbeddingModel(object):
  """BERT model which allows arbitrary continuous inputs instead of embeddings."""

  def __init__(self, config, is_training, input_tensor, input_mask,
               token_type_ids):
    """Constructor for BertFlexEmbeddingModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_tensor: float32 Tensor of shape [batch_size, seq_length,
        hidden_size].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    with tf.variable_scope("bert", reuse=tf.compat.v1.AUTO_REUSE):
      with tf.variable_scope("embeddings"):
        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = modeling.embedding_postprocessor(
            input_tensor=input_tensor,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = modeling.create_attention_mask_from_input_mask(
            input_tensor, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = modeling.transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=modeling.create_initializer(
                config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    return self.sequence_output


def create_objective_fn(output_layer, num_labels, obj_type, prob_vector=None):
  """Build an objective function using final layer of flex BERT model."""
  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # self-entropy of the input data-point, maximizing uncertainty
    per_example_self_entropy = -tf.reduce_sum(
        probabilities * log_probs, axis=-1)
    total_self_entropy = tf.reduce_mean(per_example_self_entropy)

    # margin of confidence which needs to be minimized for maximum uncertainty
    top_2_probs, _ = tf.math.top_k(probabilities, k=2)
    per_example_margin = top_2_probs[:, 0] - top_2_probs[:, 1]
    total_margin = tf.reduce_mean(per_example_margin)

    top_2_log_probs, _ = tf.math.top_k(log_probs, k=2)
    per_example_log_margin = top_2_log_probs[:, 0] - top_2_log_probs[:, 1]
    total_log_margin = tf.reduce_mean(per_example_log_margin)

    if prob_vector is not None:
      # Broadcasting multiplication of probability vector along batches
      per_example_cross_entropy = -tf.reduce_sum(
          prob_vector * log_probs, axis=-1)
      total_cross_entropy = tf.reduce_mean(per_example_cross_entropy)

  if "self_entropy" in obj_type:
    return total_self_entropy, per_example_self_entropy, probabilities

  elif "cross_entropy" in obj_type:
    return total_cross_entropy, per_example_cross_entropy, probabilities

  elif "confidence_margin" in obj_type:
    return total_margin, per_example_margin, probabilities

  elif "confidence_log_margin" in obj_type:
    return total_log_margin, per_example_log_margin, probabilities


def model_fn(input_tensor, bert_input_mask, token_type_ids, bert_config,
             num_labels, obj_type, prob_vector):
  """Create a flex input model and decide the objective function."""
  flex_bert_model = BertFlexEmbeddingModel(
      config=bert_config,
      is_training=False,
      input_tensor=input_tensor,
      input_mask=bert_input_mask,
      token_type_ids=token_type_ids)

  output_layer = flex_bert_model.get_pooled_output()

  objective, per_eg_obj, probs = create_objective_fn(
      output_layer=output_layer,
      num_labels=num_labels,
      obj_type=obj_type,
      prob_vector=prob_vector)
  return objective, per_eg_obj, probs


def run_one_hot_embeddings(one_hot_input_ids, config):
  """Extract only the word embeddings of the original BERT model."""
  with tf.variable_scope("bert", reuse=tf.compat.v1.AUTO_REUSE):
    with tf.variable_scope("embeddings"):
      # branched from modeling.embedding_lookup
      embedding_table = tf.get_variable(
          name="word_embeddings",
          shape=[config.vocab_size, config.hidden_size],
          initializer=modeling.create_initializer(config.initializer_range))

      flat_input_ids = tf.reshape(one_hot_input_ids, [-1, config.vocab_size])
      output = tf.matmul(flat_input_ids, embedding_table)

      input_shape = modeling.get_shape_list(one_hot_input_ids)

      output = tf.reshape(output, input_shape[0:-1] + [config.hidden_size])

      return (output, embedding_table)


def run_bert_embeddings(input_ids, config):
  """Extract only the word embeddings of the original BERT model."""
  with tf.variable_scope("bert", reuse=tf.compat.v1.AUTO_REUSE):
    with tf.variable_scope("embeddings"):
      # Perform embedding lookup on the word ids.
      embedding_output, embedding_var = modeling.embedding_lookup(
          input_ids=input_ids,
          vocab_size=config.vocab_size,
          embedding_size=config.hidden_size,
          initializer_range=config.initializer_range,
          word_embedding_name="word_embeddings",
          use_one_hot_embeddings=False)
      return embedding_output, embedding_var


def get_nearest_neighbour(source, reference):
  """Get the nearest neighbour for every vector in source from reference."""
  normed_reference = tf.nn.l2_normalize(reference, axis=-1)
  normed_source = tf.nn.l2_normalize(source, axis=-1)

  cosine_sim = tf.matmul(normed_source, normed_reference, transpose_b=True)

  # Calculate the nearest neighbours and their cosine similarity
  nearest_neighbour = tf.argmax(cosine_sim, axis=-1)
  nearest_neighbour_sim = tf.reduce_max(cosine_sim, axis=-1)

  return nearest_neighbour, nearest_neighbour_sim


def parse_shard(shard, current_index, max_seq_length):
  """Parse a single shard in input template to get shard string and quantity."""

  if "<opt>" in shard and "</opt>" in shard:
    # indication that these vectors need to be optimized but starting from the
    # provided input words.

    # Currently this does not support continuous space optimization starting
    # from randomly initialized word embeddings.
    shard = shard.replace("<opt>", "").replace("</opt>", "")
    opt_from_embedding = True
  else:
    opt_from_embedding = False

  if "<freq>" not in shard:
    # if <freq> is absent in shard, assume the frequency is 1
    # This is done for more convenient inputs
    shard_token = shard
    shard_quantity = "1"
  else:
    shard_token, shard_quantity = shard.split("<freq>")
  # shard_quantity is the number of tokens for current shard
  if shard_quantity == "*":
    # * tokens are only used at the end of a template
    shard_quantity = max(0, max_seq_length - current_index)
  else:
    shard_quantity = int(shard_quantity)

  return shard_token, shard_quantity, opt_from_embedding


def update_bert_input_mask(shard_token, shard_quantity, tokenizer,
                           bert_input_mask):
  """Utility function to update the BERT input mask for an incoming shard."""
  if shard_token == "[EMPTY]":
    bert_input_mask.extend([1 for _ in range(shard_quantity)])

  elif shard_token in tokenizer.vocab and shard_token == "[PAD]":
    bert_input_mask.extend([0 for _ in range(shard_quantity)])

  elif shard_token in tokenizer.vocab and shard_token != "[PAD]":
    bert_input_mask.extend([1 for _ in range(shard_quantity)])

  else:
    shard_token_pieces = tokenizer.tokenize(shard_token)
    for _ in range(shard_quantity):
      bert_input_mask.extend([1 for _ in shard_token_pieces])


def update_token_type_ids(shard_token, shard_quantity, tokenizer,
                          token_type_ids, current_token_type):
  """Utility function to update token_type_ids list for an incoming shard."""
  if shard_token == "[EMPTY]":
    token_type_ids.extend([current_token_type for _ in range(shard_quantity)])

  elif shard_token in tokenizer.vocab and shard_token == "[SEP]":
    token_type_ids.extend([current_token_type for _ in range(shard_quantity)])
    current_token_type = 1 - current_token_type

  elif shard_token in tokenizer.vocab and shard_token != "[SEP]":
    token_type_ids.extend([current_token_type for _ in range(shard_quantity)])

  else:
    # BERT tokenizer will not retain [SEP] special tokens so current_token_type
    # will not swap during this shard.
    shard_token_pieces = tokenizer.tokenize(shard_token)
    for _ in range(shard_quantity):
      token_type_ids.extend([current_token_type for _ in shard_token_pieces])

  return current_token_type


def input_to_template(input_example, label_list):
  """Convenience function to convert input into likely template."""
  if input_example.text_b is None:
    template = ("[CLS]<freq>1 <piece> <opt>%s</opt> <piece> "
                "[SEP]<freq>1 <piece> "
                "[PAD]<freq>*" % (input_example.text_a))
  else:
    template = ("[CLS]<freq>1 <piece> <opt>%s</opt> <piece> "
                "[SEP]<freq>1 <piece> <opt>%s</opt> <piece> "
                "[SEP]<freq>1 <piece> "
                "[PAD]<freq>*" % (input_example.text_a, input_example.text_b))

  if isinstance(input_example.label, list):
    # run_classifier_distillation's input file processor, vector labels
    prob_vector = np.array(input_example.label)
  else:
    # run_classifier's input file processor, string labels
    prob_vector = np.zeros(len(label_list))
    prob_vector[label_list.index(input_example.label)] = 1.0

  return template, prob_vector


def template_to_ids(template, config, tokenizer, max_seq_length):
  """Converts template to a list of input ids and its corresponding mask."""
  template_shards = template.split(" <piece> ")
  input_ids = []
  # input mask will be applied during the discrete optimization, only unmasked
  # tokens will get optimized.
  input_mask = []
  # bert_input_mask will be applied during self-attention, it is mainly applied
  # to PADDING tokens.
  bert_input_mask = []
  # token_type_ids are needed for pairwise classification tasks like MNLI.
  token_type_ids = []
  current_index = 0
  current_token_type = 0

  for shard in template_shards:
    # Precautionary measure to prevent errors for templates > max_seq_length
    if current_index > max_seq_length:
      break

    shard_token, shard_quantity, opt_from_embedding = parse_shard(
        shard=shard, current_index=current_index, max_seq_length=max_seq_length)

    update_bert_input_mask(
        shard_token=shard_token,
        shard_quantity=shard_quantity,
        tokenizer=tokenizer,
        bert_input_mask=bert_input_mask)

    current_token_type = update_token_type_ids(
        shard_token=shard_token,
        shard_quantity=shard_quantity,
        tokenizer=tokenizer,
        token_type_ids=token_type_ids,
        current_token_type=current_token_type)

    if shard_token == "[EMPTY]":
      # Fill in each of the [EMPTY] slots with random words from vocabulary
      input_ids.extend([
          random.randint(0, config.vocab_size - 1)
          for _ in range(shard_quantity)
      ])
      input_mask.extend([1 for _ in range(shard_quantity)])
      current_index += shard_quantity
    else:
      if shard_token in tokenizer.vocab:
        token_ids = [tokenizer.vocab[shard_token]]
      else:
        token_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(shard_token))
      # repeat sequence of tokens according to number of shards
      for _ in range(shard_quantity):
        input_ids.extend(token_ids)
        if opt_from_embedding:
          # Since opt_from_embedding is true, the user wants the input to be
          # used as a starting point to optimize the words
          input_mask.extend([1 for _ in token_ids])
        else:
          # These tokens are fixed, so we use 0 in their positions in the mask
          input_mask.extend([0 for _ in token_ids])
        # Move the index forward by the number of tokens done
        current_index += len(token_ids)

  return (np.array(input_ids), np.array(input_mask), np.array(bert_input_mask),
          np.array(token_type_ids))


def template_to_input_tensor(template, flex_input, config, tokenizer,
                             max_seq_length):
  """This function converts a template to a corresponding input tensor."""
  template_shards = template.split(" <piece> ")
  embedding_input_ids = []
  index_array = []
  flex_input_mask = []
  # bert_input_mask will be applied during self-attention, it is mainly applied
  # to PADDING tokens.
  bert_input_mask = []
  # token_type_ids are needed for pairwise classification tasks like MNLI.
  token_type_ids = []
  current_index = 0
  current_token_type = 0

  batch_size = flex_input.shape[0]

  for shard in template_shards:
    # Precautionary measure to prevent errors for templates > max_seq_length
    if current_index >= max_seq_length:
      tf.logging.warning("Template is longer than max_seq_length = %d",
                         max_seq_length)
      break

    shard_token, shard_quantity, _ = parse_shard(
        shard=shard, current_index=current_index, max_seq_length=max_seq_length)

    update_bert_input_mask(
        shard_token=shard_token,
        shard_quantity=shard_quantity,
        tokenizer=tokenizer,
        bert_input_mask=bert_input_mask)

    current_token_type = update_token_type_ids(
        shard_token=shard_token,
        shard_quantity=shard_quantity,
        tokenizer=tokenizer,
        token_type_ids=token_type_ids,
        current_token_type=current_token_type)

    if shard_token == "[EMPTY]":
      flex_input_mask.extend([1 for _ in range(shard_quantity)])
      # [EMPTY] shard tokens indicate using the flex_input matrix
      index_array.extend(
          [x for x in range(current_index, current_index + shard_quantity)])
      # Move the index forward by the number of tokens done
      current_index += shard_quantity
    else:
      if shard_token in tokenizer.vocab:
        token_ids = [tokenizer.vocab[shard_token]]
      else:
        token_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(shard_token))
      # repeat sequence of tokens according to number of shards
      for _ in range(shard_quantity):
        # These tokens are fixed, so we use 0 in their positions in the mask
        flex_input_mask.extend([0 for _ in token_ids])
        index_array.extend([
            max_seq_length + len(embedding_input_ids) + token_num
            for token_num in range(len(token_ids))
        ])
        # Move the index forward by the number of tokens done
        current_index += len(token_ids)
        embedding_input_ids.extend(token_ids)

  if embedding_input_ids:
    # We store the embedding table to carry out nearest neighbour computes
    embeddings, embed_var = run_bert_embeddings(
        input_ids=tf.constant([embedding_input_ids]), config=config)
    batched_embeddings = tf.tile(embeddings, [batch_size, 1, 1])
    all_vectors = tf.concat([flex_input, batched_embeddings], axis=1)
  else:
    # Just a dummy run to get the final embedding table
    _, embed_var = run_bert_embeddings(
        input_ids=tf.constant([[0]]), config=config)
    all_vectors = flex_input

  # Build final masks by replication in the batch axis and converting to boolean
  flex_input_mask = tf.cast(
      tf.tile(tf.constant([flex_input_mask], dtype=tf.int32), [batch_size, 1]),
      tf.bool)

  bert_input_mask = tf.tile(
      tf.constant([bert_input_mask], dtype=tf.int32), [batch_size, 1])

  token_type_ids = tf.tile(
      tf.constant([token_type_ids], dtype=tf.int32), [batch_size, 1])

  final_input = tf.gather(all_vectors, tf.constant(index_array), axis=1)
  return (final_input, embed_var, flex_input_mask, bert_input_mask,
          token_type_ids)


def detokenize(input_ids, tokenizer):
  input_str = " ".join(tokenizer.convert_ids_to_tokens(input_ids))
  input_str = input_str.replace("[PAD]", "")
  return input_str.strip()


def ids_to_strings(input_ids, tokenizer):
  detokenized = detokenize(input_ids, tokenizer).replace("[CLS]", "")
  detokenized = detokenized.replace(" ##", "")
  return detokenized.split("[SEP]")
