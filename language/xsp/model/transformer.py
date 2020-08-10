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
"""Transformer-based model."""

from language.xsp.model import beam_search
from language.xsp.model import common_layers
from language.xsp.model import constants
from language.xsp.model import decode_utils
import tensorflow.compat.v1 as tf

# TODO(alanesuhr): Bias beam search based on sequence length.
ALPHA = 0.0

# when we add masks to logits to effectively zero out certain elements of the
# softmax, this is the scalar value of the "off" values of the mask. If this is
# too large in magnitude, it will cause numerical stability issues during
# training.
LOGIT_MASK_VALUE = -100


def _set_initializer():
  """Set initializer used for all model variables."""
  tf.get_variable_scope().set_initializer(
      tf.variance_scaling_initializer(scale=1.0, mode="fan_avg"))


def _build_transformer_decoder(encoder_output,
                               source_len,
                               decoder_input,
                               mode,
                               model_config,
                               single_step_index=None):
  """Generate decoder output given encoder output and decoder input.

  Args:
    encoder_output: Tensor of shape (B, L, D).
    source_len: Tensor of shape (B) used to mask encoder_output.
    decoder_input: Tensor of shape (B, L, D).
    mode: Enum indicating model mode.
    model_config: ModelConfig proto.
    single_step_index: Optional scalar integer Tensor specifying an index in
      decoder_input. If set, then decoder will only calculate output for a
      single step. This can only be used during inference and if using a
      single-layer decoder, but significantly speeds up auto-regresssive
      decoding. Index should be >= 0 and < decoder_input length.

  Returns:
    decoder_output: the decoder output Tensor of shape (B, L, D)

  Raises:
    ValueError: Invalid inputs for single step decoding.
  """
  _set_initializer()

  # Only apply dropout during training, not eval or inference.
  perform_dropout = (mode == tf.estimator.ModeKeys.TRAIN)
  if perform_dropout:
    decoder_input = tf.nn.dropout(
        decoder_input,
        keep_prob=1.0 - model_config.training_options.layer_dropout_rate)

  # Get batch shape information based on input and output Tensors.
  batch_size = tf.shape(decoder_input)[0]

  # Attention bias vectors ensure padding in input sequence is ignored.
  max_encoder_length = tf.shape(encoder_output)[1]
  encoder_decoder_attention_bias = common_layers.attention_bias_ignore_padding(
      source_len, max_length=max_encoder_length)

  # This attention bias matrix ensures decoder only has access to past
  # symbols.
  max_decoder_length = tf.shape(decoder_input)[1]
  decoder_self_attention_bias = common_layers.attention_bias_lower_triangle(
      max_decoder_length)

  # Setup decoder adjacency matrix.
  adjacency_matrix = common_layers.relative_positions_adjacency_matrix(
      max_decoder_length, batch_size,
      model_config.model_parameters.max_decoder_relative_distance)
  num_labels = 2 * model_config.model_parameters.max_decoder_relative_distance + 1

  x = decoder_input
  memory_antecedent = decoder_input

  using_single_step_optimization = False

  # If set, only decode a single step, otherwise decode all steps in parallel.
  if single_step_index is not None:
    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Single step decoding only supported during inference.")

    # Important optimization if using a single layer decoder.
    # Instead of running a masked decoder self-attention over all decoder
    # inputs, only run attention of current the step against all other decoder
    # inputs.
    # Need to ensure we are only using a single layer decoder.
    # Since single step decoding only computes a single decoder step,
    # higher layers would have incomplete information for self attention
    # with this optimization.
    if model_config.model_parameters.num_decoder_layers == 1:
      using_single_step_optimization = True
      x = tf.gather(decoder_input, [single_step_index], axis=1)
      adjacency_matrix = tf.gather(
          adjacency_matrix, [single_step_index], axis=1)
      decoder_self_attention_bias = tf.gather(decoder_self_attention_bias,
                                              [single_step_index])

  # Stack of decoder layers.
  with tf.variable_scope("decoder"):
    for layer in range(model_config.model_parameters.num_decoder_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_layers.graph_attention(
              x,
              memory_antecedent,
              decoder_self_attention_bias,
              model_config.model_parameters.decoder_dims,
              model_config.model_parameters.num_heads,
              adjacency_matrix=adjacency_matrix,
              num_labels=num_labels)
          x = common_layers.layer_postprocess(x, y, model_config,
                                              perform_dropout)
        with tf.variable_scope("encdec_attention"):
          y = common_layers.attention(
              x, encoder_output, encoder_decoder_attention_bias,
              model_config.model_parameters.decoder_dims,
              model_config.model_parameters.num_heads)
          x = common_layers.layer_postprocess(x, y, model_config,
                                              perform_dropout)
        with tf.variable_scope("ffn"):
          y = common_layers.ff_layer(
              x, model_config.model_parameters.decoder_ff_layer_hidden_size,
              model_config.model_parameters.decoder_dims)
          x = common_layers.layer_postprocess(x, y, model_config,
                                              perform_dropout)
          memory_antecedent = x
  if single_step_index is not None and not using_single_step_optimization:
    x = tf.gather(x, [single_step_index], axis=1)
  return x


def _get_target_embeddings(encoder_input, output_vocab_embeddings_table,
                           decode_steps, model_config):
  """Get target embeddings from either output table or copied from input.

  Args:
    encoder_input: Tensor representing encoder output of shape (batch size,
      input length, encoder dims).
    output_vocab_embeddings_table: Embeddings for output vocabulary of shape
      (output_vocab_size, target embedding dims).
    decode_steps: DecodeSteps tuple with tensors of shape (batch size, # steps).
    model_config: ModelConfig proto.

  Returns:
    Tensor of shape (batch_size, # steps, target embedding dims) representing
    unnormalized logits for both copy and generate actions.
  """
  input_length = tf.shape(encoder_input)[1]
  # Size of one_hot is (batch size, # steps, input length).
  one_hot = tf.one_hot(decode_steps.action_ids, input_length)
  # Size of encoder_dims is (batch size, input length, encoder dims).
  # Size of matrix multiplication is then (batch size, # steps, encoder_dims).
  copy_embeddings = tf.matmul(one_hot, encoder_input)

  # Need a linear transformation to ensure copy embeddings are right size.
  # Shape will then be (batch size, # steps, target_embedding_dims)
  copy_embeddings = common_layers.linear_transform(
      copy_embeddings, model_config.model_parameters.target_embedding_dims,
      "copy_embeddings_transform")
  # Simply get the generate embeddings from the output vocab table.

  generate_steps = tf.equal(
      decode_steps.action_types,
      tf.constant(constants.GENERATE_ACTION, dtype=tf.int64))
  generate_embeddings = common_layers.embedding_lookup(
      output_vocab_embeddings_table,
      decode_steps.action_ids * tf.to_int64(generate_steps))
  # For a given step, only use either copy OR generate embeddings.
  copy_steps = tf.equal(decode_steps.action_types,
                        tf.constant(constants.COPY_ACTION, dtype=tf.int64))

  copy_mask = tf.to_float(tf.expand_dims(copy_steps, axis=-1))
  generate_mask = tf.to_float(tf.expand_dims(generate_steps, axis=-1))
  target_embeddings = copy_embeddings * copy_mask + generate_embeddings * generate_mask

  return target_embeddings


def _get_action_logits(encoder_output,
                       decoder_output,
                       output_vocab_embeddings_table,
                       output_vocab_size,
                       model_config,
                       input_copy_mask=None,
                       clean_output_mask=None,
                       use_gating_mechanism=True):
  """Generate output logits given decoder output.

  This effectively combines a Pointer Network (Vinyals et al., 2015) with a
  standard softmax output layer for selecting symbols from an output vocabulary,
  similar to:
      - Jia and Liang, 2016 (https://arxiv.org/abs/1606.03622)
      - Gulcehre et al., 2016 (https://arxiv.org/abs/1603.08148)
      - Gu et al., 2016 (https://arxiv.org/abs/1603.06393)
      - See et al. 2017 (https://arxiv.org/abs/1704.04368)

  Args:
    encoder_output: Tensor representing encoder output of shape (batch size,
      input length, encoder dims).
    decoder_output: Tensor representing decoder output of shape (batch size, #
      decoded steps, decoder dims).
    output_vocab_embeddings_table: Embeddings for output vocabulary of shape
      (output_vocab_size, target embedding dims).
    output_vocab_size: Integer size of output_vocab_embeddings_table outer dim.
    model_config: ModelConfig proto.
    input_copy_mask: Mask of the input sequence for copying.
    clean_output_mask: Mask of the output vocab. For clean
      inference only.
    use_gating_mechanism: Whether to use gating mechanism.

  Returns:
    Tensor of shape (batch_size, output_vocab_size + input length) representing
    unnormalized logits for both copy and generate actions.
  """

  with tf.variable_scope("logits_transforms"):
    decoder_dims = decoder_output.get_shape()[-1]
    target_embedding_dims = model_config.model_parameters.target_embedding_dims

    # Dot product the decoder output with representations of each of the output
    # symbols to get a set of unnormalized logits for each output vocab item.
    # We need to tile the output vocab embeddings across the batch.
    output_vocab_transform = tf.expand_dims(output_vocab_embeddings_table, 0)
    batch_size = tf.shape(decoder_output)[0]
    output_vocab_transform = tf.tile(output_vocab_transform, [batch_size, 1, 1])
    # Transform representations to the target_embedding_dims.
    if decoder_dims != target_embedding_dims:
      transformed_decoder_output = common_layers.linear_transform(
          decoder_output, target_embedding_dims, "decoder_transform")
    else:
      transformed_decoder_output = decoder_output
    generate_logits = tf.matmul(
        transformed_decoder_output, output_vocab_transform, transpose_b=True)
    generate_logits_bias = tf.get_variable(
        "generate_logits_bias", shape=(output_vocab_size))
    generate_logits += generate_logits_bias

    # Dot product the decoder output with representations from the encoder
    # output.
    # This is necessary vs. re-using the encoder-decoder attention weights
    # because those use multihead attention.
    # First, need to transform representations to the decoder dimensions.
    transformed_encoder_output = common_layers.linear_transform(
        encoder_output, decoder_dims, "encoder_transform")

    copy_logits = tf.matmul(
        decoder_output, transformed_encoder_output, transpose_b=True)
    # This contains scores representing the probability of copying from input
    # (3rd dim) to output (2nd dim).

    # Optionally apply a soft gating mechanism to determine whether
    # to select from copy or generate logits.
    # TODO(petershaw): Evaluate and improve this gating mechanism.
    # The current implementation is most likely not optimal, since it applies
    # a scalar in the range [0,1] prior to softmax.
    if use_gating_mechanism:
      prob_gen_unnormalized = common_layers.linear_transform(
          decoder_output, 1, "prob_gen")
      prob_gen_bias = tf.get_variable("prob_gen_bias", shape=(1))
      prob_gen_unnormalized += prob_gen_bias
      prob_gen = tf.sigmoid(prob_gen_unnormalized)
      # Squeeze so that prob_gen has shape [batch_size, decode_length]
      prob_gen = tf.squeeze(prob_gen, axis=2)

      # These are the 'generate' logits so are scaled by P_gen.
      generate_logits *= tf.expand_dims(prob_gen, axis=-1)
      # These are the 'copy' logits so are scaled by 1 - P_gen.
      copy_logits *= tf.expand_dims(1 - prob_gen, axis=-1)

    if clean_output_mask is not None:
      clean_mask = (1 - tf.dtypes.cast(
          clean_output_mask, dtype=tf.dtypes.float32)) * LOGIT_MASK_VALUE

      batch_size = common_layers.get_shape_list(generate_logits)[0]
      output_vocab_size = common_layers.get_shape_list(generate_logits)[-1]

      clean_mask = tf.reshape(
          tf.tile(clean_mask, [batch_size]), [batch_size, output_vocab_size])
      generate_logits += tf.expand_dims(clean_mask, axis=1)

    if input_copy_mask is not None:
      copy_mask = (1 - tf.dtypes.cast(
          input_copy_mask, dtype=tf.dtypes.float32)) * LOGIT_MASK_VALUE
      copy_logits += tf.expand_dims(copy_mask, axis=1)

    # Concatenate logits into a single vector; first N (fixed) inputs are the
    # generation probabilities, and next are the copy probabilities for each
    # input (well, they aren't really probabilities, but scores.)
    extended_logits = tf.concat([generate_logits, copy_logits], axis=2)
    return extended_logits


def _transformer_body(input_embeddings,
                      source_len,
                      target_decode_steps,
                      mode,
                      model_config,
                      output_vocab_size,
                      output_vocab_embeddings_table,
                      input_copy_mask=None):
  """Build a Transformer.

  Args:
    input_embeddings: The embeddings of the input to the Transformer.
    source_len: The length of the input utterance.
    target_decode_steps: Number of steps to generate.
    mode: Mode of the model.
    model_config: Model configuration.
    output_vocab_size: Size of the output vocabulary.
    output_vocab_embeddings_table: Table containing embeddings of output tokens.
    input_copy_mask: Mask on the input utterance for which tokens can be copied.

  Returns:
    Outputs of the transformer.
  """
  # Just apply a simple linear layer here
  encoder_output = common_layers.linear_transform(
      input_embeddings,
      output_size=model_config.model_parameters.encoder_dims,
      scope="bert_to_transformer")

  target_embeddings = _get_target_embeddings(input_embeddings,
                                             output_vocab_embeddings_table,
                                             target_decode_steps, model_config)
  decoder_output = _build_transformer_decoder(encoder_output, source_len,
                                              target_embeddings, mode,
                                              model_config)
  logits = _get_action_logits(
      encoder_output,
      decoder_output,
      output_vocab_embeddings_table,
      output_vocab_size,
      model_config,
      input_copy_mask=input_copy_mask)
  return logits


def _beam_decode(input_embeddings,
                 alpha,
                 output_vocab_size,
                 target_end_id,
                 target_start_id,
                 output_vocab_embeddings_table,
                 source_len,
                 model_config,
                 mode,
                 beam_size,
                 input_copy_mask=None,
                 clean_output_mask=None):
  """Beam search decoding."""
  # Assume batch size is 1.
  batch_size = 1
  encoder_output = common_layers.linear_transform(
      input_embeddings,
      output_size=model_config.model_parameters.encoder_dims,
      scope="bert_to_transformer")

  decode_length = model_config.data_options.max_decode_length

  # Expand decoder inputs to the beam width.
  input_embeddings = tf.tile(input_embeddings, [beam_size, 1, 1])
  encoder_output = tf.tile(encoder_output, [beam_size, 1, 1])

  def symbols_to_logits_fn(current_index, logit_indices):
    """Go from targets to logits.

    Args:
      current_index: Integer corresponding to 0-indexed decoder step.
      logit_indices: Tensor of shape [batch_size * beam_width, decode_length +
        1] to input to decoder.

    Returns:
      Tensor of shape [batch_size * beam_width, output_vocab_size] representing
      logits for the current decoder step.

    Raises:
      ValueError if inputs do not have static length.
    """
    decode_steps = decode_utils.get_decode_steps(logit_indices,
                                                 output_vocab_size,
                                                 model_config)
    target_embeddings = _get_target_embeddings(input_embeddings,
                                               output_vocab_embeddings_table,
                                               decode_steps, model_config)
    decoder_output = _build_transformer_decoder(
        encoder_output,
        source_len,
        target_embeddings,
        mode,
        model_config,
        single_step_index=current_index)
    logits = _get_action_logits(
        encoder_output,
        decoder_output,
        output_vocab_embeddings_table,
        output_vocab_size,
        model_config,
        input_copy_mask=input_copy_mask,
        clean_output_mask=clean_output_mask)
    # Squeeze length dimension, as it should be 1.
    logits = tf.squeeze(logits, axis=[1])
    # Shape of logits should now be:
    # [batch_size * beam_width, output_vocab_size].
    return logits

  initial_ids = tf.ones([batch_size], dtype=tf.int32) * target_start_id
  # ids has shape: [batch_size, beam_size, decode_length]
  # scores has shape: [batch_size, beam_size]
  decode_length = model_config.data_options.max_decode_length
  source_length = input_embeddings.get_shape()[1]

  if source_length.value is None:
    # Fall back on using dynamic shape information.
    source_length = tf.shape(input_embeddings)[1]
  extended_vocab_size = output_vocab_size + source_length
  ids, scores = beam_search.beam_search(symbols_to_logits_fn, initial_ids,
                                        beam_size, decode_length,
                                        extended_vocab_size, alpha,
                                        target_end_id, batch_size)
  # Remove start symbol from returned predicted IDs.
  predicted_ids = ids[:, :, 1:]
  # Since batch size is expected to be 1, squeeze the batch dimension.
  predicted_ids = tf.squeeze(predicted_ids, axis=[0])
  scores = tf.squeeze(scores, axis=[0])
  # This is the output dict that the function returns.
  output_decode_steps = decode_utils.get_decode_steps(predicted_ids,
                                                      output_vocab_size,
                                                      model_config)
  predictions = decode_utils.get_predictions(output_decode_steps)
  predictions[constants.SCORES_KEY] = scores
  return predictions


def _greedy_decode(input_embeddings,
                   output_vocab_size,
                   target_end_id,
                   target_start_id,
                   output_vocab_embeddings_table,
                   source_len,
                   model_config,
                   mode,
                   input_copy_mask=None,
                   clean_output_mask=None):
  """Fast decoding."""
  encoder_output = common_layers.linear_transform(
      input_embeddings,
      output_size=model_config.model_parameters.encoder_dims,
      scope="bert_to_transformer")

  decode_length = model_config.data_options.max_decode_length

  # Expand the inputs in to the beam width.
  def symbols_to_logits_fn(logit_indices, current_index):
    """Go from targets to logits."""
    logit_indices = tf.expand_dims(logit_indices, 0)
    decode_steps = decode_utils.get_decode_steps(logit_indices,
                                                 output_vocab_size,
                                                 model_config)
    target_embeddings = _get_target_embeddings(input_embeddings,
                                               output_vocab_embeddings_table,
                                               decode_steps, model_config)
    decoder_output = _build_transformer_decoder(
        encoder_output,
        source_len,
        target_embeddings,
        mode,
        model_config,
        single_step_index=current_index)

    logits = _get_action_logits(
        encoder_output,
        decoder_output,
        output_vocab_embeddings_table,
        output_vocab_size,
        model_config,
        input_copy_mask=input_copy_mask,
        clean_output_mask=clean_output_mask)

    # Squeeze batch dimension and length dimension, as both should be 1.
    logits = tf.squeeze(logits, axis=[0, 1])
    # Shape of logits should now be (output_vocab_size).
    return logits

  def loop_cond(i, decoded_ids, unused_logprobs):
    """Loop conditional that returns false to stop loop."""
    return tf.logical_and(
        tf.reduce_all(tf.not_equal(decoded_ids, target_end_id)),
        tf.less(i, decode_length))

  def inner_loop(i, decoded_ids, logprobs):
    """Decoder function invoked on each while loop iteration."""
    logits = symbols_to_logits_fn(decoded_ids, i)
    next_id = tf.argmax(logits, axis=0)
    softmax = tf.nn.softmax(logits)
    extended_vocab_size = tf.shape(softmax)[-1]
    mask = tf.one_hot(next_id, extended_vocab_size)
    prob = tf.reduce_sum(softmax * mask)
    logprob = tf.log(prob)

    # Add one-hot values to output Tensors, since values at index > i+1 should
    # still be zero.
    logprobs += tf.one_hot(
        i + 1, decode_length + 1, on_value=logprob, dtype=tf.float32)
    decoded_ids += tf.one_hot(
        i + 1, decode_length + 1, on_value=next_id, dtype=tf.int64)

    return i + 1, decoded_ids, logprobs

  initial_ids = tf.zeros(dtype=tf.int64, shape=[decode_length + 1])
  initial_ids += tf.one_hot(
      0, decode_length + 1, on_value=tf.cast(target_start_id, tf.int64))
  initial_logprob = tf.zeros(dtype=tf.float32, shape=[decode_length + 1])
  initial_i = tf.constant(0)

  initial_values = [initial_i, initial_ids, initial_logprob]

  _, decoded_ids, logprobs = tf.while_loop(loop_cond, inner_loop,
                                           initial_values)

  # Remove <START> symbol.
  decoded_ids = decoded_ids[1:]
  logprobs = logprobs[1:]
  # Sum logprobs to get scores for overall sequence.
  logprobs = tf.reduce_sum(logprobs, axis=0)

  # Expand decoded_ids and logprobs to reflect beam width dimension of 1.
  decoded_ids = tf.expand_dims(decoded_ids, 0)
  logprobs = tf.expand_dims(logprobs, 0)

  # This is the output dict that the function returns.
  output_decode_steps = decode_utils.get_decode_steps(decoded_ids,
                                                      output_vocab_size,
                                                      model_config)
  predictions = decode_utils.get_predictions(output_decode_steps)
  predictions[constants.SCORES_KEY] = logprobs

  return predictions


def train(model_config,
          input_embeddings,
          source_len,
          output_vocab_size,
          output_vocab_embeddings_table,
          target_decode_steps,
          mode,
          input_copy_mask=None):
  """Constructs encoder and decoder transformation for training and eval.

  In the shapes described below, B is batch size, L is sequence length,
  D is the dimensionality of the model embeddings, and T is the output vocab
  size.

  Args:
    model_config: ModelConfig proto.
    input_embeddings: Tensor of shape (B, L, D) representing inputs.
    source_len: Tensor of shape (B) containing length of each input sequence.
    output_vocab_size: Size of output vocabulary.
    output_vocab_embeddings_table: Tensor of shape (T, D) representing table of
      embeddings for output symbols.
    target_decode_steps: DecodeSteps representing target outputs. Each tensor
      has shape (B, L).
    mode: Enum indicating model mode, TRAIN or EVAL.
    input_copy_mask: Mask for copying actions.

  Returns:
    Tuple of (logits, predicted_ids), where logits is a tensor of shape
    (B, L, T) representing model output logits, and predicted_ids is
    a tensor of shape (B, L) containing the derived integer IDs of the
    one-best output symbol.
  """
  logits = _transformer_body(
      input_embeddings,
      source_len,
      target_decode_steps,
      mode,
      model_config,
      output_vocab_size,
      output_vocab_embeddings_table,
      input_copy_mask=input_copy_mask)

  predicted_ids = tf.to_int32(tf.argmax(logits, axis=-1))
  output_decode_steps = decode_utils.get_decode_steps(predicted_ids,
                                                      output_vocab_size,
                                                      model_config)
  predictions = decode_utils.get_predictions(output_decode_steps)
  return logits, predictions


def infer(model_config,
          input_embeddings,
          source_len,
          output_vocab_size,
          output_vocab_embeddings_table,
          mode,
          input_copy_mask=None,
          clean_output_mask=None,
          beam_size=1):
  """Constructs encoder and decoder transformation for training and eval.

  In the shapes described below, B is batch size, L is sequence length,
  D is the dimensionality of the model embeddings, T is the target vocab
  size, W is the beam width, and X is the max decode length.

  Args:
    model_config: ModelConfig proto.
    input_embeddings: Tensor of shape (B, L, D) representing inputs.
    source_len: Tensor of shape (B) containing length of each input sequence.
    output_vocab_size: Size of output vocabulary.
    output_vocab_embeddings_table: Tensor of shape (T, D) representing table of
      embeddings for output symbols.
    mode: Enum indicating model mode, which should be INFER.
    input_copy_mask: Mask for copying actions.
    clean_output_mask: MAsk for preventing generating new values.
    beam_size: Size of beam for inference.

  Returns:
    A dictionary of tensors. The dictionary will contain
    'predicted_ids' tensor of shape (W, X) and 'scores' tensor of
    shape (W).

  Raises:
    ValueError: Invalid model configuration.
  """
  # Currently, we don't allow batched inputs for inference. Support for this
  # could be added in the future, but this assumption simplifies some code.
  if input_embeddings.get_shape()[0] != 1:
    raise ValueError("Batch size must be 1 for inference.")
  if beam_size > 1:
    return _beam_decode(
        input_embeddings,
        ALPHA,
        output_vocab_size,
        constants.TARGET_END_SYMBOL_ID,
        constants.TARGET_START_SYMBOL_ID,
        output_vocab_embeddings_table,
        source_len,
        model_config,
        mode,
        beam_size,
        input_copy_mask,
        clean_output_mask=clean_output_mask)
  else:
    return _greedy_decode(
        input_embeddings,
        output_vocab_size,
        constants.TARGET_END_SYMBOL_ID,
        constants.TARGET_START_SYMBOL_ID,
        output_vocab_embeddings_table,
        source_len,
        model_config,
        mode,
        input_copy_mask,
        clean_output_mask=clean_output_mask)
