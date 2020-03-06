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
"""Library of common code for BERT preprocessing.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf
import tensorflow_text as tf_text

_CHINESE_SCRIPT_ID = 17
_CLS_TOKEN = "[CLS]"
_SEP_TOKEN = "[SEP]"
_PAD_TOKEN = "[PAD]"
_MASK_TOKEN = "[MASK]"


def collapse_dims(rt, axis=0):
  """Collapses the specified axis of a RaggedTensor.

  Suppose we have a RaggedTensor like this:
  [[1, 2, 3],
   [4, 5],
   [6]]

  If we flatten the 0th dimension, it becomes:
  [1, 2, 3, 4, 5, 6]

  Args:
    rt: a RaggedTensor.
    axis: the dimension to flatten.

  Returns:
    A flattened RaggedTensor, which now has one less dimension.
  """
  to_expand = rt.nested_row_lengths()[axis]
  to_elim = rt.nested_row_lengths()[axis + 1]

  bar = tf.RaggedTensor.from_row_lengths(to_elim, row_lengths=to_expand)
  new_row_lengths = tf.reduce_sum(bar, axis=1)
  return tf.RaggedTensor.from_nested_row_lengths(
      rt.flat_values,
      rt.nested_row_lengths()[:axis] + (new_row_lengths,))


def basic_tokenize(text_input, lower_case=False, keep_whitespace=False):
  """Performs basic word tokenization for BERT.

  Args:
    text_input: A Tensor of untokenized strings.
    lower_case: A bool indicating whether or not to perform lowercasing. Default
      is False.
    keep_whitespace: A bool indicating whether or not whitespace tokens should
      be kept in the output
  """
  # lowercase and strip accents (if option is set)
  if lower_case:
    text_input = tf_text.case_fold_utf8(text_input)

  # normalize by NFD
  text_input = tf_text.normalize_utf8(text_input, "NFD")

  # strip out control characters
  text_input = tf.strings.regex_replace(text_input, r"\p{Cc}|\p{Cf}|\p{Mn}", "")

  # For chinese and emoji characters, tokenize by unicode codepoints
  script_tokenized = tf_text.unicode_script_tokenize(
      text_input, keep_whitespace=keep_whitespace, name="UTF-8")
  token_script_ids = tf.strings.unicode_script(
      tf.strings.unicode_decode(script_tokenized.flat_values, "UTF-8"))

  is_chinese = tf.equal(token_script_ids, _CHINESE_SCRIPT_ID)[:, :1].values
  is_emoji = tf_text.wordshape(script_tokenized.flat_values,
                               tf_text.WordShape.HAS_EMOJI)
  is_punct = tf_text.wordshape(script_tokenized.flat_values,
                               tf_text.WordShape.IS_PUNCT_OR_SYMBOL)
  split_cond = is_chinese | is_emoji | is_punct
  unicode_char_split = tf.strings.unicode_split(script_tokenized, "UTF-8")

  unicode_split_tokens = tf.where(
      split_cond,
      y=tf.expand_dims(script_tokenized.flat_values, 1),
      x=unicode_char_split.values)

  # Pack back into a [batch, (num_tokens), (num_unicode_chars)] RT
  chinese_mix_tokenized = tf.RaggedTensor.from_row_lengths(
      values=unicode_split_tokens, row_lengths=script_tokenized.row_lengths())

  # Squeeze out to a [batch, (num_tokens)] RT
  return collapse_dims(chinese_mix_tokenized)


# TODO(kguu): verify that this matches public BERT's tokenizer.
# TODO(hterry): replace this with a Tokenizer class following the tf.text
#               Tokenizer RFC (https://github.com/tensorflow/community/pull/98)
def wordpiece_tokenize(text_input,
                       vocab_table,
                       token_out_type=tf.int64,
                       use_unknown_token=True,
                       lower_case=False,
                       keep_whitespace=False):
  """Perform wordpiece tokenization on a Tensor of untokenized strings.

  Args:
    text_input: <tf.string> A Tensor of untokenized strings.
    vocab_table: A LookupInterface containing the vocabulary.
    token_out_type: the dtype desired as the output, default is int64.
    use_unknown_token: (optional) A bool indicating whether to return [UNK]
      token or leave out of vocabulary words as is. Default is True.
    lower_case: (optional) A bool indicating whether or not to perform
      lowercasing. Default is False.
    keep_whitespace: A bool indicating whether or not whitespace tokens should
      be kept in the output

  Returns:
    a RaggedTensor w/ shape [num_batch, (num_tokens), (num_wordpieces)].
  """
  tokenized = basic_tokenize(text_input, lower_case, keep_whitespace)

  subwords, _, _ = tf_text.wordpiece_tokenize_with_offsets(
      tokenized,
      vocab_table,
      suffix_indicator="##",
      use_unknown_token=use_unknown_token,
      token_out_type=token_out_type,
  )
  return subwords


def construct_input_ids(inputs,
                        tokenize_fn,
                        cls_token=_CLS_TOKEN,
                        sep_token=_SEP_TOKEN,
                        pad_token=_PAD_TOKEN):
  """Perform the transformations to produce input_ids given untokenized strings.

  Given a Tensor of untokenized strings and a tokenization function, perform all
  the necessary transformation for producing the input_ids input used in BERT.
  This includes tokenization (token and wordpiece), and concat of special
  tokens.

  Args:
   inputs: [<tf.string>] a list of string Tensors each containing an untokenized
     string segment.
   tokenize_fn: a function that takes a tensor of strings and returns a
     RaggedTensor (w/ ragged_rank=2 of tokenized wordpieces).
   cls_token: string, the string used to represent the CLS token, default:
     '[CLS]'
   sep_token: string, the string used to represent the SEP token, default:
     '[SEP]'
   pad_token: string or int, the value used to pad tensors. Default: '[PAD]'

  Returns:
    a Tensor of input_ids w/ necessary padding.
  """
  assert isinstance(inputs, list) and inputs
  assert isinstance(cls_token, str)
  assert isinstance(sep_token, str)

  # TODO(hterry): add truncate budget/logic

  wordpieces = [tokenize_fn(segment) for segment in inputs]

  # Collapse the wordpiece dimension
  collapsed_wordpieces = [collapse_dims(rt) for rt in wordpieces]

  # Concat the [CLS], [SEP] tokens to get the input to BERT model_fn
  cls_tokens = tf.reshape(
      tf.tile([cls_token], [collapsed_wordpieces[0].nrows()]),
      [collapsed_wordpieces[0].nrows(), 1])
  sep_tokens = tf.reshape(
      tf.tile([sep_token], [collapsed_wordpieces[0].nrows()]),
      [collapsed_wordpieces[0].nrows(), 1])
  result = [cls_tokens]
  for rt in collapsed_wordpieces:
    result.append(rt)
    result.append(sep_tokens)
  return tf.concat(result, axis=1).to_tensor(pad_token)


def safe_zip(*iterables):
  """Like builtin `zip`, but checks that iterables have same length."""
  n = len(iterables[0])
  for iterable in iterables:
    if len(iterable) != n:
      raise ValueError("Zipped items must all have same length.")
  return zip(*iterables)


def whitespace_align_seqs(seqs):
  """Adds whitespace padding to sequence elements for vertical alignment.

  All sequence elements are cast to Unicode strings.

  Args:
    seqs (list): a list of sequences

  Returns:
    aligned_seqs (list): a list of aligned sequences
  """
  # Pad a string with spaces to a target width.
  pad = lambda s, width: u"{{:^{}}}".format(width).format(s)

  # Results will be stored here.
  aligned_seqs = [list() for _ in seqs]

  # Loop over each position.
  for vals in safe_zip(*seqs):
    text_vals = [unicode(v) for v in vals]
    max_width = max(len(v) for v in text_vals)
    for aligned_seq, text in safe_zip(aligned_seqs, text_vals):
      aligned_seq.append(pad(text, max_width))

  return aligned_seqs


def truncate_segment_tokens(segment_tokens, max_tokens):
  """Truncates segment tokens.

  Suppose we arrange our segment tokens in a matrix, e.g.:

  this  is    query  one    <pad>       <-- first segment
  this  is    query  two    <pad>       <-- second segment
  more  text  <pad>  <pad>  <pad>       <-- third segment

  If we visit the tokens in column-major order, we get this ordering:

  1     4     7     9
  2     5     8     10
  3     6

  This function removes all tokens with visit index > max_tokens.

  This function generalizes language.bert.run_classifier._truncate_seq_pair,
  and exactly matches its behavior when num_segments = 2 (a pair).

  Args:
    segment_tokens (RaggedTensor): a 2-D RaggedTensor of strings. One row for
      each segment. Each row is a list of tokens.
    max_tokens (Tensor): scalar int, max number of tokens for all segments
      combined.

  Returns:
    segment_tokens_truncated (RaggedTensor)
  """
  # mask is a 2-D int32 Tensor: 1's indicate tokens, 0's indicate padding.
  mask = tf.ones_like(segment_tokens, dtype=tf.int32).to_tensor()
  mask.set_shape([None, None])

  max_tokens = tf.cast(max_tokens, dtype=tf.int32)

  # visit_order reflects the column-major order we would visit each entry.
  transposed_mask = tf.transpose(mask)
  visit_order_flat = tf.cumsum(tf.reshape(transposed_mask, [-1]))
  visit_order = tf.transpose(
      tf.reshape(visit_order_flat, tf.shape(transposed_mask)))

  should_keep = (visit_order <= max_tokens) & tf.cast(mask, tf.bool)
  truncated_row_lengths = tf.reduce_sum(tf.cast(should_keep, tf.int64), axis=1)
  tokens_to_keep = tf.boolean_mask(segment_tokens.to_tensor(), should_keep)
  return tf.RaggedTensor.from_row_lengths(tokens_to_keep, truncated_row_lengths)


def add_special_tokens(segment_tokens, cls_token, sep_token):
  """Adds special tokens to segment tokens.

  Appends a [SEP] token to each segment.
  Prepends a [CLS] token to the first segment.

  Args:
    segment_tokens (RaggedTensor): a 2-D RaggedTensor of strings. One row for
      each segment. Each row is a list of tokens.
    cls_token (unicode): string for CLS token.
    sep_token (unicode): string for SEP token.

  Returns:
    segment_tokens (Tensor): a 2-D string Tensor.
  """
  num_rows = tf.to_int32(segment_tokens.nrows())

  # One SEP token for every row.
  sep_tokens = tf.fill([num_rows, 1], sep_token)

  # One CLS token in the first row.
  cls_tokens = tf.RaggedTensor.from_row_lengths([cls_token],
                                                row_lengths=tf.one_hot(
                                                    0, num_rows,
                                                    dtype=tf.int64))

  segment_tokens = tf.concat([cls_tokens, segment_tokens, sep_tokens], axis=1)
  return segment_tokens


def create_segment_ids(segment_tokens):
  """Creates segment IDs.

  Args:
    segment_tokens (RaggedTensor): 2-D RaggedTensor. One row for each segment.
      Each row is a list of tokens.

  Returns:
    segment_ids (RaggedTensor): 2-D RaggedTensor of int64's.
  """
  return tf.RaggedTensor.from_nested_row_splits(
      flat_values=segment_tokens.value_rowids(),
      nested_row_splits=segment_tokens.nested_row_splits)


def pad_to_length(tensor, target_length):
  """Pads a 1-D Tensor with zeros to the target length."""
  pad_amt = target_length - tf.size(tensor)
  # Assert that pad_amt is non-negative.
  assert_op = tf.Assert(pad_amt >= 0,
                        ["\nERROR: len(tensor) > target_length.", pad_amt])
  with tf.control_dependencies([assert_op]):
    padded = tf.pad(tensor, [[0, pad_amt]])
    padded.set_shape([target_length])
    return padded


def sample_mask_indices(tokens, mask_rate, mask_blacklist, max_num_to_mask):
  """Samples indices to mask.

  Args:
    tokens (Tensor): 1-D string Tensor.
    mask_rate (float): percentage of tokens to mask.
    mask_blacklist (Tensor): 1-D string Tensor of tokens to NEVER mask.
    max_num_to_mask (int): max # of masks.

  Returns:
    mask_indices (Tensor): 1-D int32 Tensor of indices to mask.
  """
  if mask_rate < 0 or mask_rate > 1:
    raise ValueError("mask_rate must be within [0, 1].")

  # Compute how many tokens to mask.
  num_tokens = tf.size(tokens)
  num_to_mask = tf.to_int32(tf.ceil(mask_rate * tf.to_float(num_tokens)))

  if mask_rate > 0:
    # If masking is enabled, then mask at least one, no matter what.
    # Original BERT code does this too.
    num_to_mask = tf.maximum(num_to_mask, 1)

  num_to_mask = tf.minimum(num_to_mask, max_num_to_mask)

  # If there are any [CLS] or [SEP], we count these as part of num_tokens.
  # Note that the original implementation of BERT does this as well.

  all_indices = tf.range(num_tokens)

  # Filter out indices containing CLS and SEP.
  allow_masking = tf.reduce_all(
      tf.not_equal(tokens, mask_blacklist[:, None]), axis=0)

  filtered_indices = tf.boolean_mask(all_indices, allow_masking)

  # Randomly select indices without replacement.
  shuffled_indices = tf.random.shuffle(filtered_indices)
  mask_indices = shuffled_indices[:num_to_mask]

  return mask_indices


def get_target_tokens_for_apply(token_ids, mask_indices):
  return tf.gather(token_ids, mask_indices)


def apply_masking(token_ids, target_token_ids, mask_indices, mask_token_id,
                  vocab_size):
  """Applies BERT masking.

  Args:
    token_ids (Tensor): 1-D Tensor of token IDs (ints)
    target_token_ids (Tensor): 1-D Tensor of token IDs (ints)
    mask_indices (Tensor): 1-D Tensor of indices (ints)
    mask_token_id (int): ID of [MASK] token.
    vocab_size (int): total size of vocabulary.

  Returns:
    token_ids_masked (Tensor): 1-D Tensor of token IDs, after target positions
      have been replaced with [MASK], a random token, or left alone.
    target_token_ids (Tensor): the original token IDs at the target positions.
  """
  num_to_mask = tf.size(mask_indices)

  mask_token_ids = tf.fill([num_to_mask], tf.cast(mask_token_id, tf.int64))
  random_token_ids = tf.random.uniform([num_to_mask],
                                       minval=0,
                                       maxval=vocab_size,
                                       dtype=tf.int64)

  # Uniform [0, 1) floats.
  randomness = tf.random.uniform([num_to_mask])

  # Replace target tokens with mask tokens.
  mask_values = tf.where(randomness < 0.8, mask_token_ids, target_token_ids)

  # Replace target tokens with random tokens.
  mask_values = tf.where(randomness > 0.9, random_token_ids, mask_values)

  # Mask out token_ids at mask_indices.
  token_ids_masked = tf.tensor_scatter_update(token_ids, mask_indices[:, None],
                                              mask_values)

  return token_ids_masked


BERT_INPUTS_LIST = [
    "input_ids", "input_mask", "segment_ids", "masked_lm_positions",
    "masked_lm_ids", "masked_lm_weights"
]


class BertInputs(collections.namedtuple("BertInputs", BERT_INPUTS_LIST)):
  """A namedtuple that groups together all BERT inputs."""

  def pretty_print(self, tokenizer, show_padding=True):
    """Pretty-prints BERT inputs for human inspection.

    NOTE: this method assumes that every attribute of BertInputs is a Numpy
    array, not a Tensor. Also, it assumes that the attributes are NOT batched:
    the first dimension of each array should not be a batch dimension.

    Args:
      tokenizer: an instance of language.bert.FullTokenizer.
      show_padding: boolean indicating whether to print pad tokens.

    Returns:
      a pretty-printed unicode string.
    """
    # Convert IDs back to tokens.
    tokens = tokenizer.convert_ids_to_tokens(self.input_ids)
    target_tokens = tokenizer.convert_ids_to_tokens(self.masked_lm_ids)

    # Annotate masked tokens with prediction targets.
    for target, pos, weight in safe_zip(target_tokens, self.masked_lm_positions,
                                        self.masked_lm_weights):
      if weight != 0:
        if weight != 1:
          raise ValueError("Weight must be either 0 or 1.")
        tokens[pos] += u"({})".format(target)

    # Sequences to display.
    seqs = [["     Tokens:"] + tokens,
            ["Segment IDs:"] + list(self.segment_ids),
            [" Input mask:"] + list(self.input_mask)]

    if not show_padding:
      max_seq_length = sum(self.input_mask)
      for seq in seqs:
        # +1 here, because we added an extra "Label" in front of each sequence.
        del seq[max_seq_length + 1:]

    # Format seqs into strings, so that tokens line up vertically.
    aligned_seqs = whitespace_align_seqs(seqs)

    return u"\n".join(u"  ".join(seq) for seq in aligned_seqs)


def bert_preprocess(text_segments,
                    vocab_table,
                    max_seq_length,
                    max_predictions_per_seq,
                    tokenize_fn=None,
                    mask_rate=0.15,
                    do_lower_case=True,
                    cls_token=_CLS_TOKEN,
                    sep_token=_SEP_TOKEN,
                    mask_token=_MASK_TOKEN):
  """Pre-processes a text tuple into BERT format.

  Args:
    text_segments (Tensor): 1-D Tensor of un-tokenized text segments (string).
      These will be concatenated into a single sequence.
    vocab_table (StaticHashTable): a map from string to integer.
    max_seq_length (int): max # of wordpiece tokens, including special tokens.
    max_predictions_per_seq (int): max # of masked positions for entire seq.
    tokenize_fn: a function that performs a tokenization on a 1D string Tensor
      of untokenized strings. By default this uses wordpiece tokenization.
    mask_rate (float): percentage of tokens to mask out. If 0, no masking is
      performed.
    do_lower_case (bool): whether to lowercase text or not. Default is True.
    cls_token (unicode): token representing CLS
    sep_token (unicode): token representing SEP (separator)
    mask_token (unicode): token representing MASK

  Returns:
    BertInputs
  """

  if not tokenize_fn:
    # pylint: disable=g-long-lambda
    tokenize_fn = lambda text_input: wordpiece_tokenize(
        text_input=text_segments,
        vocab_table=vocab_table,
        token_out_type=tf.string,
        use_unknown_token=True,
        lower_case=do_lower_case)

  segment_tokens_grouped = tokenize_fn(text_segments)
  # TODO(kguu): check that BERT handles UNK in the same way.

  # This is a RaggedTensor of shape [batch_size, (num_wordpieces)]
  # One row for each segment. Each row is a list of wordpiece tokens.
  segment_tokens = collapse_dims(segment_tokens_grouped)

  # Truncate.
  num_special_tokens = segment_tokens.nrows() + 1
  segment_tokens_truncated = truncate_segment_tokens(
      segment_tokens, max_seq_length - num_special_tokens)

  # Add special tokens.
  segment_tokens_with_special_tokens = add_special_tokens(
      segment_tokens_truncated, cls_token, sep_token)

  # Compute segment IDs.
  segment_ids_2d = create_segment_ids(segment_tokens_with_special_tokens)

  # Flatten everything to a 1-D string Tensor.
  tokens = segment_tokens_with_special_tokens.flat_values
  segment_ids = segment_ids_2d.flat_values

  # Convert to token IDs.
  token_ids = vocab_table.lookup(tokens)

  # Apply masking.
  mask_blacklist = tf.constant([cls_token, sep_token])
  mask_blacklist_ids = vocab_table.lookup(mask_blacklist)
  mask_token_id = vocab_table.lookup(tf.constant(mask_token))

  mask_indices = sample_mask_indices(token_ids, mask_rate, mask_blacklist_ids,
                                     max_predictions_per_seq)

  target_token_ids = get_target_tokens_for_apply(token_ids, mask_indices)
  token_ids_masked = apply_masking(token_ids, target_token_ids, mask_indices,
                                   mask_token_id, vocab_table.size())

  pad_mask = tf.ones_like(token_ids_masked)
  target_token_weights = tf.ones_like(target_token_ids, dtype=tf.float32)

  # Pad to max_seq_length.
  token_ids_masked = pad_to_length(token_ids_masked, max_seq_length)
  pad_mask = pad_to_length(pad_mask, max_seq_length)
  segment_ids = pad_to_length(segment_ids, max_seq_length)

  # Pad to max_predictions_per_seq.
  mask_indices = pad_to_length(mask_indices, max_predictions_per_seq)
  target_token_ids = pad_to_length(target_token_ids, max_predictions_per_seq)
  target_token_weights = pad_to_length(target_token_weights,
                                       max_predictions_per_seq)

  return BertInputs(
      input_ids=token_ids_masked,
      input_mask=pad_mask,
      segment_ids=segment_ids,
      masked_lm_positions=mask_indices,
      masked_lm_ids=target_token_ids,
      masked_lm_weights=target_token_weights)
