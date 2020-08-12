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
"""Utilities for featurizing examples."""
import collections
import functools
import hashlib

from bert import tokenization
from language.common.utils import nest_utils
from language.realm import profile
import tensorflow.compat.v1 as tf

BertInputs = collections.namedtuple('BertInputs',
                                    ['token_ids', 'mask', 'segment_ids'])
Document = collections.namedtuple('Document',
                                  ['uid', 'title_token_ids', 'body_token_ids'])
"""Represents a document that can be retrieved.

Attributes:
  uid (int): a uniquely identifying integer.
  title_token_ids (np.array): int32 Numpy array of title token IDs.
  body_token_ids (np.array): int32 Numpy array of body token IDs.
"""


def get_document_uid(title, body):
  """Generates a nearly unique ID for a document.

  Uses the SHA1 hash. Deterministically produces the same result for a given
  input, and almost never collides.

  Args:
    title (bytes): document title
    body (bytes): document body text

  Returns:
    uid (int)
  """
  b = '{} <sep> {}'.format(title.decode('utf-8'), body.decode('utf-8'))
  return int(hashlib.sha1(b.encode('utf-8')).hexdigest(), 16)


QUERY_ATTRS = ['text', 'tokens', 'mask_spans', 'orig_doc_uid']


class Query(collections.namedtuple('Query', QUERY_ATTRS)):
  """Represents a REALM pre-training query to retrieve documents for.

  Attributes:
    text (bytes): the complete text of the query, with nothing masked out.
    tokens (list[Token]): tokens corresponding to the text.
    mask_spans (list[int, int]): a list of (start, stop) byte offsets,
      indicating spans to mask out. `start` is inclusive, `stop' is exclusive.
    orig_doc_uid (int): the UID for the document where this query came from.
  """

  def __repr__(self):
    # Underline the masked part of the query.
    mask_chars = [' '] * len(self.text)
    for start, stop in self.mask_spans:
      for i in range(start, stop):
        mask_chars[i] = '-'
    mask_str = ''.join(mask_chars)

    return '\n'.join([self.text.decode(), mask_str])
    # NOTE: mask_str may not line up with self.text after self.text is
    # converted to UTF-8.


class Featurizer(object):
  """Featurizes queries and documents."""

  def __init__(self, query_seq_len, candidate_seq_len, num_candidates,
               max_masks, tokenizer):
    self.query_seq_len = query_seq_len
    self.candidate_seq_len = candidate_seq_len
    self.num_candidates = num_candidates
    self.max_masks = max_masks
    self.tokenizer = tokenizer

  @profile.profiled_function
  def mask_query(self, query):
    """Applies masking to a query.

    Args:
      query: a Query object.

    Returns:
      A dict with the attributes described below.

    Raises:
      MaskingError: if the specified mask spans cannot be properly applied.

    The returned dict contains the following:
      token_ids_after_masking (list[int]): list of token IDs, including [MASK]
        tokens.
      masked_token_ids (list[int]): the original token ID of the tokens that
        have been masked out.
      mask_indices (list[int]): the token offset for all the mask tokens present
        in token_ids_after_masking.
    """
    # Tokenize.
    token_ids = [token.id for token in query.tokens]

    # Convert mask offsets to token offsets.
    mask_indices = []
    for mask_start, mask_stop in query.mask_spans:
      for token_idx, token in enumerate(query.tokens):
        # Only tokens that are strictly within a mask_span are masked.
        if token.start >= mask_start and token.stop <= mask_stop:
          mask_indices.append(token_idx)

    if query.mask_spans and not mask_indices:
      raise MaskingError('Masks do not cover any tokens.')

    # Deduplicate and sort.
    mask_indices = sorted(set(mask_indices))

    # Make sure that masked region does not get truncated in either the query
    # query representation or the joint representation.
    if mask_indices:
      if mask_indices[-1] >= self.query_seq_len - 2:
        raise MaskingError('Masks would be truncated.')

    # Mask out the specified indices.
    token_ids_after_masking = list(token_ids)  # Make a copy, then modify it.
    masked_token_ids = []
    for mask_idx in mask_indices:
      token_ids_after_masking[mask_idx] = self.tokenizer.mask_id
      masked_token_ids.append(token_ids[mask_idx])

    return {
        'token_ids_after_masking': token_ids_after_masking,
        'masked_token_ids': masked_token_ids,
        'mask_indices': mask_indices,
    }

  def _reformat_bert_inputs(self, bert_inputs, fmt):
    """Reformats BERT input features."""
    if fmt == 'bert_inputs':
      return bert_inputs  # No reformatting necessary.
    elif fmt == 'dict':
      # This is the format expected by the original BERT code.
      return {
          'input_ids': bert_inputs.token_ids,
          'input_mask': bert_inputs.mask,
          'segment_ids': bert_inputs.segment_ids,
      }
    else:
      raise ValueError('Invalid format: {}'.format(fmt))

  @profile.profiled_function
  def featurize_query(self, query, fmt='dict'):
    """Featurizes a Query.

    Args:
      query: a Query instance.
      fmt: can be either 'dict' or 'bert_inputs'. Default is 'dict'.

    Returns:
      If fmt is 'dict', a dict with the following structure:
        input_ids: [query_seq_len] int32 Tensor
        input_mask: [query_seq_len] int32 Tensor
        segment_ids: [query_seq_len] int32 Tensor
      If fmt is 'bert_inputs', a BertInputs object.
    """
    query_mask_features = self.mask_query(query)

    bert_inputs = bert_format(
        input_seqs=[query_mask_features['token_ids_after_masking']],
        max_seq_len=self.query_seq_len,
        cls_id=self.tokenizer.cls_id,
        sep_id=self.tokenizer.sep_id)

    return self._reformat_bert_inputs(bert_inputs, fmt)

  @profile.profiled_function
  def featurize_document(self, doc, fmt='dict'):
    """Featurizes a Document.

    Args:
      doc: a Document instance.
      fmt: can be either 'dict' or 'bert_inputs'. Default is 'dict'.

    Returns:
      If fmt is 'dict', a dict with the following structure:
        input_ids: [candidate_seq_len] int32 Tensor
        input_mask: [candidate_seq_len] int32 Tensor
        segment_ids: [candidate_seq_len] int32 Tensor
      If fmt is 'bert_inputs', a BertInputs object.
    """
    bert_inputs = bert_format(
        input_seqs=[doc.title_token_ids, doc.body_token_ids],
        max_seq_len=self.candidate_seq_len,
        cls_id=self.tokenizer.cls_id,
        sep_id=self.tokenizer.sep_id)

    return self._reformat_bert_inputs(bert_inputs, fmt)

  def featurize_document_tf(self, title_token_ids, body_token_ids):
    """Featurizes a document using only TF ops (for non-eager execution).

    Args:
      title_token_ids: a 1-D int Tensor.
      body_token_ids: a 1-D int Tensor.

    Returns:
      A dict with the following structure:
        input_ids: [candidate_seq_len] int32 Tensor
        input_mask: [candidate_seq_len] int32 Tensor
        segment_ids: [candidate_seq_len] int32 Tensor
    """
    # Total number of tokens representing the doc, including special tokens.
    sequence_len = self.candidate_seq_len

    # Possibly truncate title, reserving at least 3 spots for special tokens.
    title_token_ids = title_token_ids[:sequence_len - 3]
    title_len = tf.size(title_token_ids)

    # Calculate how many spots left for body text.
    max_body_len = tf.maximum(sequence_len - 3 - title_len, 0)

    # Possibly truncate the body text.
    body_token_ids = body_token_ids[:max_body_len]
    body_len = tf.size(body_token_ids)

    # Zero-padding to occupy whatever space remains.
    pair_len = title_len + body_len
    padding = tf.zeros([sequence_len - 3 - pair_len], tf.int32)

    # Put title, body and padding together.
    input_ids = tf.concat([
        [self.tokenizer.cls_id],
        title_token_ids,
        [self.tokenizer.sep_id],
        body_token_ids,
        [self.tokenizer.sep_id],
        padding], 0)

    # input_mask indicates non-pad tokens.
    input_mask = tf.concat([
        tf.ones([pair_len + 3], tf.int32),
        padding], 0)

    # segment_ids distinguish title from body.
    segment_ids = tf.concat([
        tf.zeros([title_len + 2], tf.int32),
        tf.ones([body_len + 1], tf.int32),
        padding], 0)

    # Add static shape annotations.
    input_ids.set_shape([sequence_len])
    input_mask.set_shape([sequence_len])
    segment_ids.set_shape([sequence_len])

    return {'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids}

  @profile.profiled_function
  def featurize_query_and_docs(self, query, docs, model_timestamp):
    """Converts query and candidates into a set of final features for training.

    Args:
      query: a Query object.
      docs: a list of Documents.
      model_timestamp: Optional integer module export time for this example.

    Returns:
      features: a dict with the structure specified by
        self.query_and_docs_feature_structure.
    """
    # Features for the query embedder.
    query_features = self.featurize_query(query, fmt='bert_inputs')

    # Features for the document embedder.
    doc_features = batch_namedtuples(
        [self.featurize_document(d, fmt='bert_inputs') for d in docs])

    # Features for masked language model.
    query_mask_features = self.mask_query(query)

    # We need to shift all indices by 1 because we will add a [CLS] token in
    # front of the query tokens.
    mlm_positions = [i + 1 for i in query_mask_features['mask_indices']]
    mlm_mask = [1] * len(mlm_positions)
    mlm_targets = query_mask_features['masked_token_ids']

    def as_fixed_length_tensor(seq, seq_len):
      result = truncate_or_pad(seq, seq_len, pad_val=0)
      return tf.constant(result)

    mlm_mask = as_fixed_length_tensor(mlm_mask, self.max_masks)
    mlm_positions = as_fixed_length_tensor(mlm_positions, self.max_masks)
    mlm_targets = as_fixed_length_tensor(mlm_targets, self.max_masks)

    # Features for the reader.
    query_ids = query_mask_features['token_ids_after_masking']
    query_ids = query_ids[:self.query_seq_len - 2]

    joint_inputs_list = []
    for doc in docs:
      # Note that we ignore the doc title for this step.
      doc_ids = doc.body_token_ids
      doc_ids = doc_ids[:self.candidate_seq_len - 2]

      joint_inputs = bert_format(
          input_seqs=[query_ids, doc_ids],
          max_seq_len=self.query_seq_len + self.candidate_seq_len,
          cls_id=self.tokenizer.cls_id,
          sep_id=self.tokenizer.sep_id)
      joint_inputs_list.append(joint_inputs)

    joint_features = batch_namedtuples(joint_inputs_list)

    # NOTE: this feature is currently not used.
    candidate_labels = tf.zeros(len(docs))

    features = {
        'query_inputs': query_features,
        'candidate_inputs': doc_features,
        'joint_inputs': joint_features,
        'mlm_positions': mlm_positions,
        'mlm_mask': mlm_mask,
        'mlm_targets': mlm_targets,
        'candidate_labels': candidate_labels,
        'export_timestamp': tf.constant(model_timestamp),
    }

    return features

  @profile.profiled_function
  def query_and_docs_to_tf_example(self, query, docs, model_timestamp):
    features = self.featurize_query_and_docs(query, docs, model_timestamp)
    flat_features = nest_utils.nest_to_flat_dict(features)
    flat_features_numpy = {k: v.numpy() for k, v in flat_features.items()}
    example = nest_utils.flat_dict_to_tf_example(
        flat_features_numpy, self.query_and_docs_feature_structure)
    return example

  @property
  @functools.lru_cache(maxsize=None)
  def query_and_docs_feature_structure(self):
    """Nested structure of the features produced by featurize_query_and_docs.

    The leaves are all 'placeholders' with fully specified shapes and dtypes.
    This specification makes it easy to do generic, invertible conversions to
    different formats, such as flat lists or dictionaries.

    Note: we don't use `tf.placeholder` for our 'placeholder' tensors, because
    these are disallowed in eager execution mode. Instead, we just use constant
    zero Tensors.

    Returns:
      structure: A nested structure with fully-specified placeholders.
    """
    query_shape = [self.query_seq_len]
    candidate_shape = [self.num_candidates, self.candidate_seq_len]
    joint_shape = [
        self.num_candidates, self.query_seq_len + self.candidate_seq_len
    ]
    mlm_shape = [self.max_masks]
    return dict(
        query_inputs=BertInputs(
            token_ids=tf.zeros(query_shape, tf.int32),
            mask=tf.zeros(query_shape, tf.int32),
            segment_ids=tf.zeros(query_shape, tf.int32)),
        candidate_inputs=BertInputs(
            token_ids=tf.zeros(candidate_shape, tf.int32),
            mask=tf.zeros(candidate_shape, tf.int32),
            segment_ids=tf.zeros(candidate_shape, tf.int32)),
        joint_inputs=BertInputs(
            token_ids=tf.zeros(joint_shape, tf.int32),
            mask=tf.zeros(joint_shape, tf.int32),
            segment_ids=tf.zeros(joint_shape, tf.int32)),
        candidate_labels=tf.zeros([self.num_candidates], tf.float32),
        mlm_targets=tf.zeros(mlm_shape, tf.int32),
        mlm_positions=tf.zeros(mlm_shape, tf.int32),
        mlm_mask=tf.zeros(mlm_shape, tf.int32),
        export_timestamp=tf.zeros([], tf.int32))


class MaskingError(Exception):
  pass


Token = collections.namedtuple('Token', ['text', 'start', 'stop', 'id'])
"""Represents a single token in a piece of text.

Attributes:
  text (bytes): the actual text of the token
  start (int): starting byte offset (inclusive)
  stop (int): ending byte offset (exclusive)
  id (int): id of the token in the vocabulary
"""


class Tokenizer(object):
  """Tokenizes text."""

  def __init__(self, vocab_path, do_lower_case):
    if isinstance(vocab_path, bytes):
      vocab_path = vocab_path.decode()

    self.vocab_path = vocab_path
    self.do_lower_case = do_lower_case

    self._base_tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_path, do_lower_case=do_lower_case)

    # Look up special tokens.
    self.cls_id, self.sep_id, self.mask_id = (
        self._base_tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]',
                                                    '[MASK]']))

  def tokenize(self, text, compute_token_boundaries=True):
    """Returns a list of Tokens."""
    token_strs = self._base_tokenizer.tokenize(text)

    if compute_token_boundaries:
      # This runs in about half the time it takes for tokenization.
      token_boundaries = self._compute_token_boundaries(text, token_strs)
    else:
      token_boundaries = [(-1, -1)] * len(token_strs)

    token_ids = self._base_tokenizer.convert_tokens_to_ids(token_strs)

    tokens = []
    for text, tid, (start, stop) in zip(token_strs, token_ids,
                                        token_boundaries):
      tokens.append(Token(text, start, stop, tid))
    return tokens

  def token_ids_to_str(self, token_ids):
    """Converts a list of token IDs back to a human-readable string.

    WARNING: this is ONLY for debug purposes. It does NOT perfectly invert BERT
    tokenization to recover the original text.

    Args:
      token_ids: a list of ints.

    Returns:
      a str
    """
    token_strs = self._base_tokenizer.convert_ids_to_tokens(token_ids)

    formatted_strs = []
    for i, token_str in enumerate(token_strs):
      if token_str.startswith('##'):
        # No whitespace in front, and strip the double hash.
        formatted_strs.append(token_str[2:])
      else:
        if i != 0:
          # Whitespace before the token, unless you're the first token.
          formatted_strs.append(' ')
        formatted_strs.append(token_str)

    return ''.join(formatted_strs)

  def _get_char_to_byte_map(self, text):
    """Returns a map from character to byte offsets.

    Args:
      text (str): a Unicode string.

    Returns:
      a map from each character offset to a byte offset.
    """
    char_to_byte_map = {}
    char_offset = 0
    byte_offset = 0
    for character in text:
      char_to_byte_map[char_offset] = byte_offset
      char_offset += 1
      byte_offset += len(character.encode('utf-8'))
    # Corresponds to the position right after the last character.
    char_to_byte_map[char_offset] = byte_offset
    return char_to_byte_map

  # NOTE: this may not work for some non-English languages.
  def _compute_token_boundaries(self, text, token_strs):
    """Computes the byte offset boundaries for each token in a text.

    Uses greedy left-to-right string-matching to align each token with a
    position in the text.

    Args:
      text (bytes): a byte string.
      token_strs (list[str]): a list of token strings produced by the BERT
        tokenizer.

    Returns:
      A list of (start, stop) byte offsets (right-exclusive), one for each
      token.

    Raises:
      TokenizationError: if we fail to identify the byte offset for a token.
    """
    text = tokenization.convert_to_unicode(text)
    c2b = self._get_char_to_byte_map(text)

    # Normalization to match tokens.
    normalized_text = (
        self._base_tokenizer.basic_tokenizer._run_strip_accents(text))  # pylint: disable=protected-access
    if self.do_lower_case:
      normalized_text = normalized_text.lower()

    # NOTE: this is a crude check to see if lowercasing and accent-stripping
    # have altered character byte offsets. It is not perfect.
    if len(normalized_text) != len(text):
      raise TokenizationError('Normalization affected character byte offsets.')

    # Strip wordpiece prefixes.
    token_strs = [t[2:] if t.startswith('##') else t for t in token_strs]

    # Align each token with its position in normalized_text
    search_start = 0
    token_char_boundaries = []
    for token_str in token_strs:
      token_start = normalized_text.find(token_str, search_start)
      token_stop = token_start + len(token_str)
      if token_start == -1:
        raise TokenizationError('Cannot align "{}" in text: {}'.format(
            token_str, normalized_text))

      token_char_boundaries.append((token_start, token_stop))
      search_start = token_stop  # Start searching right after this token.

    # Convert char boundaries to byte boundaries
    token_byte_boundaries = [
        (c2b[start], c2b[stop]) for start, stop in token_char_boundaries
    ]
    return token_byte_boundaries


class TokenizationError(Exception):
  pass


def bert_format(input_seqs, max_seq_len, cls_id, sep_id):
  """Combines multiple sequences of tokens into a single sequence for BERT.

  The format is:
  [CLS] first seq [SEP] second seq [SEP] third_seq [SEP] ... [SEP]

  If the combined tokens exceed max_seq_len:
  - We try to keep as much of the first sequence as possible.
  - If room remains, we try to keep as much of the second sequence.
  - ... and so on.
  - We always make room for all of the [CLS] and [SEP] tokens.

  Args:
    input_seqs (list): a list of sequences, where each seq is a list of ints.
    max_seq_len (int): max sequence length.
    cls_id (int): integer ID for the [CLS] token.
    sep_id (int): integer ID for the [SEP] token.

  Returns:
    an instance of BertInputs, where each attribute is an int32
    Tensor with shape [max_seq_len].
  """
  # Overall token budget, after reserving space for special tokens.
  token_budget = max_seq_len - 1 - len(input_seqs)
  if token_budget < 0:
    raise ValueError(
        'max_seq_len not large enough to include all special tokens.')

  input_ids = [cls_id]
  segment_ids = [0]
  input_mask = [1]

  for segment_id, raw_input_seq in enumerate(input_seqs):
    # Truncate to stay within the remaining token budget.
    input_seq = raw_input_seq[:token_budget]
    input_seq_len = len(input_seq)

    input_ids.extend(input_seq)
    input_ids.append(sep_id)

    segment_ids.extend([segment_id] * (input_seq_len + 1))
    input_mask.extend([1] * (input_seq_len + 1))

    # Subtract from budget.
    token_budget -= input_seq_len

  assert len(input_ids) == len(input_mask)
  assert len(input_ids) == len(segment_ids)

  # Pad with zeroes up to max_seq_len, and convert to TF Tensor.
  as_tensor = lambda arr: tf.constant(truncate_or_pad(arr, max_seq_len, 0))
  input_ids = as_tensor(input_ids)
  input_mask = as_tensor(input_mask)
  segment_ids = as_tensor(segment_ids)

  return BertInputs(
      token_ids=input_ids, mask=input_mask, segment_ids=segment_ids)


def flatten_list(seq):
  flat = []
  for sublist in seq:
    for item in sublist:
      flat.append(item)
  return flat


def truncate_or_pad(seq, seq_len, pad_val):
  if len(seq) < seq_len:
    return seq + [pad_val] * (seq_len - len(seq))
  return seq[:seq_len]


def batch_feature_dicts(feature_dicts):
  """Combines a batch of feature dicts into a single dict.

  Args:
    feature_dicts: a list of dicts, each mapping strings to Numpy arrays or TF
      Tensors. They must all share the same keys.

  Returns:
    a single dict, where each value is the result of stacking all values from
    the original inputs along the the first dimension.
  """
  if not feature_dicts:
    return feature_dicts

  batched = {}
  feature_names = feature_dicts[0].keys()
  for feature_name in feature_names:
    batched[feature_name] = tf.stack([d[feature_name] for d in feature_dicts])
  return batched


def batch_namedtuples(namedtuples):
  """Combines a batch of namedtuples into a single namedtuple of the same type.

  Args:
    namedtuples: a list of namedtuples, all of the same type.

  Returns:
    a single namedtuple, where each value is the result of stacking all values
    from the original inputs along the the first dimension.
  """
  if not namedtuples:
    return namedtuples

  nt_type = type(namedtuples[0])
  dicts = [nt._asdict() for nt in namedtuples]
  batched_dict = batch_feature_dicts(dicts)
  return nt_type(**batched_dict)
