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
# Implicit length check with bool is incorrect with numpy/tensorflow.
# pylint: disable=g-explicit-length-test
"""Preprocessing for fever, primarily tokenizers and encoders."""
import abc
import hashlib
import json
import re


from language.serene import tokenizers
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


SEPARATOR_TITLE = '@@septitle@@'
SEPARATOR_SID_RE = '@@sid_([0-9])+@@'
SEPARATOR_RE = '@@[a-z0-9_]+@@'


def create_evidence_input(
    wikipedia_url,
    sentence_id,
    text):
  """Convert evidence inputs to their string representation.

  In particular, this means output looks like:
  Albert Einstein @@septitle@@ @@sid_1@@ the evidence text is here.

  Args:
    wikipedia_url: None or the url
    sentence_id: None or the string sentence_id
    text: evidence text

  Returns:
    The formatted string
  """
  tokens = []
  if wikipedia_url is not None:
    wikipedia_url = tf.compat.as_text(wikipedia_url)
    tokens.append(wikipedia_url.replace('_', ' '))
    tokens.append(SEPARATOR_TITLE)

  if sentence_id is not None:
    sentence_id = tf.compat.as_text(sentence_id)
    tokens.append(f'@@sid_{sentence_id}@@')

  tokens.append(text)
  return ' '.join(tokens)


def filter_claim_fn(example, _):
  """Filter out claims/evidence that have zero length."""
  if 'claim_text_word_ids' in example:
    claim_length = len(example['claim_text_word_ids'])
  else:
    claim_length = len(example['claim_text'])

  # Explicit length check required.
  # Implicit length check causes TensorFlow to fail during tracing.
  if claim_length != 0:
    return True
  else:
    return False


def filter_evidence_fn(
    example, y_input):
  # pylint: disable=unused-argument
  """Filter out claims/evidence that have zero length.

  Args:
    example: The encoded example
    y_input: Unused, contains the label, included for API compat
  Returns:
    True to preserve example, False to filter it out
  """
  # Bert encodes text in evidence_text_word_ids.
  # Word embedding model uses evidence_text.
  if 'evidence_text_word_ids' in example:
    evidence_length = len(example['evidence_text_word_ids'])
  else:
    evidence_length = len(example['evidence_text'])

  # Explicit length check required.
  # Implicit length check causes TensorFlow to fail during tracing.
  if evidence_length != 0:
    return True
  else:
    return False


class FeverTextEncoder(abc.ABC, tfds.deprecated.text.TextEncoder):
  """TextEncoder with fever specific additional methods."""

  def encode_from_json(self, example):
    return self.encode_example(
        claim=example['claim'], evidence=example['evidence'],
        claim_label=example['claim_label'],
        evidence_label=example['evidence_label'],
        metadata=example['metadata'],
        wikipedia_url=example['wikipedia_url'],
        sentence_id=str(example['sentence_id']))

  @abc.abstractmethod
  def encode_example(
      self, *,
      claim, evidence,
      claim_label, evidence_label,
      metadata,
      wikipedia_url, sentence_id):
    """Encode a single example from text/integers to a single TF example.

    Args:
      claim: Text of the claim
      evidence: Text of the evidence
      claim_label: The label as an integer
      evidence_label: The label as an integer
      metadata: Metadata dictionary, not needed, useful for debugging
      wikipedia_url: The wikipedia_url of the evidence, if any
      sentence_id: The sentence_id of the evidence, if any

    Returns:
      An example mapping dictionaries to TF tensor ready for TF to batch
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def build_encoder_fn(self):
    """Function to be called by tf.data.Dataset.map from TFDS to encoder input.

    For example, the encoder will probably want to take the text input,
    and numericalize it in a specific way, then pass this to the model.

    Returns:
      A function to use in encoding the inputs to the encoder.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def padded_shapes(self):
    """Return padding shapes for use by tf.data.Dataset.padded_batch.

    Returns:
      Padding shapes used by the encoder
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_input_shapes(self):
    """Return the expected input shapes of the encoder.

    Sometimes keras can infer this, but often with named dictionaries it does
    not infer correctly. This explicitly enumerates them so that the end model
    can be wrapped with something like
    keras.Model(inputs=inputs, outputs=outputs).

    Returns:
      Input shapes to the keras encoder
    """
    raise NotImplementedError()


# Copy/Paste from tfds.deprecated.text.TokenTextEncoder since their method for
# loading/saving is difficult to customize to add additional parameters.
class BasicTextEncoder(FeverTextEncoder):
  r"""TextEncoder backed by a list of tokens.

  Tokenization splits on (and drops) non-alphanumeric characters with
  regex "\W+".
  """

  def __init__(
      self,
      *,
      vocab_list,
      tokenizer,
      include_title,
      include_sentence_id,
      oov_buckets = 1,
      oov_token = 'UNK',
      lowercase = False,
      strip_vocab = True,
      decode_token_separator = ' ',
      max_claim_tokens = None,
      max_evidence_tokens = None):
    """Constructs a TokenTextEncoder.

    To load from a file saved with `TokenTextEncoder.save_to_file`, use
    `TokenTextEncoder.load_from_file`.

    Args:
      vocab_list: `list<str>`, list of tokens.
      tokenizer: `Tokenizer`, responsible for converting incoming text into a
        list of tokens.
      include_title: Whether to include wikipedia title in evidence
      include_sentence_id: Whether to include sentence_id in evidence
      oov_buckets: `int`, the number of `int`s to reserve for OOV hash buckets.
        Tokens that are OOV will be hash-modded into a OOV bucket in `encode`.
      oov_token: `str`, the string to use for OOV ids in `decode`.
      lowercase: `bool`, whether to make all text and tokens lowercase.
      strip_vocab: `bool`, whether to strip whitespace from the beginning and
        end of elements of `vocab_list`.
      decode_token_separator: `str`, the string used to separate tokens when
        decoding.
      max_claim_tokens: maximum number of claim tokens to use
      max_evidence_tokens: maximum number of evidence tokens to use
    """
    self._max_claim_tokens = max_claim_tokens
    self._max_evidence_tokens = max_evidence_tokens
    self._include_title = include_title
    self._include_sentence_id = include_sentence_id
    self._vocab_list = [tf.compat.as_text(el) for el in vocab_list]
    if strip_vocab:
      self._vocab_list = [el.strip() for el in self._vocab_list]
    self._lowercase = lowercase
    if self._lowercase:
      self._vocab_list = [t.lower() for t in self._vocab_list]
    # Note that internally everything is 0-indexed. Padding is dealt with at the
    # end of encode and the beginning of decode.
    self._token_to_id = dict(
        zip(self._vocab_list, range(len(self._vocab_list))))
    self._oov_buckets = oov_buckets
    self._oov_token = tf.compat.as_text(oov_token)

    self._tokenizer = tokenizer
    self._decode_token_separator = decode_token_separator

  def encode(self, s):
    s = tf.compat.as_text(s)
    if self.lowercase:
      s = s.lower()
    ids = []

    for token in self._tokenizer.tokenize(s):
      int_id = self._token_to_id.get(token, -1)
      if int_id < 0:
        int_id = self._oov_bucket(token)
        if int_id is None:
          raise ValueError('Out of vocabulary token %s' % token)
      ids.append(int_id)

    # This increments the ids of all words in ids by one, to ensure that 0
    # is preserved as the padding token
    return tfds.deprecated.text.text_encoder.pad_incr(ids)

  def decode(self, ids):
    # This decrements the ids of all words in ids by one, to ensure that 0
    # is preserved as the padding token
    # *Also*, this strips padding tokens
    ids = tfds.deprecated.text.text_encoder.pad_decr(ids)

    tokens = []
    for int_id in ids:
      if int_id < len(self._vocab_list):
        tokens.append(self._vocab_list[int_id])
      else:
        tokens.append(self._oov_token)
    return self._decode_token_separator.join(tokens)

  @property
  def vocab_size(self):
    # Plus 1 for pad.
    return len(self._vocab_list) + self._oov_buckets + 1

  @property
  def tokens(self):
    return list(self._vocab_list)

  @property
  def oov_token(self):
    return self._oov_token

  @property
  def lowercase(self):
    return self._lowercase

  @property
  def tokenizer(self):
    return self._tokenizer

  def _oov_bucket(self, token):
    if self._oov_buckets <= 0:
      return None
    if self._oov_buckets == 1:
      return len(self._vocab_list)
    hash_val = int(hashlib.md5(tf.compat.as_bytes(token)).hexdigest(), 16)
    return len(self._vocab_list) + hash_val % self._oov_buckets

  @classmethod
  def _filename(cls, filename_prefix):
    return filename_prefix + '.tokens'

  def save_to_file(self, filename_prefix):
    filename = self._filename(filename_prefix)
    kwargs = {
        'oov_buckets': self._oov_buckets,
        'lowercase': self._lowercase,
        'oov_token': self._oov_token,
        'max_claim_tokens': self._max_claim_tokens,
        'max_evidence_tokens': self._max_evidence_tokens,
        'include_title': self._include_title,
        'include_sentence_id': self._include_sentence_id,
    }
    self._tokenizer.save_to_file(filename)
    self._write_lines_to_file(filename, self._vocab_list, kwargs)

  @classmethod
  def load_from_file(cls, filename_prefix):
    filename = cls._filename(filename_prefix)
    vocab_lines, kwargs = cls._read_lines_from_file(filename)
    tokenizer = tfds.deprecated.text.Tokenizer.load_from_file(filename)
    return cls(vocab_list=vocab_lines, tokenizer=tokenizer, **kwargs)

  def encode_example(self, *, claim, evidence, claim_label, evidence_label,
                     metadata, wikipedia_url, sentence_id):
    encoded_claim = self.encode(claim)
    encoded_claim = encoded_claim[:self._max_claim_tokens]
    evidence = create_evidence_input(
        wikipedia_url if self._include_title else None,
        sentence_id if self._include_sentence_id else None,
        evidence
    )
    encoded_evidence = self.encode(evidence)
    encoded_evidence = encoded_evidence[:self._max_evidence_tokens]
    return {
        'claim_text': encoded_claim,
        'evidence_text': encoded_evidence,
        'claim_label': claim_label,
        'evidence_label': evidence_label,
        'metadata': metadata,
    }

  def build_encoder_fn(self):

    def encode(claim, evidence, claim_label, evidence_label, metadata,
               wikipedia_url, sentence_id):
      """Encode the example using the input TF example.

      Args:
        claim: Claim text
        evidence: Evidence text
        claim_label: Claim verification label
        evidence_label: Label if is relevant evidence/claim
        metadata: Metadata about example
        wikipedia_url: wikipedia_url of example, or None
        sentence_id: sentence_id of the example, or None

      Returns:
        Input fields of ClassicEmbedder model
      """
      encoded_claim = self.encode(claim.numpy())
      encoded_claim = encoded_claim[:self._max_claim_tokens]
      evidence = create_evidence_input(
          wikipedia_url.numpy().decode('utf8')
          if self._include_title else None,
          sentence_id.numpy().decode('utf8')
          if self._include_sentence_id else None,
          evidence.numpy().decode('utf8')
      )

      encoded_evidence = self.encode(evidence)
      encoded_evidence = encoded_evidence[:self._max_evidence_tokens]
      return (encoded_claim, encoded_evidence, claim_label, evidence_label,
              metadata)

    def encode_map_fn(example):
      """Read example from tfds and constructed named tensor inputs."""
      tensors = tf.py_function(
          encode,
          inp=[
              example['claim_text'], example['evidence_text'],
              example['claim_label'], example['evidence_label'],
              example['metadata'], example['wikipedia_url'],
              example['sentence_id']
          ],
          Tout=(tf.int64, tf.int64, tf.int64, tf.int64, tf.string))
      # Keras expects (x, y), where x can be a dictionary.
      x = {
          'claim_text': tensors[0],
          'evidence_text': tensors[1],
          'claim_label': tensors[2],
          'evidence_label': tensors[3],
          'metadata': tensors[4]
      }
      y = {'claim_classification': tensors[2], 'evidence_matching': tensors[3]}
      return x, y

    return encode_map_fn

  def padded_shapes(self):
    return {
        'claim_text': [-1],
        'evidence_text': [-1],
        'claim_label': [],
        'evidence_label': [],
        'metadata': []
    }

  def compute_input_shapes(self):
    return {
        'claim_text': tf.keras.Input(shape=(None,), dtype=tf.int64),
        'evidence_text': tf.keras.Input(shape=(None,), dtype=tf.int64),
        'claim_label': tf.keras.Input(shape=(1,), dtype=tf.int64),
        'evidence_label': tf.keras.Input(shape=(1,), dtype=tf.int64),
        'metadata': tf.keras.Input(shape=(1,), dtype=tf.string),
    }


class BertTextEncoder(FeverTextEncoder):
  """Bert Text encoder."""

  def __init__(
      self, *,
      tokenizer, max_seq_length,
      include_title, include_sentence_id):
    super().__init__()
    self._tokenizer = tokenizer
    self._max_seq_length = max_seq_length
    self._include_title = include_title
    self._include_sentence_id = include_sentence_id
    self._cls_id = self._tokenizer.vocab['[CLS]']
    self._sep_id = self._tokenizer.vocab['[SEP]']
    self._pad_id = self._tokenizer.vocab['[PAD]']

  def encode_example(self, *, claim, evidence, claim_label, evidence_label,
                     metadata, wikipedia_url, sentence_id):
    claim_tokens, claim_word_ids, claim_mask, claim_segment_ids = self.encode(
        claim)
    evidence = create_evidence_input(
        wikipedia_url if self._include_title else None,
        sentence_id if self._include_sentence_id else None,
        evidence
    )
    evidence_tokens, evidence_word_ids, evidence_mask, evidence_segment_ids = self.encode(
        evidence)
    return {
        'claim_text_tokens': claim_tokens,
        'claim_text_word_ids': claim_word_ids,
        'claim_text_mask': claim_mask,
        'claim_text_segment_ids': claim_segment_ids,
        'evidence_text_tokens': evidence_tokens,
        'evidence_text_word_ids': evidence_word_ids,
        'evidence_text_mask': evidence_mask,
        'evidence_text_segment_ids': evidence_segment_ids,
        'claim_label': claim_label,
        'evidence_label': evidence_label,
        'metadata': metadata
    }

  def build_encoder_fn(self):

    def encode(claim, evidence, claim_label, evidence_label, metadata,
               wikipedia_url, sentence_id):
      """Encode the example using the input TF example.

      Args:
        claim: Claim text
        evidence: Evidence text
        claim_label: Claim verification label
        evidence_label: Label if is relevant evidence/claim
        metadata: Metadata about example
        wikipedia_url: wikipedia_url of example, or None
        sentence_id: sentence_id of the example, or None

      Returns:
        Input fields of ClassicEmbedder model
      """
      claim_tokens, claim_word_ids, claim_mask, claim_segment_ids = self.encode(
          claim.numpy())
      evidence = create_evidence_input(
          wikipedia_url.numpy().decode('utf8')
          if self._include_title else None,
          sentence_id.numpy().decode('utf8')
          if self._include_sentence_id else None,
          evidence.numpy().decode('utf8')
      )

      evidence_tokens, evidence_word_ids, evidence_mask, evidence_segment_ids = self.encode(
          evidence)
      return (claim_tokens, claim_word_ids, claim_mask, claim_segment_ids,
              evidence_tokens, evidence_word_ids, evidence_mask,
              evidence_segment_ids, claim_label, evidence_label, metadata)

    def encode_map_fn(example):
      """Read example from tfds and constructed named tensor inputs."""
      tensors = tf.py_function(
          encode,
          inp=[
              example['claim_text'], example['evidence_text'],
              example['claim_label'], example['evidence_label'],
              example['metadata'], example['wikipedia_url'],
              example['sentence_id']
          ],
          Tout=(
              tf.string,
              tf.int32,
              tf.int32,
              tf.int32,
              tf.string,
              tf.int32,
              tf.int32,
              tf.int32,
              tf.int64,
              tf.int64,
              tf.string,
          ))
      # Keras expects (x, y), where x can be a dictionary.
      x = {
          'claim_text_tokens': tensors[0],
          'claim_text_word_ids': tensors[1],
          'claim_text_mask': tensors[2],
          'claim_text_segment_ids': tensors[3],
          'evidence_text_tokens': tensors[4],
          'evidence_text_word_ids': tensors[5],
          'evidence_text_mask': tensors[6],
          'evidence_text_segment_ids': tensors[7],
          'claim_label': tensors[8],
          'evidence_label': tensors[9],
          'metadata': tensors[10]
      }
      # Order matters, must match training.py.
      # [evidence_label, claim_label]
      y = {'claim_classification': tensors[8], 'evidence_matching': tensors[9]}
      return x, y

    return encode_map_fn

  def padded_shapes(self):
    return {
        'claim_text_tokens': [-1],
        'claim_text_word_ids': [-1],
        'claim_text_mask': [-1],
        'claim_text_segment_ids': [-1],
        'evidence_text_tokens': [-1],
        'evidence_text_word_ids': [-1],
        'evidence_text_mask': [-1],
        'evidence_text_segment_ids': [-1],
        'claim_label': [],
        'evidence_label': [],
        'metadata': []
    }

  def compute_input_shapes(self):
    return {
        'claim_text_tokens':
            tf.keras.Input(shape=(None,), dtype=tf.string),
        'claim_text_word_ids':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'claim_text_mask':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'claim_text_segment_ids':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'evidence_text_tokens':
            tf.keras.Input(shape=(None,), dtype=tf.string),
        'evidence_text_word_ids':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'evidence_text_mask':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'evidence_text_segment_ids':
            tf.keras.Input(shape=(None,), dtype=tf.int32),
        'claim_label':
            tf.keras.Input(shape=(1,), dtype=tf.int64),
        'evidence_label':
            tf.keras.Input(shape=(1,), dtype=tf.int64),
        'metadata':
            tf.keras.Input(shape=(1,), dtype=tf.string),
    }

  def tokenize(self, text):
    return self._tokenizer.tokenize(text)

  def encode(self, s):
    full_text = tf.compat.as_text(s)
    if self._include_title or self._include_sentence_id:
      evidence_position = 0
      title_match = re.search(SEPARATOR_TITLE, full_text)
      if title_match is None:
        title_text = None
      else:
        # Grab everything before the title token
        start = title_match.start()
        title_text = full_text[:start].strip()
        evidence_position = title_match.end()

      sentence_id_match = re.search(SEPARATOR_SID_RE, full_text)
      if sentence_id_match is None:
        sentence_id_text = None
      else:
        sentence_id_text = sentence_id_match.group(1).strip()
        evidence_position = sentence_id_match.end()

      left_tokens = []
      if title_text is not None:
        left_tokens.extend(self._tokenizer.tokenize(title_text))

      if sentence_id_text is not None:
        left_tokens.extend(self._tokenizer.tokenize(sentence_id_text))

      evidence_text = full_text[evidence_position:].strip()
      right_tokens = self._tokenizer.tokenize(evidence_text)

      left_word_ids = self._tokenizer.convert_tokens_to_ids(left_tokens)
      right_word_ids = self._tokenizer.convert_tokens_to_ids(right_tokens)
      word_ids, masks, segment_ids = self._pad_or_truncate_pair(
          left_word_ids, right_word_ids)
      return left_tokens + right_tokens, word_ids, masks, segment_ids
    else:
      tokens = self._tokenizer.tokenize(full_text)
      word_ids = self._tokenizer.convert_tokens_to_ids(tokens)
      word_ids, masks, segment_ids = self._pad_or_truncate(word_ids)
      return tokens, word_ids, masks, segment_ids

  def decode(self, ids):
    raise NotImplementedError()

  def vocab_size(self):
    return len(self._tokenizer.vocab)

  def save_to_file(self, filename_prefix):
    vocab_file = f'{filename_prefix}.vocab'
    tf.io.gfile.copy(self._tokenizer.vocab_file, vocab_file, overwrite=True)
    with tf.io.gfile.GFile(f'{filename_prefix}.encoder', 'w') as f:
      json.dump({
          'vocab_file': vocab_file,
          'do_lower_case': self._tokenizer.do_lower_case,
          'max_seq_length': self._max_seq_length,
          'include_title': self._include_title,
          'include_sentence_id': self._include_sentence_id,
      }, f)

  @classmethod
  def load_from_file(cls, filename_prefix):
    with tf.io.gfile.GFile(f'{filename_prefix}.encoder') as f:
      params = json.load(f)
    bert_tokenizer = tokenizers.BertTokenizer(
        vocab_file=params['vocab_file'],
        do_lower_case=params['do_lower_case']
    )
    bert_encoder = BertTextEncoder(
        tokenizer=bert_tokenizer,
        max_seq_length=params['max_seq_length'],
        include_title=params['include_title'],
        include_sentence_id=params['include_sentence_id']
    )
    return bert_encoder

  @classmethod
  def _write_lines_to_file(cls, filename, lines, metadata_dict=None):
    raise NotImplementedError()

  @classmethod
  def _read_lines_from_file(cls, filename):
    raise NotImplementedError()

  # pylint:disable=line-too-long
  # Modified: google/third_party/py/language/google/orqa/google/utils/bert_utils.py
  # pylint:enable=line-too-long
  def _pad_or_truncate(self, token_ids):
    token_ids = token_ids[:self._max_seq_length - 2]
    truncated_len = tf.size(token_ids)
    padding = tf.zeros([self._max_seq_length - 2 - truncated_len], tf.int32)
    token_ids = tf.concat([[self._cls_id], token_ids, [self._sep_id], padding],
                          0)
    masks = tf.concat([tf.ones([truncated_len + 2], tf.int32), padding], 0)
    token_ids.set_shape([self._max_seq_length])
    masks.set_shape([self._max_seq_length])
    segment_ids = tf.zeros([self._max_seq_length], tf.int32)
    return token_ids, masks, segment_ids

  def _pad_or_truncate_pair(
      self, token_ids_a, token_ids_b):
    """Pad or truncate pair."""
    token_ids_a = token_ids_a[:self._max_seq_length - 3]
    truncated_len_a = tf.size(token_ids_a)
    maximum_len_b = tf.maximum(self._max_seq_length - 3 - truncated_len_a, 0)
    token_ids_b = token_ids_b[:maximum_len_b]
    truncated_len_b = tf.size(token_ids_b)
    truncated_len_pair = truncated_len_a + truncated_len_b
    padding = tf.zeros([self._max_seq_length - 3 - truncated_len_pair],
                       tf.int32)
    token_ids = tf.concat([[self._cls_id], token_ids_a, [self._sep_id],
                           token_ids_b, [self._sep_id], padding], 0)
    mask = tf.concat([tf.ones([truncated_len_pair + 3], tf.int32), padding], 0)
    segment_ids = tf.concat([
        tf.zeros([truncated_len_a + 2], tf.int32),
        tf.ones([truncated_len_b + 1], tf.int32), padding
    ], 0)
    token_ids.set_shape([self._max_seq_length])
    mask.set_shape([self._max_seq_length])
    segment_ids.set_shape([self._max_seq_length])

    return token_ids, mask, segment_ids
