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
"""Utilities to encode text given BERT model."""

from absl import flags

from bert import modeling
from language.labs.drkit import run_dualencoder_qa
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tqdm import tqdm

FLAGS = flags.FLAGS


def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False


def preprocess_text(text, tokenizer):
  """Tokenize and convert raw text to ids.

  Args:
    text: String to be tokenized.
    tokenizer: An instance of Tokenizer class.

  Returns:
    tokens: List of strings.
    char_to_token_map: List of token ids mapping each character in the original
      string to the first and last index of the split tokens it appears in.
  """
  # Split by whitespace.
  wspace_tokens = []
  char_to_wspace = []
  prev_is_whitespace = True
  for c in text:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        wspace_tokens.append(c)
      else:
        wspace_tokens[-1] += c
      prev_is_whitespace = False
    char_to_wspace.append(len(wspace_tokens) - 1)

  # Tokenize each split token.
  orig_to_tok_index = []
  tokens = []
  for token in wspace_tokens:
    orig_to_tok_index.append(len(tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tokens.append(sub_token)

  # Map characters to tokens.
  char_to_token_map = []
  for cc in char_to_wspace:
    st = orig_to_tok_index[cc]
    if cc == len(orig_to_tok_index) - 1:
      en = len(tokens) - 1
    else:
      en = orig_to_tok_index[cc + 1] - 1
    char_to_token_map.append((st, en))

  return tokens, char_to_token_map


class FastPredict:
  """Class to prevent re-initialization of Estimator when predicting."""

  def __init__(self, estimator, input_fn):
    self.estimator = estimator
    self.first_run = True
    self.closed = False
    self.input_fn = input_fn

  def _create_generator(self):
    while not self.closed:
      for feature in self.next_features:
        yield feature

  def predict(self, feature_batch):
    """Runs a prediction on a set of features.

    Calling multiple times does *not* regenerate the graph which makes predict
    much faster.

    Args:
      feature_batch: A list of list of features. IMPORTANT: If you're only
        classifying 1 thing, you still need to make it a batch of 1 by wrapping
        it in a list (i.e. predict([my_feature]), not predict(my_feature).

    Returns:
      results: A list of results from estimator.predict.
    """
    self.next_features = feature_batch
    if self.first_run:
      self.batch_size = len(feature_batch)
      self.predictions = self.estimator.predict(
          input_fn=self.input_fn(self._create_generator))
      self.first_run = False
    elif self.batch_size != len(feature_batch):
      raise ValueError("All batches must be of the same size. First-batch:" +
                       str(self.batch_size) + " This-batch:" +
                       str(len(feature_batch)))

    results = []
    for _ in range(self.batch_size):
      results.append(next(self.predictions))
    return results


class BERTPredictor:
  """Wrapper around a BERT model to encode text."""

  def __init__(self, tokenizer, init_checkpoint, estimator=None):
    """Setup BERT model."""
    self.max_seq_length = FLAGS.max_seq_length
    self.max_qry_length = FLAGS.max_query_length
    self.max_ent_length = FLAGS.max_entity_length
    self.batch_size = FLAGS.predict_batch_size
    self.tokenizer = tokenizer

    if estimator is None:
      bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
      run_config = tf_estimator.tpu.RunConfig()
      qa_config = run_dualencoder_qa.QAConfig(
          doc_layers_to_use=FLAGS.doc_layers_to_use,
          doc_aggregation_fn=FLAGS.doc_aggregation_fn,
          qry_layers_to_use=FLAGS.qry_layers_to_use,
          qry_aggregation_fn=FLAGS.qry_aggregation_fn,
          projection_dim=FLAGS.projection_dim,
          normalize_emb=FLAGS.normalize_emb,
          share_bert=True,
          exclude_scopes=None)

      model_fn_builder = run_dualencoder_qa.model_fn_builder
      model_fn = model_fn_builder(
          bert_config=bert_config,
          qa_config=qa_config,
          init_checkpoint=init_checkpoint,
          learning_rate=0.0,
          num_train_steps=0,
          num_warmup_steps=0,
          use_tpu=False,
          use_one_hot_embeddings=False)

      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf_estimator.tpu.TPUEstimator(
          use_tpu=False,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=self.batch_size,
          predict_batch_size=self.batch_size)

    self.fast_predictor = FastPredict(estimator, self.get_input_fn)
    self.emb_dim = FLAGS.projection_dim

  def get_input_fn(self, generator):
    """Return an input_fn which accepts a generator."""

    def _input_fn(params):
      """Convert input into features."""
      del params
      seq_length = self.max_seq_length
      qry_length = self.max_qry_length
      ent_length = self.max_ent_length
      d = tf.data.Dataset.from_generator(
          generator,
          output_types={
              "unique_ids": tf.int32,
              "doc_input_ids": tf.int32,
              "doc_input_mask": tf.int32,
              "doc_segment_ids": tf.int32,
              "qry_input_ids": tf.int32,
              "qry_input_mask": tf.int32,
              "qry_segment_ids": tf.int32,
              "ent_input_ids": tf.int32,
              "ent_input_mask": tf.int32,
          },
          output_shapes={
              "unique_ids": tf.TensorShape([]),
              "doc_input_ids": tf.TensorShape([seq_length]),
              "doc_input_mask": tf.TensorShape([seq_length]),
              "doc_segment_ids": tf.TensorShape([seq_length]),
              "qry_input_ids": tf.TensorShape([qry_length]),
              "qry_input_mask": tf.TensorShape([qry_length]),
              "qry_segment_ids": tf.TensorShape([qry_length]),
              "ent_input_ids": tf.TensorShape([ent_length]),
              "ent_input_mask": tf.TensorShape([ent_length]),
          })
      d = d.batch(batch_size=self.batch_size)
      return d

    return _input_fn

  def _run_on_features(self, features):
    """Run predictions for given features."""
    current_size = len(features)
    if current_size < self.batch_size:
      features += [features[-1]] * (self.batch_size - current_size)
    return self.fast_predictor.predict(features)[:current_size]

  def get_features(self, doc_tokens, qry_tokens, ent_tokens, uid):
    """Convert list of tokens to a feature dict."""
    max_tokens_doc = self.max_seq_length - 2
    max_tokens_qry = self.max_qry_length - 2
    max_tokens_ent = self.max_ent_length
    doc_input_ids = self.tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + doc_tokens[:max_tokens_doc] + ["[SEP]"])
    doc_segment_ids = [1] * len(doc_input_ids)
    doc_input_mask = [1] * len(doc_input_ids)
    while len(doc_input_ids) < self.max_seq_length:
      doc_input_ids.append(0)
      doc_input_mask.append(0)
      doc_segment_ids.append(0)
    qry_input_ids = self.tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + qry_tokens[:max_tokens_qry] + ["[SEP]"])
    qry_segment_ids = [0] * len(qry_input_ids)
    qry_input_mask = [1] * len(qry_input_ids)
    while len(qry_input_ids) < self.max_qry_length:
      qry_input_ids.append(0)
      qry_input_mask.append(0)
      qry_segment_ids.append(0)
    ent_input_ids = self.tokenizer.convert_tokens_to_ids(
        ent_tokens[:max_tokens_ent])
    ent_input_mask = [1] * len(ent_input_ids)
    while len(ent_input_ids) < self.max_ent_length:
      ent_input_ids.append(0)
      ent_input_mask.append(0)
    return {
        "unique_ids": uid,
        "doc_input_ids": doc_input_ids,
        "doc_input_mask": doc_input_mask,
        "doc_segment_ids": doc_segment_ids,
        "qry_input_ids": qry_input_ids,
        "qry_input_mask": qry_input_mask,
        "qry_segment_ids": qry_segment_ids,
        "ent_input_ids": ent_input_ids,
        "ent_input_mask": ent_input_mask,
    }

  def get_doc_embeddings(self, docs):
    """Run BERT to get features for docs.

    Args:
      docs: List of list of tokens.

    Returns:
      embeddings: Numpy array of token features.
    """
    num_batches = (len(docs) // self.batch_size) + 1
    tf.logging.info("Total batches for BERT = %d", num_batches)
    embeddings = np.empty((len(docs), self.max_seq_length, 2 * self.emb_dim),
                          dtype=np.float32)
    for nb in tqdm(range(num_batches)):
      min_ = nb * self.batch_size
      max_ = (nb + 1) * self.batch_size
      if min_ >= len(docs):
        break
      if max_ > len(docs):
        max_ = len(docs)
      current_features = [
          self.get_features(docs[ii], ["dummy"], ["dummy"], ii)
          for ii in range(min_, max_)
      ]
      results = self._run_on_features(current_features)
      for ir, rr in enumerate(results):
        embeddings[min_ + ir, :, :] = rr["doc_features"]
    return embeddings[:, 1:, :]  # remove [CLS]

  def get_qry_embeddings(self, qrys, ents):
    """Run BERT to get features for queries.

    Args:
      qrys: List of list of tokens.
      ents: List of list of tokens.

    Returns:
      st_embeddings: Numpy array of token features.
      en_embeddings: Numpy array of token features.
      bow_embeddings: Numpy array of token features.
    """
    num_batches = (len(qrys) // self.batch_size) + 1
    tf.logging.info("Total batches for BERT = %d", num_batches)
    st_embeddings = np.empty((len(qrys), self.emb_dim), dtype=np.float32)
    en_embeddings = np.empty((len(qrys), self.emb_dim), dtype=np.float32)
    bow_embeddings = np.empty((len(qrys), self.emb_dim), dtype=np.float32)
    for nb in tqdm(range(num_batches)):
      min_ = nb * self.batch_size
      max_ = (nb + 1) * self.batch_size
      if min_ >= len(qrys):
        break
      if max_ > len(qrys):
        max_ = len(qrys)
      current_features = [
          self.get_features(["dummy"], qrys[ii], ents[ii], ii)
          for ii in range(min_, max_)
      ]
      results = self._run_on_features(current_features)
      for ir, rr in enumerate(results):
        st_embeddings[min_ + ir, :] = rr["qry_st_features"]
        en_embeddings[min_ + ir, :] = rr["qry_en_features"]
        bow_embeddings[min_ + ir, :] = rr["qry_bow_features"]
    return st_embeddings, en_embeddings, bow_embeddings
