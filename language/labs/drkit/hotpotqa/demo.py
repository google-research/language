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
# coding=utf-8
"""Run relation following over pre-trained corpus index."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import json
import os
import re
import urllib

from absl import flags
import jinja2
from bert import modeling
from bert import tokenization
from language.labs.drkit import input_fns
from language.labs.drkit import model_fns
from language.labs.drkit import search_utils

import numpy as np
from sklearn.preprocessing import normalize
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tornado.web
import tornado.wsgi
from tensorflow.contrib import tpu as contrib_tpu

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("port", 8080, "Port to listen on.")

flags.DEFINE_string("web_path", "", "Directory containing all web resources")

flags.DEFINE_string("mode", "demo", "Whether to run demo or offline eval.")

## Other parameters
flags.DEFINE_string("predict_file", None,
                    "For eval mode: input file to predict on.")

flags.DEFINE_string("answer_file", None,
                    "For eval mode: output file to store results.")

flags.DEFINE_string("model_type", "onehop",
                    "Whether to use `onehop` or `twohop` model.")

flags.DEFINE_integer("num_para", 5, "Number of paragraphs to train on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "hotpot_init_checkpoint", None,
    "Initial checkpoint for answer prediction.")

flags.DEFINE_string(
    "raw_passages", None,
    "File containing Wiki passages.")

flags.DEFINE_string(
    "train_data_dir", None,
    "Location of entity / mention files for training data.")

flags.DEFINE_integer(
    "num_entities_linked", 20, "Number of entities to extract from question.")

flags.DEFINE_integer(
    "num_hops", 2, "Number of hops in rule template.")

flags.DEFINE_integer(
    "max_entity_len", 15,
    "Maximum number of tokens in an entity name.")

flags.DEFINE_integer(
    "num_mips_neighbors", 15000,
    "Number of nearest neighbor mentions to retrieve for queries in each hop.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("projection_dim", 200,
                     "Number of dimensions to project embeddings to. "
                     "Set to None to use full dimensions.")

flags.DEFINE_integer(
    "max_hotpot_seq_length", 512, "Maximum context for hotpot.")

flags.DEFINE_integer(
    "max_hotpot_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "max_answer_length", 30, "Maximum length of answer.")

flags.DEFINE_integer(
    "max_query_length", 30,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_float("entity_score_threshold", 1e-2,
                   "Minimum score of an entity to retrieve sparse neighbors.")

flags.DEFINE_float("softmax_temperature", 2.,
                   "Temperature before computing softmax.")

flags.DEFINE_string("sparse_reduce_fn", "max",
                    "Function to aggregate sparse search results for a set of "
                    "entities.")

flags.DEFINE_string("sparse_strategy", "dense_first",
                    "How to combine sparse and dense components.")

flags.DEFINE_boolean("light", False,
                     "If true run in light mode.")

flags.DEFINE_string(
    "qry_layers_to_use", "-1",
    "Comma-separated list of layer representations to use as the fixed "
    "query representation.")

flags.DEFINE_string(
    "qry_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`qry_layers`.")

flags.DEFINE_string(
    "entity_score_aggregation_fn", "max",
    "Aggregation method for combining the mention logits to entities.")

flags.DEFINE_float("question_dropout", 0.2,
                   "Dropout probability for question BiLSTMs.")

flags.DEFINE_integer("question_num_layers", 5,
                     "Number of layers for question BiLSTMs.")

flags.DEFINE_boolean("ensure_answer_sparse", False,
                     "If true, ensures answer is among sparse retrieval results"
                     "during training.")

flags.DEFINE_boolean("ensure_answer_dense", False,
                     "If true, ensures answer is among dense retrieval results "
                     "during training.")

flags.DEFINE_boolean("train_with_sparse", True,
                     "If true, multiplies logits with sparse retrieval results "
                     "during training.")

flags.DEFINE_boolean("predict_with_sparse", True,
                     "If true, multiplies logits with sparse retrieval results "
                     "during inference.")

flags.DEFINE_boolean("fix_sparse_to_one", True,
                     "If true, sparse search matrix is fixed to {0,1}.")

flags.DEFINE_boolean("l2_normalize_db", False,
                     "If true, pre-trained embeddings are normalized to 1.")

flags.DEFINE_boolean("load_only_bert", False,
                     "To load only BERT variables from init_checkpoint.")

flags.DEFINE_boolean("use_best_ckpt_for_predict", False,
                     "If True, loads the best_model checkpoint in model_dir, "
                     "instead of the latest one.")

flags.DEFINE_bool("profile_model", False, "Whether to run profiling.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("random_seed", 1,
                     "Random seed for reproducibility.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


class QAConfig(object):
  """Hyperparameters for the QA model."""

  def __init__(self,
               qry_layers_to_use,
               qry_aggregation_fn,
               dropout,
               qry_num_layers,
               projection_dim,
               num_entities,
               max_entity_len,
               ensure_answer_sparse,
               ensure_answer_dense,
               train_with_sparse,
               predict_with_sparse,
               fix_sparse_to_one,
               supervision,
               l2_normalize_db,
               entity_score_aggregation_fn,
               entity_score_threshold,
               softmax_temperature,
               sparse_reduce_fn,
               intermediate_loss,
               train_batch_size,
               predict_batch_size,
               light,
               sparse_strategy,
               load_only_bert):
    self.qry_layers_to_use = [int(vv) for vv in qry_layers_to_use.split(",")]
    self.qry_aggregation_fn = qry_aggregation_fn
    self.dropout = dropout
    self.qry_num_layers = qry_num_layers
    self.projection_dim = projection_dim
    self.num_entities = num_entities
    self.max_entity_len = max_entity_len
    self.load_only_bert = load_only_bert
    self.ensure_answer_sparse = ensure_answer_sparse
    self.ensure_answer_dense = ensure_answer_dense
    self.train_with_sparse = train_with_sparse
    self.predict_with_sparse = predict_with_sparse
    self.fix_sparse_to_one = fix_sparse_to_one
    self.supervision = supervision
    self.l2_normalize_db = l2_normalize_db
    self.entity_score_aggregation_fn = entity_score_aggregation_fn
    self.entity_score_threshold = entity_score_threshold
    self.softmax_temperature = softmax_temperature
    self.sparse_reduce_fn = sparse_reduce_fn
    self.intermediate_loss = intermediate_loss
    self.train_batch_size = train_batch_size
    self.predict_batch_size = predict_batch_size
    self.light = light
    self.sparse_strategy = sparse_strategy


class MipsConfig(object):
  """Hyperparameters for the QA model."""

  def __init__(self,
               ckpt_path,
               ckpt_var_name,
               num_mentions,
               emb_size,
               num_neighbors):
    self.ckpt_path = ckpt_path
    self.ckpt_var_name = ckpt_var_name
    self.num_mentions = num_mentions
    self.emb_size = emb_size
    self.num_neighbors = num_neighbors


class FastPredict(object):
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


def detokenize_wordpiece(toks):
  """Combine split tokens from worpiece."""
  tok_text = " ".join(toks)
  # De-tokenize WordPieces that have been split off.
  tok_text = tok_text.replace(" ##", "")
  tok_text = tok_text.replace("##", "")
  # Clean whitespace
  tok_text = tok_text.strip()
  tok_text = " ".join(tok_text.split())
  tok_text = tok_text.replace(" - ", "-").replace(" ' s", "'s").replace(
      " , 000", ",000")
  return tok_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def hotpot_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                            num_train_steps, num_warmup_steps, use_tpu,
                            use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  del learning_rate, num_train_steps, num_warmup_steps

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    (start_logits,
     end_logits,
     qtype_logits,
     sp_logits) = model_fns.create_hotpot_answer_model(
         bert_config=bert_config,
         is_training=is_training,
         input_ids=input_ids,
         input_mask=input_mask,
         segment_ids=segment_ids,
         use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf_estimator.ModeKeys.PREDICT:
      sp_mask = tf.cast(features["supporting_mask"], tf.float32)  # B x N
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "qtype_logits": qtype_logits,
          "sp_logits": sp_logits * sp_mask,
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def to_tokens(text, tokenizer):
  """Tokenize the text and return mapping from chars to tokens."""
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in text:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)
  orig_to_tok_index = []
  all_doc_tokens = []
  for token in doc_tokens:
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      all_doc_tokens.append(sub_token)
  char_to_final_offset = []
  for c in char_to_word_offset:
    char_to_final_offset.append(orig_to_tok_index[c])
  return all_doc_tokens, char_to_final_offset


class BERTPredictor(object):
  """Wrapper around a BERT model to make hotpot predictions."""

  def __init__(self, tokenizer, init_checkpoint):
    """Setup BERT model."""
    self.max_seq_length = FLAGS.max_hotpot_seq_length
    self.max_qry_length = FLAGS.max_hotpot_query_length
    self.batch_size = 1
    self.tokenizer = tokenizer
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    with tf.device("/cpu:0"):
      model_fn = hotpot_model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=init_checkpoint,
          learning_rate=0.0,
          num_train_steps=0,
          num_warmup_steps=0,
          use_tpu=False,
          use_one_hot_embeddings=False)
    run_config = contrib_tpu.RunConfig()
    estimator = contrib_tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=self.batch_size,
        predict_batch_size=self.batch_size)
    self.fast_predictor = FastPredict(estimator,
                                      self.get_input_fn)
    self._PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])

  def get_input_fn(self, generator):
    """Return an input_fn which accepts a generator."""

    def _input_fn(params):
      """Convert input into features."""
      del params
      seq_length = self.max_seq_length
      d = tf.data.Dataset.from_generator(
          generator,
          output_types={
              "unique_ids": tf.int32,
              "input_ids": tf.int32,
              "input_mask": tf.int32,
              "segment_ids": tf.int32,
              "supporting_mask": tf.int32,
              },
          output_shapes={
              "unique_ids": tf.TensorShape([]),
              "input_ids": tf.TensorShape([seq_length]),
              "input_mask": tf.TensorShape([seq_length]),
              "segment_ids": tf.TensorShape([seq_length]),
              "supporting_mask": tf.TensorShape([seq_length]),
              })
      d = d.batch(batch_size=self.batch_size)
      return d

    return _input_fn

  def postprocess(self, doc_tokens, token_to_sp_fact_map,
                  result, n_best_size, max_answer_length):
    """Get final prediction from logits."""
    # Collect supporting facts.
    supporting_facts = set()
    sp_indexes = np.where(np.asarray(result["sp_logits"]) > 0.)[0]
    for spi in sp_indexes:
      if spi in token_to_sp_fact_map:
        supporting_facts.add(token_to_sp_fact_map[spi])
    supporting_facts = [list(x) for x in supporting_facts]

    # Check if answer is yes / no.
    probs = result["qtype_logits"]
    tf.logging.info(probs)
    max_i = np.argmax(probs)
    tf.logging.info(max_i)
    if max_i == 1:
      return "yes", supporting_facts
    elif max_i == 2:
      return "no", supporting_facts
    start_indexes = _get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = _get_best_indexes(result["end_logits"], n_best_size)
    prelim_predictions = []
    for start_index in start_indexes:
      for end_index in end_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(doc_tokens):
          continue
        if end_index >= len(doc_tokens):
          continue
        if end_index < start_index:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        prelim_predictions.append(
            self._PrelimPrediction(
                start_index=start_index,
                end_index=end_index,
                start_logit=result["start_logits"][start_index],
                end_logit=result["end_logits"][end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_pos", "end_pos"])

    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = doc_tokens[pred.start_index:(pred.end_index + 1)]
        final_text = detokenize_wordpiece(tok_tokens)
        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_pos=pred.start_index,
                end_pos=pred.end_index))
    if not nbest:
      return "empty", supporting_facts
    else:
      return nbest[0].text, supporting_facts

  def _run_on_features(self, features):
    """Run predictions for given features."""
    current_size = len(features)
    if current_size < self.batch_size:
      features += [features[-1]] * (self.batch_size - current_size)
    return self.fast_predictor.predict(features)[:current_size]

  def get_features(self, doc_tokens, qry_tokens, pos_to_sp, uid):
    """Convert list of tokens to a feature dict."""
    max_tokens_doc = self.max_seq_length - self.max_qry_length - 1
    max_tokens_qry = self.max_qry_length - 2
    doc_input_ids = self.tokenizer.convert_tokens_to_ids(
        doc_tokens[:max_tokens_doc] + ["[SEP]"])
    doc_segment_ids = [1] * len(doc_input_ids)
    doc_input_mask = [1] * len(doc_input_ids)
    qry_input_ids = self.tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + qry_tokens[:max_tokens_qry] + ["[SEP]"])
    qry_segment_ids = [0] * len(qry_input_ids)
    qry_input_mask = [1] * len(qry_input_ids)
    input_ids = qry_input_ids + doc_input_ids
    input_mask = qry_input_mask + doc_input_mask
    segment_ids = qry_segment_ids + doc_segment_ids
    all_tokens = (["[CLS]"] + qry_tokens[:max_tokens_qry] + ["[SEP]"] +
                  doc_tokens[:max_tokens_doc] + ["[SEP]"])
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    doc_start = len(qry_tokens[:max_tokens_qry]) + 2
    supporting_mask = []
    for ii in range(self.max_seq_length):
      if ii - doc_start in pos_to_sp:
        supporting_mask.append(1)
      else:
        supporting_mask.append(0)
    return {
        "unique_ids": uid,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "supporting_mask": supporting_mask,
    }, all_tokens, doc_start

  def get_prediction(self, doc_tokens, query_tokens, pos_to_sp_fact):
    """Run BERT to get answer prediction."""
    tf.logging.info("Features for answer BERT")
    current_features, current_toks, doc_start = self.get_features(
        doc_tokens, query_tokens, pos_to_sp_fact, 0)
    tf.logging.info("Running answer BERT")
    result = self._run_on_features([current_features])
    tf.logging.info("Mapping positions to supporting facts")
    token_to_sp_fact_map = {}
    for pos, sp in pos_to_sp_fact.items():
      token_to_sp_fact_map[doc_start + pos] = sp
    tf.logging.info("Postprocess answer.")
    answer, sp_facts = self.postprocess(
        current_toks, token_to_sp_fact_map,
        result[0], 20, FLAGS.max_answer_length)
    return answer, sp_facts


class Predictor(object):
  """Wrapper around an estimator to predict given text."""

  def __init__(self, tokenizer, estimator, entity2id, id2name,
               all_passages, answer_predictor,
               base_dir, num_entities, linker="tfidf"):
    """Setup BERT model."""
    self.max_qry_length = FLAGS.max_query_length
    self.batch_size = 1
    self.tokenizer = tokenizer
    self.entity2id = entity2id
    self.id2entity = {i: e for e, i in self.entity2id.items()}
    self.id2name = id2name
    self.all_passages = all_passages
    self.answer_predictor = answer_predictor
    self.num_entities = num_entities
    self.fast_predictor = FastPredict(estimator, self.get_input_fn)
    self.linker = linker
    self.load_entities(base_dir)

  def load_entities(self, base_dir):
    """Load entity ids and masks."""
    tf.reset_default_graph()
    id_ckpt = os.path.join(base_dir, "entity_ids")
    entity_ids = search_utils.load_database(
        "entity_ids", None, id_ckpt, dtype=tf.int32)
    mask_ckpt = os.path.join(base_dir, "entity_mask")
    entity_mask = search_utils.load_database(
        "entity_mask", None, mask_ckpt)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      tf.logging.info("Loading entity ids and masks...")
      np_ent_ids, np_ent_mask = sess.run([entity_ids, entity_mask])
    tf.logging.info("Building entity count matrix...")
    entity_count_matrix = search_utils.build_count_matrix(np_ent_ids,
                                                          np_ent_mask)
    tf.logging.info("Computing IDFs...")
    self.idfs = search_utils.counts_to_idfs(entity_count_matrix, cutoff=1e-5)
    tf.logging.info("Computing entity Tf-IDFs...")
    ent_tfidfs = search_utils.counts_to_tfidf(entity_count_matrix, self.idfs)
    self.ent_tfidfs = normalize(ent_tfidfs, norm="l2", axis=0)

  def tfidf_linking(self, qry_ids, qry_mask, top_k):
    """Match questions to entities via Tf-IDF."""
    tf.logging.info("Building question count matrix...")
    question_count_matrix = search_utils.build_count_matrix([qry_ids],
                                                            [qry_mask])
    tf.logging.info("Computing question TF-IDFs...")
    qry_tfidfs = search_utils.counts_to_tfidf(question_count_matrix, self.idfs)
    qry_tfidfs = normalize(qry_tfidfs, norm="l2", axis=0)
    tf.logging.info("Searching...")
    distances = qry_tfidfs.transpose().dot(self.ent_tfidfs)[0, :].tocsr()
    if len(distances.data) <= top_k:
      o_sort = np.argsort(-distances.data)
    else:
      o_sort = np.argpartition(-distances.data, top_k)[:top_k]
    top_doc_indices = distances.indices[o_sort]
    mentions = []
    for m in range(top_k):
      mentions.append(self.id2entity[top_doc_indices[m]])
    tf.logging.info("Mentions: %r", mentions)
    return mentions

  def get_input_fn(self, generator):
    """Return an input_fn which accepts a generator."""

    def _input_fn(params):
      """Convert input into features."""
      del params
      qry_length = self.max_qry_length
      d = tf.data.Dataset.from_generator(
          generator,
          output_types={
              "qas_ids": tf.int32,
              "qry_input_ids": tf.int32,
              "qry_input_mask": tf.int32,
              "qry_entity_id": tf.int32,
              "num_entities": tf.int32,
          },
          output_shapes={
              "qas_ids": tf.TensorShape([]),
              "qry_input_ids": tf.TensorShape([qry_length]),
              "qry_input_mask": tf.TensorShape([qry_length]),
              "qry_entity_id": tf.TensorShape([self.num_entities]),
              "num_entities": tf.TensorShape([]),
          })
      d = d.batch(batch_size=self.batch_size)
      return d

    return _input_fn

  def postprocess(self, preds, query):
    """Post-process model outputs to readable format."""
    # Combine all paragraphs into one context.
    tf.logging.info("Constructing evidence")
    offset = 0
    doc_tokens = []
    evidences = []
    pos_to_sp_fact = {}
    for pp in preds["top_idx"][:FLAGS.num_para]:
      title, passage, mentions = self.all_passages[self.id2entity[pp]]
      sentences = passage.split(" . ")
      evidences.append([title, sentences, mentions])
      sentences = [sentence + " ." for sentence in sentences]
      # sentences = [title] + sentences
      for jj, sentence in enumerate(sentences):
        # pos_to_sp_fact[offset] = tuple([title, jj-1])
        pos_to_sp_fact[offset] = tuple([title, jj])
        sentence = sentence.strip()
        tokens, _ = to_tokens(sentence, self.tokenizer)
        doc_tokens.extend(tokens)
        offset += len(tokens)
        if jj < len(sentences) - 1:
          doc_tokens.append("[unused0]")
          offset += 1
      doc_tokens.append("[PAD]")
      offset += 1
    tf.logging.info(" ".join(doc_tokens))
    query_tokens = self.tokenizer.tokenize(query)
    tf.logging.info("query: %s", " ".join(query_tokens))
    answer, sp_facts = self.answer_predictor.get_prediction(
        doc_tokens, query_tokens, pos_to_sp_fact)
    tf.logging.info("%s :: %r", answer, sp_facts)
    return answer, evidences, sp_facts

  def run(self, query):
    """Run model on given query and entity.

    Args:
      query: String.

    Returns:
      logits: Numpy array of logits for each output entity.
    """
    tf.logging.info("Tokenizing query: %s", query)
    qry_input_ids, qry_input_mask, _ = input_fns.get_tokens_and_mask(
        query, self.tokenizer, self.max_qry_length)
    tf.logging.info("Finding entities using %s", self.linker)
    mentions = self.tfidf_linking(qry_input_ids, qry_input_mask,
                                  self.num_entities)
    tf.logging.info("Converting to features")
    entities = [self.entity2id[mention] for mention in mentions]
    current_features = [{
        "qas_ids": 0,
        "qry_input_ids": qry_input_ids,
        "qry_input_mask": qry_input_mask,
        "qry_entity_id": entities + [0] * (self.num_entities - len(entities)),
        "num_entities": len(entities),
    }]
    tf.logging.info("Running DrKIT")
    preds = self.fast_predictor.predict(current_features)[0]
    return self.postprocess(preds, query)

  def run_given_entities(self, query, entities):
    """Run model given the passage title entities."""
    preds = {"top_idx": []}
    for ee in entities:
      try:
        preds["top_idx"].append(self.entity2id[urllib.quote(ee.lower())])
      except KeyError:
        continue
    return self.postprocess(preds, query)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint,
                                       load_only_bert=False):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    if load_only_bert and ("bert" not in name):
      continue
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def model_fn_builder(bert_config,
                     qa_config,
                     mips_config,
                     init_checkpoint,
                     e2m_checkpoint,
                     m2e_checkpoint,
                     entity_id_checkpoint,
                     entity_mask_checkpoint,
                     use_tpu,
                     use_one_hot_embeddings,
                     create_model_fn,
                     summary_obj=None):
  """Returns `model_fn` closure for TPUEstimator."""
  tf.random.set_random_seed(FLAGS.random_seed)

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    # Initialize sparse tensors.
    with tf.device("/cpu:0"):
      tf_e2m_data, tf_e2m_indices, tf_e2m_rowsplits = (
          search_utils.load_ragged_matrix("ent2ment", e2m_checkpoint))
      with tf.name_scope("RaggedConstruction"):
        e2m_ragged_ind = tf.RaggedTensor.from_row_splits(
            values=tf_e2m_indices,
            row_splits=tf_e2m_rowsplits,
            validate=False)
        e2m_ragged_val = tf.RaggedTensor.from_row_splits(
            values=tf_e2m_data,
            row_splits=tf_e2m_rowsplits,
            validate=False)

    tf_m2e_map = search_utils.load_database(
        "coref", [mips_config.num_mentions], m2e_checkpoint, dtype=tf.int32)
    entity_ids = search_utils.load_database(
        "entity_ids", [qa_config.num_entities, qa_config.max_entity_len],
        entity_id_checkpoint, dtype=tf.int32)
    entity_mask = search_utils.load_database(
        "entity_mask", [qa_config.num_entities, qa_config.max_entity_len],
        entity_mask_checkpoint)

    _, predictions = create_model_fn(
        bert_config=bert_config,
        qa_config=qa_config,
        mips_config=mips_config,
        is_training=is_training,
        features=features,
        ent2ment_ind=e2m_ragged_ind,
        ent2ment_val=e2m_ragged_val,
        ment2ent_map=tf_m2e_map,
        entity_ids=entity_ids,
        entity_mask=entity_mask,
        use_one_hot_embeddings=use_one_hot_embeddings,
        summary_obj=summary_obj)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = get_assignment_map_from_checkpoint(
          tvars, init_checkpoint, load_only_bert=qa_config.load_only_bert)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf_estimator.ModeKeys.PREDICT:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only PREDICT mode is supported: %s" % (mode))

    return output_spec

  return model_fn


def render_entities_html(passage, m_starts, m_ends):
  """Add <b> tags to mark entities in string passage."""
  sorted_i = sorted(enumerate(m_ends), key=lambda x: x[1], reverse=True)
  for ii in sorted_i:
    passage = passage[:m_ends[ii[0]]] + "</b>" + passage[m_ends[ii[0]]:]
    passage = passage[:m_starts[ii[0]]] + "<b>" + passage[m_starts[ii[0]]:]
  return passage


def markup_evidence(evidences, highlight):
  """Make entities bold and highlight evidence sentences."""
  marked_up = []
  tf.logging.info(highlight)
  for title, passage, mentions in evidences:
    my_passage = ""
    mention_offset = None
    for ii, sentence in enumerate(passage):
      if [title, ii] in highlight:
        my_passage += "<mark>"
        mention_offset = ii
      my_passage += sentence
      if [title, ii] in highlight:
        my_passage += "</mark>"
      if ii < len(passage) - 1:
        my_passage += " . "
    starts, ends = [], []
    for mention in mentions:
      if mention_offset is None or mention["sent_id"] < mention_offset:
        starts.append(mention["start"])
        ends.append(mention["start"] + len(mention["text"]))
      elif mention["sent_id"] == mention_offset:
        starts.append(mention["start"] + 6)
        ends.append(mention["start"] + 6 + len(mention["text"]))
      elif mention["sent_id"] > mention_offset:
        starts.append(mention["start"] + 13)
        ends.append(mention["start"] + 13 + len(mention["text"]))
    marked_up.append([title, render_entities_html(my_passage, starts, ends)])
  return marked_up


class MainHandler(tornado.web.RequestHandler):
  """Main handler."""

  def initialize(self, env, predictor):
    self._tmpl = env.get_template("drkit.html")
    self._predict_fn = predictor.run

  def get(self):
    question = self.get_argument("question", default="")
    answer = ""
    evidence0_title = ""
    evidence0_context = ""
    evidence1_title = ""
    evidence1_context = ""
    evidence2_title = ""
    evidence2_context = ""
    evidence3_title = ""
    evidence3_context = ""
    evidence4_title = ""
    evidence4_context = ""
    if question:
      answer, evidences, highlight = self._predict_fn(question)
      if answer:
        evidences = markup_evidence(evidences, highlight)
        tf.logging.info("=" * 80)
        tf.logging.info(question)
        tf.logging.info(answer)
        tf.logging.info(" ".join(evidences[0]))
        tf.logging.info(" ".join(evidences[1]))
        tf.logging.info(highlight)
        tf.logging.info("=" * 80)
        evidence0_title = evidences[0][0]
        evidence0_context = evidences[0][1]
        evidence1_title = evidences[1][0]
        evidence1_context = evidences[1][1]
        evidence2_title = evidences[2][0]
        evidence2_context = evidences[2][1]
        evidence3_title = evidences[3][0]
        evidence3_context = evidences[3][1]
        evidence4_title = evidences[4][0]
        evidence4_context = evidences[4][1]
    self.write(
        self._tmpl.render(
            question=question,
            answer=answer,
            evidence0_title=evidence0_title,
            evidence0_context=evidence0_context,
            evidence1_title=evidence1_title,
            evidence1_context=evidence1_context,
            evidence2_title=evidence2_title,
            evidence2_context=evidence2_context,
            evidence3_title=evidence3_title,
            evidence3_context=evidence3_context,
            evidence4_title=evidence4_title,
            evidence4_context=evidence4_context))


def get_predict_fn():
  """Setup and initialize function to get answers."""

  if FLAGS.model_type == "onehop":
    create_model_fn = model_fns.create_onehop_model
  elif FLAGS.model_type == "twohop":
    create_model_fn = model_fns.create_twohop_model
  elif FLAGS.model_type == "twohop-cascaded":
    create_model_fn = model_fns.create_twohopcascade_model
  elif FLAGS.model_type == "threehop":
    create_model_fn = functools.partial(
        model_fns.create_twohop_model, num_hops=3)
  elif FLAGS.model_type == "threehop-cascaded":
    create_model_fn = functools.partial(
        model_fns.create_twohopcascade_model, num_hops=3)
  elif FLAGS.model_type == "wikimovie":
    create_model_fn = model_fns.create_wikimovie_model
  elif FLAGS.model_type == "wikimovie-2hop":
    create_model_fn = functools.partial(
        model_fns.create_wikimovie_model, num_hops=2)
  elif FLAGS.model_type == "wikimovie-3hop":
    create_model_fn = functools.partial(
        model_fns.create_wikimovie_model, num_hops=3)
  elif FLAGS.model_type == "hotpotqa":
    create_model_fn = functools.partial(
        model_fns.create_hotpotqa_model, num_hops=FLAGS.num_hops)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # Load mention and entity files.
  mention2text = json.load(tf.gfile.Open(os.path.join(FLAGS.train_data_dir,
                                                      "mention2text.json")))
  tf.logging.info("Loading metadata about entities and mentions...")
  entity2id, entity2name = json.load(tf.gfile.Open(os.path.join(
      FLAGS.train_data_dir, "entities.json")))
  entityid2name = {str(i): entity2name[e] for e, i in entity2id.items()}

  # Load raw paragraphs.
  tf.logging.info("Loading raw passages...")
  all_passages = {}
  with tf.gfile.Open(FLAGS.raw_passages) as f:
    for ii, line in enumerate(f):
      if ii % 100000 == 0:
        tf.logging.info("Loaded %d", ii)
      item = json.loads(line.strip())
      all_passages[item["kb_id"].lower()] = (
          item["title"], item["context"], item["mentions"])

  qa_config = QAConfig(
      qry_layers_to_use=FLAGS.qry_layers_to_use,
      qry_aggregation_fn=FLAGS.qry_aggregation_fn,
      dropout=FLAGS.question_dropout,
      qry_num_layers=FLAGS.question_num_layers,
      projection_dim=FLAGS.projection_dim,
      load_only_bert=FLAGS.load_only_bert,
      num_entities=len(entity2id),
      max_entity_len=FLAGS.max_entity_len,
      ensure_answer_sparse=FLAGS.ensure_answer_sparse,
      ensure_answer_dense=FLAGS.ensure_answer_dense,
      train_with_sparse=FLAGS.train_with_sparse,
      predict_with_sparse=FLAGS.predict_with_sparse,
      fix_sparse_to_one=FLAGS.fix_sparse_to_one,
      supervision="entity",
      l2_normalize_db=FLAGS.l2_normalize_db,
      entity_score_aggregation_fn=FLAGS.entity_score_aggregation_fn,
      entity_score_threshold=FLAGS.entity_score_threshold,
      softmax_temperature=FLAGS.softmax_temperature,
      sparse_reduce_fn=FLAGS.sparse_reduce_fn,
      intermediate_loss=False,
      light=FLAGS.light,
      sparse_strategy=FLAGS.sparse_strategy,
      train_batch_size=1,
      predict_batch_size=1)

  mips_config = MipsConfig(
      ckpt_path=os.path.join(FLAGS.train_data_dir, "mention_feats"),
      ckpt_var_name="db_emb",
      num_mentions=len(mention2text),
      emb_size=FLAGS.projection_dim * 2,
      num_neighbors=FLAGS.num_mips_neighbors)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=1000,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=1000,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      session_config=tf.ConfigProto(log_device_placement=False))

  if tf.gfile.Exists(os.path.join(FLAGS.init_checkpoint, "best_model.index")):
    bert_ckpt = os.path.join(FLAGS.init_checkpoint, "best_model")
  else:
    bert_ckpt = tf.train.latest_checkpoint(FLAGS.init_checkpoint)
  tf.logging.info("Initializing model with %s.data-00000-of-00001", bert_ckpt)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      qa_config=qa_config,
      mips_config=mips_config,
      init_checkpoint=bert_ckpt,
      e2m_checkpoint=os.path.join(FLAGS.train_data_dir, "ent2ment.npz"),
      m2e_checkpoint=os.path.join(FLAGS.train_data_dir, "coref.npz"),
      entity_id_checkpoint=os.path.join(FLAGS.train_data_dir, "entity_ids"),
      entity_mask_checkpoint=os.path.join(FLAGS.train_data_dir, "entity_mask"),
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      create_model_fn=create_model_fn,
      summary_obj=None)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=1,
      predict_batch_size=1)

  # Answer predictor.
  if tf.gfile.Exists(os.path.join(FLAGS.hotpot_init_checkpoint,
                                  "best_model.index")):
    bert_ckpt = os.path.join(FLAGS.hotpot_init_checkpoint, "best_model")
  else:
    bert_ckpt = tf.train.latest_checkpoint(FLAGS.hotpot_init_checkpoint)
  tf.logging.info("Initializing answer model with %s.data-00000-of-00001",
                  bert_ckpt)
  answer_predictor = BERTPredictor(tokenizer, bert_ckpt)
  predictor = Predictor(tokenizer, estimator, entity2id, entityid2name,
                        all_passages, answer_predictor, FLAGS.train_data_dir,
                        FLAGS.num_entities_linked)
  return predictor


def predict_and_save(predictor, in_file, out_file):
  """Run predictions on question in in_file."""
  with tf.gfile.Open(in_file) as f:
    data = json.load(f)

  answers = {"answer": {}, "sp": {}}
  for item in data:
    answer, _, sp = predictor.run(item["question"])
    answers["answer"][item["_id"]] = answer if answer else ""
    answers["sp"][item["_id"]] = sp if sp else []

  with tf.gfile.Open(out_file, "w") as f:
    json.dump(answers, f)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  predictor = get_predict_fn()

  if FLAGS.mode == "demo":
    predictor.run("Who voices the dog in Family Guy?")
    web_path = FLAGS.web_path
    tmpl_path = web_path + "/templates"
    static_path = web_path + "/static"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(tmpl_path))
    application = tornado.web.Application([
        (r"/", MainHandler, {
            "env": env,
            "predictor": predictor,
        }),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {
            "path": static_path
        }),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
    tf.logging.info("READY!")

  elif FLAGS.mode == "eval":
    predict_and_save(predictor,
                     FLAGS.predict_file,
                     FLAGS.answer_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
