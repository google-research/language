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
# coding=utf-8
"""Run BERT on SQuAD under Phrase-Indexed QA setting."""

import collections
import copy
import json
import math
import os
import random
import re
import time

from absl import flags
from bert import modeling
from bert import optimization
from bert import tokenization
from language.labs.drkit import evaluate
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import layers as contrib_layers
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

## Other parameters
flags.DEFINE_string("train_file", None, "JSON for training.")

flags.DEFINE_string("train_tfrecord_file", None,
                    "Optionally provide a processed training TFRecords file.")

flags.DEFINE_string("predict_file", None, "JSON for predictions.")

flags.DEFINE_string("test_file", None, "JSON for testing.")

flags.DEFINE_string(
    "doc_embed_file", None,
    "JSON file containing documents to embed. Only used if do_doc_embed==True.")

flags.DEFINE_string(
    "qry_embed_file", None,
    "JSON file containing queries to embed. Only used if do_qry_embed==True.")

flags.DEFINE_string(
    "doc_embed_output_dir", None,
    "The output directory where the document features will be written.")

flags.DEFINE_string(
    "qry_embed_output_dir", None,
    "The output directory where the query features will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 192,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 64,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 20,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("max_entity_length", 15,
                     "The maximum number of tokens in the question entity.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", False, "Whether to run eval on the test set.")

flags.DEFINE_bool("do_embed_docs", False,
                  "Whether to compute document features.")

flags.DEFINE_bool("do_embed_qrys", False, "Whether to compute query features.")

flags.DEFINE_bool("filter_tokens_to_keep", False,
                  "If true, only saves embeddings of mention tokens.")

flags.DEFINE_float(
    "subject_mention_probability", 0.0,
    "Fraction of training instances for which we use subject "
    "mentions in the text as opposed to canonical names.")

flags.DEFINE_float("ent_decomp_weight", 1.0,
                   "Weight multiplier for entity selection loss.")

flags.DEFINE_float("rel_decomp_weight", 1.0,
                   "Weight multiplier for relation extraction loss.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 16,
                     "Total batch size for predictions.")

flags.DEFINE_float(
    "normalize_emb", None, "Fixed norm to normalize document embeddings to. "
    "If None or 0.0, no normalization is done.")

flags.DEFINE_float("qry_reconstruction_weight", 0.0,
                   "Weight multiplier for the query BOW reconstruction loss.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 10,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_string(
    "doc_layers_to_use", "-1",
    "Comma-separated list of layer representations to use as the fixed "
    "document representation.")

flags.DEFINE_string(
    "doc_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`doc_layers`.")

flags.DEFINE_string(
    "qry_layers_to_use", "-1",
    "Comma-separated list of layer representations to use as the fixed "
    "query representation.")

flags.DEFINE_string(
    "qry_aggregation_fn", "concat",
    "Aggregation method for combining the outputs of layers specified using "
    "`qry_layers`.")

flags.DEFINE_float("question_dropout", 0.2,
                   "Dropout probability for question BiLSTMs.")

flags.DEFINE_integer("question_num_layers", 5,
                     "Number of layers for question BiLSTMs.")

flags.DEFINE_integer(
    "projection_dim", None, "Number of dimensions to project embeddings to. "
    "Set to None to use full dimensions.")

flags.DEFINE_boolean("include_entity_in_question", True,
                     "Whether to include entity in question text.")

flags.DEFINE_boolean("shared_bert_for_qry", False,
                     "Whether to share BERT between doc and qry.")

flags.DEFINE_boolean("train_bert", True,
                     "Whether to train document encoder or not.")

flags.DEFINE_boolean("load_only_bert", False,
                     "To load only BERT variables from init_checkpoint.")

flags.DEFINE_boolean(
    "use_best_ckpt_for_predict", False,
    "If True, loads the best_model checkpoint in model_dir, "
    "instead of the latest one.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("num_eval_examples", None,
                     "Number of evaluation examples.")

flags.DEFINE_integer("random_seed", 1, "Random seed for reproducibility.")

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

flags.DEFINE_bool(
    "version_2_with_negative", True,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class BertModel(modeling.BertModel):
  """See modeling.BertModel."""

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None,
               reuse=False):
    """Constructor for BertModel which adds an option to reuse variables.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".
      reuse: (optional) if True, reuse previously initialized variables.

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = modeling.get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert", reuse=reuse):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.embedding_table) = (
            modeling.embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings))

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = modeling.embedding_postprocessor(
            input_tensor=self.word_embedding_output,
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
            input_ids, input_mask)

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


class SquadExample:
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               entity_text,
               doc_tokens,
               indices_to_keep=None,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               start_relation=None,
               end_relation=None,
               orig_relation_text=None,
               is_impossible=False,
               has_entity=False,
               has_relation=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.entity_text = entity_text
    self.doc_tokens = doc_tokens
    self.indices_to_keep = indices_to_keep if indices_to_keep else []
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.start_relation = start_relation
    self.end_relation = end_relation
    self.orig_relation_text = orig_relation_text
    self.is_impossible = is_impossible
    self.has_entity = has_entity
    self.has_relation = has_relation

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position,)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position,)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible,)
    return s


class InputFeatures:
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               qas_id,
               example_index,
               doc_span_index,
               doc_tokens,
               doc_token_index_to_keep,
               doc_token_to_orig_map,
               doc_token_is_max_context,
               doc_input_ids,
               doc_input_mask,
               doc_segment_ids,
               qry_tokens,
               qry_input_ids,
               qry_input_mask,
               qry_segment_ids,
               ent_tokens,
               ent_input_ids,
               ent_input_mask,
               start_position=None,
               end_position=None,
               start_relation=None,
               end_relation=None,
               is_impossible=None,
               has_entity=None,
               has_relation=None):
    self.unique_id = unique_id
    self.qas_id = qas_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.doc_tokens = doc_tokens
    self.doc_token_index_to_keep = doc_token_index_to_keep
    self.doc_token_to_orig_map = doc_token_to_orig_map
    self.doc_token_is_max_context = doc_token_is_max_context
    self.doc_input_ids = doc_input_ids
    self.doc_input_mask = doc_input_mask
    self.doc_segment_ids = doc_segment_ids
    self.qry_tokens = qry_tokens
    self.qry_input_ids = qry_input_ids
    self.qry_input_mask = qry_input_mask
    self.qry_segment_ids = qry_segment_ids
    self.ent_tokens = ent_tokens
    self.ent_input_ids = ent_input_ids
    self.ent_input_mask = ent_input_mask
    self.start_position = start_position
    self.end_position = end_position
    self.start_relation = start_relation
    self.end_relation = end_relation
    self.is_impossible = is_impossible
    self.has_entity = has_entity
    self.has_relation = has_relation


class QAConfig:
  """Hyperparameters for the QA model."""

  def __init__(self, doc_layers_to_use, doc_aggregation_fn, qry_layers_to_use,
               qry_aggregation_fn, dropout, qry_num_layers, projection_dim,
               normalize_emb, reconstruction_weight, ent_decomp_weight,
               rel_decomp_weight, train_bert, shared_bert_for_qry,
               load_only_bert):
    self.doc_layers_to_use = [int(vv) for vv in doc_layers_to_use.split(",")]
    self.doc_aggregation_fn = doc_aggregation_fn
    self.qry_layers_to_use = [int(vv) for vv in qry_layers_to_use.split(",")]
    self.qry_aggregation_fn = qry_aggregation_fn
    self.dropout = dropout
    self.qry_num_layers = qry_num_layers
    self.projection_dim = projection_dim
    self.normalize_emb = normalize_emb
    self.reconstruction_weight = reconstruction_weight
    self.ent_decomp_weight = ent_decomp_weight
    self.rel_decomp_weight = rel_decomp_weight
    self.train_bert = train_bert
    self.shared_bert_for_qry = shared_bert_for_qry
    self.load_only_bert = load_only_bert


def read_squad_examples(input_file,
                        is_training,
                        include_entity_in_question=True,
                        p=1.0):
  """Read a SQuAD json file into a list of SquadExample."""

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  with tf.gfile.Open(input_file, "r") as reader:
    examples = []
    for line in tqdm(reader):
      item = json.loads(line.strip())
      paragraph_text = item["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      qas_id = item["id"]
      if item["subject"]["name"] is None or (is_training and
                                             random.uniform(0., 1.) < p):
        entity_text = random.choice(item["subject"]["mentions"])["text"]
      else:
        entity_text = item["subject"]["name"]
      if include_entity_in_question:
        question_text = (
            entity_text + " . " + random.choice(item["relation"]["text"]))
      else:
        question_text = random.choice(item["relation"]["text"])

      start_position = None
      end_position = None
      start_relation = None
      end_relation = None
      orig_answer_text = None
      orig_relation_text = None
      is_impossible = False
      has_entity = False
      has_relation = False
      if is_training:

        if FLAGS.version_2_with_negative:
          is_impossible = item["is_impossible"]
        if not is_impossible:
          answer = item["object"]["mention"]
          orig_answer_text = answer["text"]
          answer_offset = answer["start"]
          answer_length = len(orig_answer_text)
          start_position = char_to_word_offset[answer_offset]
          end_position = char_to_word_offset[answer_offset + answer_length - 1]
          start_relation = start_position
          end_relation = end_position
          orig_relation_text = answer["text"]
          has_entity = True
          has_relation = True
          # Only add answers where the text can be exactly recovered from the
          # document. If this CAN'T happen it's likely due to weird Unicode
          # stuff so we will just skip the example.
          #
          # Note that this means for training mode, every example is NOT
          # guaranteed to be preserved.
          actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
          cleaned_answer_text = " ".join(
              tokenization.whitespace_tokenize(orig_answer_text))
          if actual_text.find(cleaned_answer_text) == -1:
            tf.logging.warning("Example %d", len(examples))
            tf.logging.warning(json.dumps(item, indent=2))
            tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                               actual_text, cleaned_answer_text)
            continue
        else:
          start_position = -1
          end_position = -1
          orig_answer_text = ""
          start_relation = -1
          end_relation = -1
          if item["context_type"] == "entity negative":
            has_entity = True
          elif item["context_type"] == "relation negative":
            answer = item["object"]["mention"]
            orig_relation_text = answer["text"]
            answer_offset = answer["start"]
            answer_length = len(orig_answer_text)
            start_relation = char_to_word_offset[answer_offset]
            end_relation = char_to_word_offset[answer_offset + answer_length -
                                               1]
            has_relation = True

      example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          entity_text=entity_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          start_relation=start_relation,
          end_relation=end_relation,
          orig_relation_text=orig_relation_text,
          is_impossible=is_impossible,
          has_entity=has_entity,
          has_relation=has_relation)
      examples.append(example)

  return examples


def read_docs_to_embed(input_file):
  """Read json file containing Wikipedia articles into list of SquadExample."""

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  with tf.gfile.Open(input_file, "r") as reader:
    examples = []
    for line in tqdm(reader):
      item = json.loads(line.strip())
      paragraph_text = item["context"]
      doc_tokens = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False

      qas_id = item["id"]
      question_text = ""
      entity_text = ""
      is_impossible = True
      start_position = -1
      end_position = -1
      orig_answer_text = ""

      indices_to_keep = []
      if "mention_tokens" in item:
        indices_to_keep = item["mention_tokens"]

      example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          entity_text=entity_text,
          doc_tokens=doc_tokens,
          indices_to_keep=indices_to_keep,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
      examples.append(example)

  return examples


def read_qrys_to_embed(input_file):
  """Read json file containing Wikidata queries into list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    examples = []
    for line in tqdm(reader):
      item = json.loads(line.strip())
      qas_id = item["id"]
      if "name" in item["subject"]:
        entity_text = item["subject"]["name"]
      else:
        entity_text = random.choice(item["subject"]["mentions"])["text"]
      question_text = (
          entity_text + " . " + random.choice(item["relation"]["text"]))

      is_impossible = True
      start_position = -1
      end_position = -1
      orig_answer_text = ""

      example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          entity_text=entity_text,
          doc_tokens=["dummy"],
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
      examples.append(example)

  return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_doc_length,
                                 doc_stride,
                                 max_query_length,
                                 max_entity_length,
                                 is_training,
                                 output_fn,
                                 keep_all_indices=True):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in tqdm(enumerate(examples)):
    query_tokens = tokenizer.tokenize(example.question_text)
    entity_tokens = tokenizer.tokenize(example.entity_text)

    if len(query_tokens) > max_query_length - 2:
      query_tokens = query_tokens[0:max_query_length - 2]  # -2 for [CLS], [SEP]
    if len(entity_tokens) > max_entity_length:
      entity_tokens = entity_tokens[0:max_entity_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_index_to_keep = set()
    for i in example.indices_to_keep:
      i_st = orig_to_tok_index[i]
      if i < len(orig_to_tok_index) - 1:
        i_en = orig_to_tok_index[i + 1] - 1
      else:
        i_en = len(all_doc_tokens) - 1
      tok_index_to_keep.update([i_st, i_en])

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    tok_start_relation = None
    tok_end_relation = None
    if is_training and not example.has_relation:
      tok_start_relation = -1
      tok_end_relation = -1
    if is_training and example.has_relation:
      tok_start_relation = orig_to_tok_index[example.start_relation]
      if example.end_relation < len(example.doc_tokens) - 1:
        tok_end_relation = orig_to_tok_index[example.end_relation + 1] - 1
      else:
        tok_end_relation = len(all_doc_tokens) - 1
      (tok_start_relation, tok_end_relation) = _improve_answer_span(
          all_doc_tokens, tok_start_relation, tok_end_relation, tokenizer,
          example.orig_relation_text)

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_doc_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      doc_tokens, qry_tokens = [], []
      doc_token_index_to_keep = []
      doc_token_to_orig_map = {}
      doc_token_is_max_context = {}
      doc_segment_ids, qry_segment_ids = [], []

      # Question
      qry_tokens.append("[CLS]")
      qry_segment_ids.append(0)
      for token in query_tokens:
        qry_tokens.append(token)
        qry_segment_ids.append(0)
      qry_tokens.append("[SEP]")
      qry_segment_ids.append(0)

      # Document
      doc_tokens.append("[CLS]")
      doc_segment_ids.append(1)
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        doc_token_to_orig_map[len(
            doc_tokens)] = tok_to_orig_index[split_token_index]

        if keep_all_indices or split_token_index in tok_index_to_keep:
          # +1 below for [CLS]
          doc_token_index_to_keep.append(i + 1)

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        doc_token_is_max_context[len(doc_tokens)] = is_max_context
        doc_tokens.append(all_doc_tokens[split_token_index])
        doc_segment_ids.append(1)
      doc_tokens.append("[SEP]")
      doc_segment_ids.append(1)

      doc_input_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
      qry_input_ids = tokenizer.convert_tokens_to_ids(qry_tokens)
      ent_input_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      doc_input_mask = [1] * len(doc_input_ids)
      qry_input_mask = [1] * len(qry_input_ids)
      ent_input_mask = [1] * len(ent_input_ids)

      # Zero-pad up to the sequence length.
      while len(doc_input_ids) < max_doc_length:
        doc_input_ids.append(0)
        doc_input_mask.append(0)
        doc_segment_ids.append(0)
      while len(qry_input_ids) < max_query_length:
        qry_input_ids.append(0)
        qry_input_mask.append(0)
        qry_segment_ids.append(0)
      while len(ent_input_ids) < max_entity_length:
        ent_input_ids.append(0)
        ent_input_mask.append(0)

      assert len(doc_input_ids) == max_doc_length
      assert len(doc_input_mask) == max_doc_length
      assert len(doc_segment_ids) == max_doc_length
      assert len(qry_input_ids) == max_query_length
      assert len(qry_input_mask) == max_query_length
      assert len(qry_segment_ids) == max_query_length
      assert len(ent_input_ids) == max_entity_length
      assert len(ent_input_mask) == max_entity_length

      start_position = None
      end_position = None
      start_relation = None
      end_relation = None
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      doc_offset = 1
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset
      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      if is_training and example.has_relation:
        out_of_span = False
        if not (tok_start_relation >= doc_start and
                tok_end_relation <= doc_end):
          out_of_span = True
        if out_of_span:
          start_relation = 0
          end_relation = 0
        else:
          start_relation = tok_start_relation - doc_start + doc_offset
          end_relation = tok_end_relation - doc_start + doc_offset
      if is_training and not example.has_relation:
        start_relation = 0
        end_relation = 0

      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s", unique_id)
        tf.logging.info("example_index: %s", example_index)
        tf.logging.info("doc_span_index: %s", doc_span_index)
        tf.logging.info(
            "doc_tokens: %s",
            " ".join([tokenization.printable_text(x) for x in doc_tokens]))
        tf.logging.info("doc_token_index_to_keep: %s",
                        " ".join(["%d" % x for x in doc_token_index_to_keep]))
        tf.logging.info(
            "qry_tokens: %s",
            " ".join([tokenization.printable_text(x) for x in qry_tokens]))
        tf.logging.info(
            "ent_tokens: %s",
            " ".join([tokenization.printable_text(x) for x in entity_tokens]))
        tf.logging.info(
            "doc_token_to_orig_map: %s", " ".join(
                ["%d:%d" % (x, y) for (x, y) in doc_token_to_orig_map.items()]))
        tf.logging.info(
            "doc_token_is_max_context: %s", " ".join([
                "%d:%s" % (x, y) for (x, y) in doc_token_is_max_context.items()
            ]))
        tf.logging.info("doc_input_ids: %s",
                        " ".join([str(x) for x in doc_input_ids]))
        tf.logging.info("doc_input_mask: %s",
                        " ".join([str(x) for x in doc_input_mask]))
        tf.logging.info("doc_segment_ids: %s",
                        " ".join([str(x) for x in doc_segment_ids]))
        tf.logging.info("qry_input_ids: %s",
                        " ".join([str(x) for x in qry_input_ids]))
        tf.logging.info("qry_input_mask: %s",
                        " ".join([str(x) for x in qry_input_mask]))
        tf.logging.info("ent_input_ids: %s",
                        " ".join([str(x) for x in ent_input_ids]))
        tf.logging.info("ent_input_mask: %s",
                        " ".join([str(x) for x in ent_input_mask]))
        tf.logging.info("qry_segment_ids: %s",
                        " ".join([str(x) for x in qry_segment_ids]))
        if is_training and example.is_impossible:
          tf.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(doc_tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d", start_position)
          tf.logging.info("end_position: %d", end_position)
          tf.logging.info("answer: %s",
                          tokenization.printable_text(answer_text))
        if is_training and not example.has_entity:
          tf.logging.info("not entity negative")
        if is_training and example.has_entity:
          tf.logging.info("is an entity negative")
        if is_training and not example.has_relation:
          tf.logging.info("not relation negative")
        if is_training and example.has_relation:
          answer_text = " ".join(doc_tokens[start_relation:(end_relation + 1)])
          tf.logging.info("start_relation: %d", start_relation)
          tf.logging.info("end_relation: %d", end_relation)
          tf.logging.info("answer: %s",
                          tokenization.printable_text(answer_text))

      feature = InputFeatures(
          unique_id=unique_id,
          qas_id=example.qas_id.encode("utf-8"),
          example_index=example_index,
          doc_span_index=doc_span_index,
          doc_tokens=doc_tokens,
          doc_token_index_to_keep=doc_token_index_to_keep,
          doc_token_to_orig_map=doc_token_to_orig_map,
          doc_token_is_max_context=doc_token_is_max_context,
          doc_input_ids=doc_input_ids,
          doc_input_mask=doc_input_mask,
          doc_segment_ids=doc_segment_ids,
          qry_tokens=qry_tokens,
          qry_input_ids=qry_input_ids,
          qry_input_mask=qry_input_mask,
          qry_segment_ids=qry_segment_ids,
          ent_tokens=entity_tokens,
          ent_input_ids=ent_input_ids,
          ent_input_mask=ent_input_mask,
          start_position=start_position,
          end_position=end_position,
          start_relation=start_relation,
          end_relation=end_relation,
          is_impossible=example.is_impossible,
          has_entity=example.has_entity,
          has_relation=example.has_relation)

      # Run callback
      output_fn(feature)

      unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def _get_bert_embeddings(model, layers_to_use, aggregation_fn, name="bert"):
  """Extract embeddings from BERT model."""
  all_hidden = model.get_all_encoder_layers()
  layers_hidden = [all_hidden[i] for i in layers_to_use]
  hidden_shapes = [
      modeling.get_shape_list(hid, expected_rank=3) for hid in all_hidden
  ]

  if len(layers_hidden) == 1:
    hidden_emb = layers_hidden[0]
    hidden_size = hidden_shapes[0][2]
  elif aggregation_fn == "concat":
    hidden_emb = tf.concat(layers_hidden, 2)
    hidden_size = sum([hidden_shapes[i][2] for i in layers_to_use])
  elif aggregation_fn == "average":
    hidden_size = hidden_shapes[0][2]
    assert all([shape[2] == hidden_size for shape in hidden_shapes
               ]), hidden_shapes
    hidden_emb = tf.add_n(layers_hidden) / len(layers_hidden)
  elif aggregation_fn == "attention":
    hidden_size = hidden_shapes[0][2]
    mixing_weights = tf.get_variable(
        name + "/mixing/weights", [len(layers_hidden)],
        initializer=tf.zeros_initializer())
    mixing_scores = tf.nn.softmax(mixing_weights)
    hidden_emb = tf.tensordot(
        tf.stack(layers_hidden, axis=-1), mixing_scores, [[-1], [0]])
  else:
    raise ValueError("Unrecognized aggregation function %s." % aggregation_fn)

  return hidden_emb, hidden_size


def create_model(bert_config, qa_config, is_training, doc_input_ids,
                 doc_input_mask, doc_segment_ids, qry_input_ids, qry_input_mask,
                 qry_segment_ids, ent_input_ids, ent_input_mask,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  tf.random.set_random_seed(FLAGS.random_seed)

  # document embedding
  doc_model = BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=doc_input_ids,
      input_mask=doc_input_mask,
      token_type_ids=doc_segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  doc_hidden, _ = _get_bert_embeddings(
      doc_model,
      qa_config.doc_layers_to_use,
      qa_config.doc_aggregation_fn,
      name="doc")
  word_emb_table = doc_model.get_embedding_table()

  # question embedding
  qry_word_emb = tf.nn.embedding_lookup(word_emb_table, qry_input_ids)

  # question BOW embedding
  with tf.variable_scope("qry/bow"):
    word_weights = tf.get_variable(
        "word_weights", [bert_config.vocab_size, 1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    ent_word_emb = tf.nn.embedding_lookup(word_emb_table, ent_input_ids)
    ent_word_weights = tf.nn.embedding_lookup(word_weights, ent_input_ids)
    qry_bow_emb = tf.reduce_sum(
        ent_word_emb * ent_word_weights *
        tf.cast(tf.expand_dims(ent_input_mask, 2), tf.float32),
        axis=1)

  dropout = qa_config.dropout if is_training else 0.0
  attention_mask = modeling.create_attention_mask_from_input_mask(
      qry_input_ids, qry_input_mask)

  # question shared encoder
  if qa_config.shared_bert_for_qry:
    qry_model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=qry_input_ids,
        input_mask=qry_input_mask,
        token_type_ids=qry_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        reuse=True)
    qry_seq_emb, _ = _get_bert_embeddings(
        qry_model,
        qa_config.qry_layers_to_use,
        qa_config.qry_aggregation_fn,
        name="qry")
  else:
    with tf.variable_scope("qry/encoder"):
      qry_seq_emb = modeling.transformer_model(
          input_tensor=qry_word_emb,
          attention_mask=attention_mask,
          hidden_size=bert_config.hidden_size,
          num_hidden_layers=qa_config.qry_num_layers - 1,
          num_attention_heads=bert_config.num_attention_heads,
          intermediate_size=bert_config.intermediate_size,
          intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
          hidden_dropout_prob=dropout,
          attention_probs_dropout_prob=dropout,
          initializer_range=bert_config.initializer_range,
          do_return_all_layers=False)

  # question start
  with tf.variable_scope("qry/start"):
    qry_start_seq_emb = modeling.transformer_model(
        input_tensor=qry_seq_emb,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False)
    qry_start_emb = tf.squeeze(qry_start_seq_emb[:, 0:1, :], axis=1)

  # question end
  with tf.variable_scope("qry/end"):
    qry_end_seq_emb = modeling.transformer_model(
        input_tensor=qry_seq_emb,
        attention_mask=attention_mask,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        initializer_range=bert_config.initializer_range,
        do_return_all_layers=False)
    qry_end_emb = tf.squeeze(qry_end_seq_emb[:, 0:1, :], axis=1)

  na_emb = tf.get_variable(
      "noanswer_emb", [1, bert_config.hidden_size],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer())
  # initializer=tf.zeros_initializer())

  if qa_config.projection_dim is not None:
    with tf.variable_scope("projection"):
      doc_hidden = contrib_layers.fully_connected(
          doc_hidden,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          scope="doc_projection")
      qry_start_emb = contrib_layers.fully_connected(
          qry_start_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          scope="qry_projection")
      qry_end_emb = contrib_layers.fully_connected(
          qry_end_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=True,
          scope="qry_projection")
      na_emb = contrib_layers.fully_connected(
          na_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          reuse=True,
          scope="doc_projection")
      qry_bow_emb = contrib_layers.fully_connected(
          qry_bow_emb,
          qa_config.projection_dim,
          activation_fn=tf.nn.tanh,
          scope="bow_projection")

  if qa_config.normalize_emb is not None and qa_config.normalize_emb > 0.:
    doc_hidden = qa_config.normalize_emb * tf.math.l2_normalize(
        doc_hidden, axis=2)
    na_emb = qa_config.normalize_emb * tf.math.l2_normalize(na_emb, axis=1)

  def _inner_logits(na, emb, qry):
    """Returns logits computed using inner product of qry and embeddings."""
    na_logit = tf.matmul(qry, na, transpose_b=True)
    logits = tf.reduce_sum(tf.expand_dims(qry, 1) * emb[:, 1:, :], 2)
    return tf.concat([na_logit, logits], axis=1)

  ent_logits = _inner_logits(na_emb, doc_hidden, qry_bow_emb)
  rel_start_logits = _inner_logits(na_emb, doc_hidden, qry_start_emb)
  rel_end_logits = _inner_logits(na_emb, doc_hidden, qry_end_emb)
  start_logits = ent_logits + rel_start_logits
  end_logits = ent_logits + rel_end_logits

  return (start_logits, end_logits, ent_logits, rel_start_logits,
          rel_end_logits, doc_hidden, qry_start_emb, qry_end_emb, qry_bow_emb,
          word_emb_table)


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     exclude_bert):
  """Creates an optimizer training op, optionally excluding BERT vars."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = optimization.AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  if exclude_bert:
    bert_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bert")
    tvars = [vv for vv in tvars if vv not in bert_vars]

  tf.logging.info("Training the following variables:")
  for vv in tvars:
    tf.logging.info(vv.name)

  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


def get_assignment_map_from_checkpoint(tvars,
                                       init_checkpoint,
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


def _convert_ids_to_onehot(input_ids, input_mask, vocab_size):
  """Returns a k-hot vector for the given input ids in vocab.

  Args:
    input_ids: (tf.int32) batch_size x seq_length
    input_mask: (tf.int32) batch_size x seq_length
    vocab_size: scalar

  Returns:
    input_onehot: (tf.int32) batch_size x vocab_size
  """
  onehot = tf.one_hot(input_ids, vocab_size, dtype=tf.int32, axis=-1)
  masked_onehot = onehot * tf.expand_dims(input_mask, axis=2)
  input_onehot = tf.minimum(1, tf.reduce_sum(masked_onehot, axis=1))
  return input_onehot


def compute_bow_reconstruction_loss(hidden, word_emb, input_ids, input_mask,
                                    word_emb_size, vocab_size):
  """Creates loss for BOW reconstruction from hidden states.

  Args:
    hidden: (tf.float32) batch_size x dim1
    word_emb: (tf.float32) vocab_size x dim2
    input_ids: (tf.float32) batch_size x seq_length
    input_mask: (tf.int32) batch_size x seq_length
    word_emb_size: scalar
    vocab_size: scalar

  Returns:
    loss: (tf.float32) scalar
  """
  hidden = contrib_layers.fully_connected(
      hidden, word_emb_size, activation_fn=tf.nn.relu)
  input_onehot = _convert_ids_to_onehot(input_ids, input_mask, vocab_size)
  logits = tf.matmul(hidden, word_emb, transpose_b=True)
  bce_losses = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.cast(input_onehot, tf.float32), logits=logits)
  return tf.reduce_mean(bce_losses)


def model_fn_builder(bert_config,
                     qa_config,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     summary_obj=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

    unique_ids = features["unique_ids"]
    doc_input_ids = features["doc_input_ids"]
    doc_input_mask = features["doc_input_mask"]
    doc_segment_ids = features["doc_segment_ids"]
    qry_input_ids = features["qry_input_ids"]
    qry_input_mask = features["qry_input_mask"]
    qry_segment_ids = features["qry_segment_ids"]
    ent_input_ids = features["ent_input_ids"]
    ent_input_mask = features["ent_input_mask"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits, ent_logits, rel_start_logits, rel_end_logits,
     doc_hidden, qry_start, qry_end, qry_bow, word_emb_table) = create_model(
         bert_config=bert_config,
         qa_config=qa_config,
         is_training=is_training,
         doc_input_ids=doc_input_ids,
         doc_input_mask=doc_input_mask,
         doc_segment_ids=doc_segment_ids,
         qry_input_ids=qry_input_ids,
         qry_input_mask=qry_input_mask,
         qry_segment_ids=qry_segment_ids,
         ent_input_ids=ent_input_ids,
         ent_input_mask=ent_input_mask,
         use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = get_assignment_map_from_checkpoint(
           tvars, init_checkpoint, load_only_bert=qa_config.load_only_bert)
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
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(doc_input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      def compute_bce_loss(logits, entity_bool, mask):
        """Compute binary cross-entropy loss for logits."""
        batch_labels = tf.expand_dims(entity_bool, 1)
        position_labels = batch_labels * tf.ones(
            (1, seq_length - 1), dtype=tf.float32)
        # Ignore [CLS] for this loss.
        loss_ = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=position_labels, logits=logits[:, 1:])
        mask_f = tf.cast(mask[:, 1:], tf.float32)
        loss = tf.reduce_sum(loss_ * mask_f) / tf.maximum(
            1e-12, tf.reduce_sum(mask_f))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      has_entity = tf.cast(features["has_entity"], tf.float32)
      total_ent_loss = compute_bce_loss(ent_logits, has_entity, doc_input_mask)

      start_relation = features["start_relation"]
      end_relation = features["end_relation"]
      rel_start_loss = compute_loss(rel_start_logits, start_relation)
      rel_end_loss = compute_loss(rel_end_logits, end_relation)

      start_bow_loss = compute_bow_reconstruction_loss(
          qry_start, word_emb_table, qry_input_ids, qry_input_mask,
          bert_config.hidden_size, bert_config.vocab_size)
      end_bow_loss = compute_bow_reconstruction_loss(qry_end, word_emb_table,
                                                     qry_input_ids,
                                                     qry_input_mask,
                                                     bert_config.hidden_size,
                                                     bert_config.vocab_size)

      total_ans_loss = (start_loss + end_loss) / 2.0
      total_rel_loss = (rel_start_loss + rel_end_loss) / 2.0
      total_span_loss = (
          total_ans_loss + qa_config.ent_decomp_weight * total_ent_loss +
          qa_config.rel_decomp_weight * total_rel_loss)

      total_bow_loss = (start_bow_loss + end_bow_loss) / 2.0
      total_loss = (
          total_span_loss + qa_config.reconstruction_weight * total_bow_loss)

      if summary_obj is not None:
        summary_obj.scalar("Entity Loss", tf.expand_dims(total_ent_loss, 0))
        summary_obj.scalar("Relation Loss", tf.expand_dims(total_rel_loss, 0))
        summary_obj.scalar("Answer Loss", tf.expand_dims(total_ans_loss, 0))
        summary_obj.scalar("Span Loss", tf.expand_dims(total_span_loss, 0))
        summary_obj.scalar("BOW Loss", tf.expand_dims(total_bow_loss, 0))
        summary_obj.scalar("Total Loss", tf.expand_dims(total_loss, 0))

      train_op = create_optimizer(total_loss, learning_rate, num_train_steps,
                                  num_warmup_steps, use_tpu,
                                  not qa_config.train_bert)

      if summary_obj:
        host_call = summary_obj.get_host_call()
      else:
        host_call = None
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          host_call=host_call)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "start_rel_logits": rel_start_logits,
          "end_rel_logits": rel_end_logits,
          "ent_logits": ent_logits,
          "doc_features": doc_hidden,
          "qry_st_features": qry_start,
          "qry_en_features": qry_end,
          "qry_bow_features": qry_bow,
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, qry_length, ent_length,
                     is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "doc_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "doc_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "doc_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "qry_input_ids": tf.FixedLenFeature([qry_length], tf.int64),
      "qry_input_mask": tf.FixedLenFeature([qry_length], tf.int64),
      "qry_segment_ids": tf.FixedLenFeature([qry_length], tf.int64),
      "ent_input_ids": tf.FixedLenFeature([ent_length], tf.int64),
      "ent_input_mask": tf.FixedLenFeature([ent_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["start_relation"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_relation"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["has_entity"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult", [
    "unique_id", "start_logits", "end_logits", "start_rel_logits",
    "end_rel_logits", "ent_logits"
])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s", output_prediction_file)
  tf.logging.info("Writing nbest to: %s", output_nbest_file)

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction", [
          "feature_index", "start_index", "end_index", "start_logit",
          "end_logit", "start_rel_logit", "end_rel_logit", "start_ent_logit",
          "end_ent_logit"
      ])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
          null_start_rel_logit = result.start_rel_logits[0]
          null_end_rel_logit = result.end_rel_logits[0]
          null_ent_logit = result.ent_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.doc_tokens):
            continue
          if end_index >= len(feature.doc_tokens):
            continue
          if start_index not in feature.doc_token_to_orig_map:
            continue
          if end_index not in feature.doc_token_to_orig_map:
            continue
          if not feature.doc_token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index],
                  start_rel_logit=result.start_rel_logits[start_index],
                  end_rel_logit=result.end_rel_logits[end_index],
                  start_ent_logit=result.ent_logits[start_index],
                  end_ent_logit=result.ent_logits[end_index],
              ))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit,
              start_rel_logit=null_start_rel_logit,
              end_rel_logit=null_end_rel_logit,
              start_ent_logit=null_ent_logit,
              end_ent_logit=null_ent_logit,
          ))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", [
            "text", "start_logit", "end_logit", "start_rel_logit",
            "end_rel_logit", "start_ent_logit", "end_ent_logit"
        ])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.doc_tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.doc_token_to_orig_map[pred.start_index]
        orig_doc_end = feature.doc_token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              start_rel_logit=pred.start_rel_logit,
              end_rel_logit=pred.end_rel_logit,
              start_ent_logit=pred.start_ent_logit,
              end_ent_logit=pred.end_ent_logit,
          ))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(
              text="empty",
              start_logit=0.0,
              end_logit=0.0,
              start_rel_logit=0.0,
              end_rel_logit=0.0,
              start_ent_logit=0.0,
              end_ent_logit=0.0))
    assert len(nbest) >= 1

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="",
                start_logit=null_start_logit,
                end_logit=null_end_logit,
                start_rel_logit=null_start_rel_logit,
                end_rel_logit=null_end_rel_logit,
                start_ent_logit=null_ent_logit,
                end_ent_logit=null_ent_logit))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    if not best_non_null_entry:
      best_non_null_entry = _NbestPrediction(
          text="empty",
          start_logit=0.0,
          end_logit=0.0,
          start_rel_logit=0.0,
          end_rel_logit=0.0,
          start_ent_logit=0.0,
          end_ent_logit=0.0)

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_rel_logit"] = entry.start_rel_logit
      output["end_rel_logit"] = entry.end_rel_logit
      output["start_ent_logit"] = entry.start_ent_logit
      output["end_ent_logit"] = entry.end_ent_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info("Unable to find text: '%s' in '%s'", pred_text, orig_text)
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter:
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["qas_id"] = create_bytes_feature(feature.qas_id)
    features["doc_input_ids"] = create_int_feature(feature.doc_input_ids)
    features["doc_input_mask"] = create_int_feature(feature.doc_input_mask)
    features["doc_segment_ids"] = create_int_feature(feature.doc_segment_ids)
    features["doc_token_index_to_keep"] = create_int_feature(
        feature.doc_token_index_to_keep)
    features["qry_input_ids"] = create_int_feature(feature.qry_input_ids)
    features["qry_input_mask"] = create_int_feature(feature.qry_input_mask)
    features["qry_segment_ids"] = create_int_feature(feature.qry_segment_ids)
    features["ent_input_ids"] = create_int_feature(feature.ent_input_ids)
    features["ent_input_mask"] = create_int_feature(feature.ent_input_mask)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["start_relation"] = create_int_feature([feature.start_relation])
      features["end_relation"] = create_int_feature([feature.end_relation])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])
      has_entity = 0
      if feature.has_entity:
        has_entity = 1
      features["has_entity"] = create_int_feature([has_entity])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def train(tfrecord_filename, estimator, num_train_steps):
  """Run one training loop over given TFRecords file."""
  train_input_fn = input_fn_builder(
      input_file=tfrecord_filename,
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      ent_length=FLAGS.max_entity_length,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


def single_eval(eval_tfrecord_filename, estimator, eval_examples, eval_features,
                ckpt_path, output_prediction_file, output_nbest_file,
                output_null_log_odds_file):
  """Run one evaluation using given checkpoint."""
  tf.logging.info("***** Running predictions using %s *****", ckpt_path)
  tf.logging.info("  Num orig examples = %d", len(eval_examples))
  tf.logging.info("  Num split examples = %d", len(eval_features))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  all_results = []

  predict_input_fn = input_fn_builder(
      input_file=eval_tfrecord_filename,
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      ent_length=FLAGS.max_entity_length,
      is_training=False,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_results = []
  for result in estimator.predict(
      predict_input_fn, yield_single_examples=True, checkpoint_path=ckpt_path):
    if len(all_results) % 1000 == 0:
      tf.logging.info("Processing example: %d", len(all_results))
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    start_rel_logits = [float(x) for x in result["start_rel_logits"].flat]
    end_rel_logits = [float(x) for x in result["end_rel_logits"].flat]
    ent_logits = [float(x) for x in result["ent_logits"].flat]
    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits,
            start_rel_logits=start_rel_logits,
            end_rel_logits=end_rel_logits,
            ent_logits=ent_logits))

  write_predictions(eval_examples, eval_features, all_results,
                    FLAGS.n_best_size, FLAGS.max_answer_length,
                    FLAGS.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file)


def _copy_model(in_path, out_path):
  """Copy model checkpoint for future use."""
  tf.logging.info("Copying checkpoint from %s to %s.", in_path, out_path)
  tf.gfile.Copy(
      in_path + ".data-00000-of-00001",
      out_path + ".data-00000-of-00001",
      overwrite=True)
  tf.gfile.Copy(in_path + ".index", out_path + ".index", overwrite=True)
  tf.gfile.Copy(in_path + ".meta", out_path + ".meta", overwrite=True)


def continuous_eval(eval_tfrecord_filename, estimator, eval_examples,
                    eval_features):
  """Run continuous evaluation on given TFRecords file."""
  current_ckpt = None
  stop_evaluating = False
  num_waits = 0
  best_f = 0.
  f_log = tf.gfile.Open(os.path.join(FLAGS.output_dir, "scores.log"), "w")
  if not tf.gfile.Exists(os.path.join(FLAGS.output_dir, "eval")):
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, "eval"))
  event_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_dir, "eval"))
  while not stop_evaluating:
    if FLAGS.use_best_ckpt_for_predict:
      ckpt_path = os.path.join(FLAGS.output_dir, "best_model")
      if not os.path.exists(ckpt_path + ".meta"):
        tf.logging.info("No best_model checkpoint found in %s",
                        FLAGS.output_dir)
        tf.logging.info("Skipping evaluation.")
        break
      output_prediction_file = os.path.join(FLAGS.output_dir,
                                            "predictions.json")
      output_nbest_file = os.path.join(FLAGS.output_dir,
                                       "nbest_predictions.json")
      output_null_log_odds_file = os.path.join(FLAGS.output_dir,
                                               "null_odds.json")
      stop_evaluating = True
    else:
      ckpt_path = tf.train.latest_checkpoint(FLAGS.output_dir)
      if ckpt_path is None:
        tf.logging.info("No checkpoint in %s", FLAGS.output_dir)
        tf.logging.info("Waiting for 100s")
        time.sleep(100)
        continue
      if ckpt_path == current_ckpt:
        tf.logging.info("No new checkpoint in %s", FLAGS.output_dir)
        num_waits += 1
        if num_waits == 15:
          tf.logging.info("Waited for 1000s, exiting now.")
          # stop_evaluating = True
        else:
          tf.logging.info("Waiting for 100s")
          time.sleep(100)
        continue
      num_waits = 0
      current_ckpt = ckpt_path
      model_name = os.path.basename(ckpt_path)
      output_prediction_file = os.path.join(FLAGS.output_dir,
                                            "predictions_%s.json" % model_name)
      output_nbest_file = os.path.join(FLAGS.output_dir,
                                       "nbest_predictions_%s.json" % model_name)
      output_null_log_odds_file = os.path.join(FLAGS.output_dir,
                                               "null_odds_%s.json" % model_name)
      output_relationwise_file = os.path.join(
          FLAGS.output_dir, "relationwise_%s.json" % model_name)

    single_eval(eval_tfrecord_filename, estimator, eval_examples, eval_features,
                ckpt_path, output_prediction_file, output_nbest_file,
                output_null_log_odds_file)

    micro, macro, relationwise, _ = evaluate.compute_scores(
        FLAGS.predict_file, output_prediction_file)
    message = (
        "Model %s Micro-P %.3f Micro-R %.3f Micro-F %.3f "
        "Macro-P %.3f Macro-R %.3f Macro-F %.3f" %
        (ckpt_path, micro[0], micro[1], micro[2], macro[0], macro[1], macro[2]))
    tf.logging.info(message)
    f_log.write(message + "\n")
    f_log.flush()
    ckpt_number = int(ckpt_path.rsplit("-", 1)[1])
    micro_f_summary = tf.Summary(value=[
        tf.Summary.Value(tag="micro_f1", simple_value=micro[2]),
    ])
    event_writer.add_summary(micro_f_summary, global_step=ckpt_number)
    macro_f_summary = tf.Summary(value=[
        tf.Summary.Value(tag="macro_f1", simple_value=macro[2]),
    ])
    event_writer.add_summary(macro_f_summary, global_step=ckpt_number)
    for rel, (metrics, _) in relationwise.items():
      rel_summary = tf.Summary(value=[
          tf.Summary.Value(tag="relation/" + rel, simple_value=metrics[2]),
      ])
      event_writer.add_summary(rel_summary, global_step=ckpt_number)
    json.dump(
        {key: val[0].tolist() + [val[1]] for key, val in relationwise.items()},
        tf.gfile.Open(output_relationwise_file, "w"))
    if micro[2] > best_f and ckpt_path is not None:
      best_f = micro[2]
      _copy_model(ckpt_path, os.path.join(FLAGS.output_dir, "best_model"))
  f_log.close()


def embed_docs(docs_tfrecord_filename, estimator, ckpt_path, tok_index_to_keep,
               num_tokens, dimension):
  """Compute document features for given TFRecords."""
  tf.logging.info("***** Embedding documents using %s *****", ckpt_path)

  predict_input_fn = input_fn_builder(
      input_file=docs_tfrecord_filename,
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      ent_length=FLAGS.max_entity_length,
      is_training=False,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_features = np.empty((num_tokens, dimension), dtype=np.float32)
  doc_spans = []
  batch_start, num_features = 0, 0
  for result in estimator.predict(
      predict_input_fn, yield_single_examples=False, checkpoint_path=ckpt_path):
    if batch_start % 1000 == 0:
      tf.logging.info("Processing batch: %d", batch_start)
    for ii in range(result["doc_features"].shape[0]):
      doc_idx = batch_start + ii
      st = num_features
      for ti in tok_index_to_keep[doc_idx]:
        all_features[num_features, :] = result["doc_features"][ii, ti, :]
        num_features += 1
      doc_spans.append([st, num_features])
    batch_start += result["doc_features"].shape[0]

  return all_features, np.stack(doc_spans, axis=0)


def embed_qrys(qrys_tfrecord_filename, estimator, ckpt_path):
  """Compute query features for given TFRecords."""
  tf.logging.info("***** Embedding queries using %s *****", ckpt_path)

  predict_input_fn = input_fn_builder(
      input_file=qrys_tfrecord_filename,
      seq_length=FLAGS.max_seq_length,
      qry_length=FLAGS.max_query_length,
      ent_length=FLAGS.max_entity_length,
      is_training=False,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_starts, all_ends, all_bows = [], [], []
  for result in estimator.predict(
      predict_input_fn, yield_single_examples=False, checkpoint_path=ckpt_path):
    if len(all_starts) % 1000 == 0:
      tf.logging.info("Processing example: %d", len(all_starts))
    all_starts += [result["qry_st_features"]]
    all_ends += [result["qry_en_features"]]
    all_bows += [result["qry_bow_features"]]

  return (np.concatenate(all_starts, axis=0), np.concatenate(all_ends, axis=0),
          np.concatenate(all_bows, axis=0))


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if (not FLAGS.do_train and not FLAGS.do_predict and
      not FLAGS.do_embed_docs and not FLAGS.do_embed_qrys and
      not FLAGS.do_test):
    raise ValueError("At least one of `do_train` or `do_predict` or "
                     "`do_embed_docs` or `do_embed_qrys` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")
  if FLAGS.do_embed_docs:
    if not FLAGS.doc_embed_file:
      raise ValueError(
          "If `do_embed_docs` is True, then `doc_embed_file` must be "
          "specified.")
  if FLAGS.do_embed_qrys:
    if not FLAGS.qry_embed_file:
      raise ValueError(
          "If `do_embed_qrys` is True, then `qry_embed_file` must be "
          "specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  qa_config = QAConfig(
      doc_layers_to_use=FLAGS.doc_layers_to_use,
      doc_aggregation_fn=FLAGS.doc_aggregation_fn,
      qry_layers_to_use=FLAGS.qry_layers_to_use,
      qry_aggregation_fn=FLAGS.qry_aggregation_fn,
      dropout=FLAGS.question_dropout,
      qry_num_layers=FLAGS.question_num_layers,
      projection_dim=FLAGS.projection_dim,
      normalize_emb=FLAGS.normalize_emb,
      reconstruction_weight=FLAGS.qry_reconstruction_weight,
      ent_decomp_weight=FLAGS.ent_decomp_weight,
      rel_decomp_weight=FLAGS.rel_decomp_weight,
      train_bert=FLAGS.train_bert,
      shared_bert_for_qry=FLAGS.shared_bert_for_qry,
      load_only_bert=FLAGS.load_only_bert)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  if FLAGS.do_train:
    json.dump(tf.app.flags.FLAGS.flag_values_dict(),
              tf.gfile.Open(os.path.join(FLAGS.output_dir, "flags.json"), "w"))

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    if FLAGS.train_tfrecord_file is None:
      train_examples = read_squad_examples(
          input_file=FLAGS.train_file,
          is_training=True,
          include_entity_in_question=FLAGS.include_entity_in_question,
          p=FLAGS.subject_mention_probability)
      num_examples = len(train_examples)
      num_train_steps = int(num_examples / FLAGS.train_batch_size *
                            FLAGS.num_train_epochs)
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

      # Pre-shuffle the input to avoid having to make a very large shuffle
      # buffer in in the `input_fn`.
      rng = random.Random(12345)
      rng.shuffle(train_examples)

      # We write to a temporary file to avoid storing very large
      # constant tensors in memory.
      train_writer = FeatureWriter(
          filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
          is_training=True)
      convert_examples_to_features(
          examples=train_examples,
          tokenizer=tokenizer,
          max_doc_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          max_entity_length=FLAGS.max_entity_length,
          is_training=True,
          output_fn=train_writer.process_feature)
      train_writer.close()
      tfrecord_filename = train_writer.filename
      del train_examples
    else:
      num_examples = sum(
          1 for _ in tf.python_io.tf_record_iterator(FLAGS.train_tfrecord_file))
      num_train_steps = int(num_examples / FLAGS.train_batch_size *
                            FLAGS.num_train_epochs)
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
      tfrecord_filename = FLAGS.train_tfrecord_file

  summary_obj = None
  model_fn = model_fn_builder(
      bert_config=bert_config,
      qa_config=qa_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      summary_obj=summary_obj)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", num_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train(tfrecord_filename, estimator, num_train_steps)

  if FLAGS.do_test:
    test_examples = read_squad_examples(
        input_file=FLAGS.test_file, is_training=False)
    test_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "test.tf_record"),
        is_training=False)
    test_features = []

    def append_feature(feature):
      test_features.append(feature)
      test_writer.process_feature(feature)

    convert_examples_to_features(
        examples=test_examples,
        tokenizer=tokenizer,
        max_doc_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        max_entity_length=FLAGS.max_entity_length,
        is_training=False,
        output_fn=append_feature)
    test_writer.close()

    ckpt_path = os.path.join(FLAGS.output_dir, "best_model")
    output_prediction_file = os.path.join(FLAGS.output_dir,
                                          "test_predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "test_nbest.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir,
                                             "test_null_log_odds.json")
    single_eval(test_writer.filename, estimator, test_examples, test_features,
                ckpt_path, output_prediction_file, output_nbest_file,
                output_null_log_odds_file)
    micro, macro, _, _ = evaluate.compute_scores(FLAGS.test_file,
                                                 output_prediction_file)
    message = (
        "Model %s Micro-P %.3f Micro-R %.3f Micro-F %.3f "
        "Macro-P %.3f Macro-R %.3f Macro-F %.3f" %
        (ckpt_path, micro[0], micro[1], micro[2], macro[0], macro[1], macro[2]))
    tf.logging.info(message)
    with tf.gfile.Open(os.path.join(FLAGS.output_dir, "test_scores.txt"),
                       "w") as fo:
      fo.write(message + "\n")

  if FLAGS.do_predict:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    if FLAGS.num_eval_examples is not None:
      eval_examples = random.sample(eval_examples, FLAGS.num_eval_examples)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    # pylint: disable=function-redefined
    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_doc_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        max_entity_length=FLAGS.max_entity_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    continuous_eval(eval_writer.filename, estimator, eval_examples,
                    eval_features)

  if FLAGS.do_embed_docs:
    tf.gfile.MakeDirs(FLAGS.doc_embed_output_dir)

    doc_examples = read_docs_to_embed(input_file=FLAGS.doc_embed_file)

    doc_writer = FeatureWriter(
        filename=os.path.join(FLAGS.doc_embed_output_dir, "docs.tf_record"),
        is_training=False)
    doc_token_index_to_keep = []
    total_tokens = [0]

    def append_feature_doc(feature):
      doc_token_index_to_keep.append(feature.doc_token_index_to_keep)
      total_tokens[0] += len(feature.doc_token_index_to_keep)
      doc_writer.process_feature(feature)

    convert_examples_to_features(
        examples=doc_examples,
        tokenizer=tokenizer,
        max_doc_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        max_entity_length=FLAGS.max_entity_length,
        is_training=False,
        output_fn=append_feature_doc,
        keep_all_indices=not FLAGS.filter_tokens_to_keep)
    doc_writer.close()

    ckpt_path = None
    if FLAGS.use_best_ckpt_for_predict:
      ckpt_path = os.path.join(FLAGS.output_dir, "best_model")

    if FLAGS.projection_dim is not None:
      dimension = FLAGS.projection_dim
    else:
      dimension = bert_config.hidden_size
    doc_features, doc_spans = embed_docs(doc_writer.filename, estimator,
                                         ckpt_path, doc_token_index_to_keep,
                                         total_tokens[0], dimension)
    np.save(
        tf.gfile.Open(
            os.path.join(FLAGS.doc_embed_output_dir, "doc_features.npy"), "w"),
        doc_features,
    )
    np.save(
        tf.gfile.Open(
            os.path.join(FLAGS.doc_embed_output_dir, "doc_spans.npy"), "w"),
        doc_spans,
    )

  if FLAGS.do_embed_qrys:
    tf.gfile.MakeDirs(FLAGS.qry_embed_output_dir)

    qry_examples = read_qrys_to_embed(input_file=FLAGS.qry_embed_file)

    qry_writer = FeatureWriter(
        filename=os.path.join(FLAGS.qry_embed_output_dir, "qrys.tf_record"),
        is_training=False)

    def append_feature_qry(feature):
      qry_writer.process_feature(feature)

    convert_examples_to_features(
        examples=qry_examples,
        tokenizer=tokenizer,
        max_doc_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        max_entity_length=FLAGS.max_entity_length,
        is_training=False,
        output_fn=append_feature_qry)
    qry_writer.close()

    ckpt_path = None
    if FLAGS.use_best_ckpt_for_predict:
      ckpt_path = os.path.join(FLAGS.output_dir, "best_model")

    qry_st_features, qry_en_features, qry_bow_features = embed_qrys(
        qry_writer.filename, estimator, ckpt_path)
    np.save(
        tf.gfile.Open(
            os.path.join(FLAGS.qry_embed_output_dir, "qry_st_features.npy"),
            "w"),
        qry_st_features,
    )
    np.save(
        tf.gfile.Open(
            os.path.join(FLAGS.qry_embed_output_dir, "qry_en_features.npy"),
            "w"),
        qry_en_features,
    )
    np.save(
        tf.gfile.Open(
            os.path.join(FLAGS.qry_embed_output_dir, "qry_bow_features.npy"),
            "w"),
        qry_bow_features,
    )


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
