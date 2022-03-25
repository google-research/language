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
"""Run BERT on MRQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import math
import os
import random

from bert import modeling
from bert import optimization
from bert import tokenization as bert_tokenization
from language.tek_representations import background
from language.tek_representations import tokenization
from language.tek_representations.utils import mrqa_official_eval
from language.tek_representations.utils import triviaqa_evaluation
from language.tek_representations.utils import util
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained muppet model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "tokenizer_type", "roberta",
    "roberta | bert. This is used to call the appropriate tokenizer.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the muppet model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("eval_metric", "f1",
                    "Eval metric used for selecting model on the dev set.")

flags.DEFINE_string(
    "prefix", None,
    "String that denotes the preprocessing (e.g., type.None-msl.512-mbg.0 indicates a max seq len of 512 and no background knowledge)"
)

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_float(
    "include_unknowns", 1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

tf.flags.DEFINE_string(
    "num_train_file", None,
    "File that maps the dataset and preprocessing to number of training data points."
)

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

flags.DEFINE_string("datasets", None,
                    "A 0-separated list of datasets used for training.")

flags.DEFINE_string(
    "eval_datasets", None,
    "A 0-separated list of datasets used for validation/testing")

flags.DEFINE_bool("triviaqa_eval", False, "Prefix for triviaqa")

flags.DEFINE_string(
    "predict_file", None,
    "MRQA annotated json for predictions. E.g., dev-v1.1.jsonl")

flags.DEFINE_string("metrics_file", None, "File containing metrics")

flags.DEFINE_string("nbest_file", None, "File containing debug info")

flags.DEFINE_string(
    "eval_tf_filename", None,
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

flags.DEFINE_string(
    "eval_features_file", None,
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in MRQA prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("background_type", None,
                    "Use one of rare | following | ngram | contains")

flags.DEFINE_bool(
    "unconstrained_predictions", False,
    "Whether to allow predcitions from the background portion of the context window"
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("max_background_tokens", 256,
                     "The maximum number of tokens for the background.")

flags.DEFINE_integer("max_single_source_tokens", 50,
                     "The maximum number of tokens for the background.")

flags.DEFINE_float(
    "annotation_confidence_threshold", 0.1,
    "Adds only those entity pages that have a confidence above this threshold")

flags.DEFINE_string("corpus_file",
                    "",
                    "File containing Wikipedia articles")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 32,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("num_evals_per_epoch", 3,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal NQ evaluation.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")


def parse_prefix():
  """Parse the string signifying the preprocessing."""
  if FLAGS.prefix is not None:
    parts = FLAGS.prefix.split("-")
    for part in parts:
      fname, value = part.split(".")
      if fname == "type":
        FLAGS.background_type = value if value != "None" else None
      elif fname == "msl":
        FLAGS.max_seq_length = int(value)
      elif fname == "mbg":
        FLAGS.max_background_tokens = int(value)
      else:
        raise NotImplementedError()


def set_train_num_precomputed():
  """Set FLAGS: train_num_precomputed, datasets, eval_datasets."""
  FLAGS.datasets = FLAGS.datasets.split("0")
  FLAGS.eval_datasets = FLAGS.datasets if FLAGS.eval_datasets is None or not FLAGS.eval_datasets else FLAGS.eval_datasets.split(
      "0")
  lines = []
  if FLAGS.num_train_file is not None:
    fnames = tf.io.gfile.glob(FLAGS.num_train_file)
    FLAGS.train_num_precomputed = 0
    for fname in fnames:
      with tf.gfile.Open(fname) as f:
        for line in f:
          lines.append(line)
          parts = line.strip().split(",")
          value = int(parts[1][:-1].strip())
          key = parts[0][2:-1]
          dataset, prefix = key.split("/")
          if prefix == FLAGS.prefix and dataset in set(FLAGS.datasets):
            FLAGS.train_num_precomputed += value
    tf.logging.info("FLAGS.train_num_precomputed: {}".format(
        FLAGS.train_num_precomputed))
    if FLAGS.train_num_precomputed == 0:
      raise ValueError("Can't find config")


class MRQAExample(object):
  """A single training/test example for the MRQA dataset.

  For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               int_id,
               doc_text,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               char_to_word_offset=None,
               start_position=None,
               end_position=None,
               word_to_char_offset=None,
               document_annotation=None,
               question_annotation=None):
    self.qas_id = qas_id
    self.int_id = int_id
    self.question_text = question_text
    self.doc_text = doc_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.char_to_word_offset = char_to_word_offset
    self.start_position = start_position
    self.end_position = end_position
    self.document_annotation = document_annotation
    self.question_annotation = question_annotation
    self.word_to_char_offset = word_to_char_offset

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (self.qas_id)
    s += ", question_text: %s" % (self.question_text)
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.end_position:
      s += ", end_position: %d" % (self.end_position)
    return s


class CreateTFExampleFn(object):
  """Functor for creating MRQA tf.Examples."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.corpus = util.get_corpus(FLAGS.corpus_file)
    self.tokenizer = tokenization.get_tokenizer(
        FLAGS.tokenizer_type,
        FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)

  def process(self, example):
    """Coverts an MRQA example in a list of serialized tf examples."""
    mrqa_examples = read_mrqa_entry(example, self.is_training)
    input_features = []
    local_cache = {}
    for idx, mrqa_example in enumerate(mrqa_examples):
      input_features.extend(
          convert_single_example(
              mrqa_example,
              self.tokenizer,
              self.is_training,
              self.corpus,
              local_cache,
              idx=idx))

    for input_feature in input_features:
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["example_index"] = create_int_feature(
          [input_feature.example_index])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
      else:
        token_map = [-1] * len(input_feature.input_ids)
        for k, v in input_feature.token_to_orig_map.items():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString(), input_feature.to_json()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               start_position=None,
               end_position=None,
               match_score=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.start_position = start_position
    self.end_position = end_position
    self.match_score = match_score

  def to_json(self):
    """Convert to json string."""
    json_dict = {}
    json_dict["unique_id"] = self.unique_id
    json_dict["example_index"] = self.example_index
    json_dict["doc_span_index"] = self.doc_span_index
    json_dict["tokens"] = self.tokens
    json_dict["token_to_orig_map"] = self.token_to_orig_map
    json_dict["token_is_max_context"] = self.token_is_max_context
    json_dict["input_ids"] = self.input_ids
    json_dict["input_mask"] = self.input_mask
    json_dict["start_position"] = self.start_position
    json_dict["end_position"] = self.end_position
    json_dict["match_score"] = self.match_score
    return json.dumps(json_dict)

  @classmethod
  def from_json(cls, json_dict):
    json_dict["token_to_orig_map"] = {
        int(k): v for k, v in json_dict["token_to_orig_map"].items()
    }
    json_dict["token_is_max_context"] = {
        int(k): v for k, v in json_dict["token_is_max_context"].items()
    }
    return cls(
        unique_id=json_dict["unique_id"],
        example_index=json_dict["example_index"],
        doc_span_index=json_dict["doc_span_index"],
        tokens=json_dict["tokens"],
        token_to_orig_map=json_dict["token_to_orig_map"],
        token_is_max_context=json_dict["token_is_max_context"],
        input_ids=json_dict.get("input_ids", None),
        input_mask=json_dict.get("input_mask", None),
        start_position=json_dict.get("start_position", None),
        end_position=json_dict.get("end_position", None),
        match_score=json_dict.get("match_score", None))


def read_mrqa_entry(entry, is_training):
  """Converts a MRQA entry into a list of MRQAExamples."""

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  paragraph_text = entry["context"]
  document_annotation = convert_byte_to_char(
      paragraph_text, entry.get("document_annotation", None))
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

  for qa in entry["qas"]:
    qas_id = qa["qid"]
    int_id = qa["int_id"]
    question_text = qa["question"]
    start_position = None
    end_position = None
    orig_answer_text = None
    if is_training:
      spans = []
      for answer in qa["detected_answers"]:
        spans += answer["char_spans"]
      spans = sorted(spans)
      if not spans:
        continue
      # take first span
      char_start, char_end = spans[0][0], spans[0][1]
      orig_answer_text = paragraph_text[char_start:char_end + 1]
      start_position, end_position = char_to_word_offset[
          char_start], char_to_word_offset[char_end]
    word_to_char_begin, word_to_char_end = {}, {}
    for cindex, windex in enumerate(char_to_word_offset):
      word_to_char_end[windex] = cindex
      word_to_char_begin[windex] = word_to_char_begin.get(windex, cindex)

    example = MRQAExample(
        qas_id=qas_id,
        int_id=int_id,
        question_text=question_text,
        doc_text=paragraph_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        char_to_word_offset=char_to_word_offset,
        start_position=start_position,
        end_position=end_position,
        word_to_char_offset=(word_to_char_begin, word_to_char_end),
        document_annotation=document_annotation,
        question_annotation=convert_byte_to_char(
            question_text, qa.get("question_annotation", None)))
    examples.append(example)
  return examples


def convert_byte_to_char(text, annotation):
  """Convert the byte offset in the annotation to char offsets."""
  if annotation is not None:
    byte_to_char = {}
    byte_offset = 0
    for i, c in enumerate(text):
      byte_to_char[byte_offset] = i
      byte_offset += len(c.encode("utf-8"))

    for entity in annotation:
      for mention in entity["mentions"]:
        mention["begin"] = byte_to_char[mention["begin"]] if mention[
            "begin"] in byte_to_char else mention["begin"]
        mention["end"] = byte_to_char[mention["end"]] if mention[
            "end"] in byte_to_char else mention["end"]
  return annotation


def read_mrqa_entries(lines, is_training):
  examples = []
  for line in lines:
    examples.extend(read_mrqa_entry(line, is_training))
  return examples


def read_mrqa_examples(input_file, is_training):
  """Read a MRQA json file into a list of MRQAExample."""
  with gzip.GzipFile(fileobj=tf.gfile.Open(input_file)) as reader:
    # skip header
    content = reader.read().decode("utf-8").strip().split("\n")[1:]
    content = [json.loads(line) for line in content]
    examples = read_mrqa_entries(content, is_training)
  return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 is_training,
                                 features_file=None):
  """Converts a list of NqExamples into InputFeatures."""
  eval_features = []
  if features_file is not None:
    for features_file_shard in tf.gfile.Glob(features_file):
      with tf.gfile.Open(features_file_shard) as f:
        for line in f:
          features_json = json.loads(line)
          if "positions" in features_json:
            del features_json["positions"]
            del features_json["segment_ids"]
          del features_json["input_mask"]
          del features_json["input_ids"]
          eval_features.append(InputFeatures.from_json(features_json))
  else:
    corpus = util.get_corpus(FLAGS.corpus_file)
    for example in examples:
      local_cache = {}
      features = convert_single_example(example, tokenizer, is_training, corpus,
                                        local_cache)

      for feature in features:
        feature.unique_id = feature.example_index + feature.doc_span_index
        eval_features.append(feature)
  tf.logging.info("Loaded {} features".format(len(eval_features)))
  return eval_features


def convert_single_example(example,
                           tokenizer,
                           is_training,
                           corpus,
                           local_cache,
                           idx=-1):
  """Converts a single MRQAExample into a list of InputFeatures."""
  example_id = example.int_id
  query_tokens, _ = tokenizer.tokenize(example.question_text)

  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[0:FLAGS.max_query_length]

  # subtoken to orig token
  tok_to_orig_index = []
  # orig to starting subtoken
  orig_to_tok_index = []
  all_doc_tokens = []
  features = []
  doc_tokens = example.doc_tokens
  doc_subtokens, doc_word_starts = tokenizer.tokenize(doc_tokens)
  token_idx = -1
  for sub_token, is_start in zip(doc_subtokens, doc_word_starts):
    if is_start:
      token_idx += 1
      orig_to_tok_index.append(len(all_doc_tokens))
    all_doc_tokens.append(sub_token)
    tok_to_orig_index.append(token_idx)
  subtoken_to_doc_token = [i for i in tok_to_orig_index]

  tok_start_position = 0
  tok_end_position = 0
  if is_training:
    tok_start_position = -1
    tok_end_position = -1
  if is_training:
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
      tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
      tok_end_position = len(all_doc_tokens) - 1
    (tok_start_position, tok_end_position) = _improve_answer_span(
        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
        example.orig_answer_text)

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(
      query_tokens) - 3 - FLAGS.max_background_tokens

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
    start_offset += min(length, FLAGS.doc_stride)

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    tokens.append(tokenizer.bos)
    for token in query_tokens:
      tokens.append(token)
    tokens.append(tokenizer.eos)
    qlen = len(tokens)

    char_to_window_idx = {}
    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      char_offset = example.word_to_char_offset[0][subtoken_to_doc_token[
          doc_span.start + i]] - example.word_to_char_offset[0][
              subtoken_to_doc_token[doc_span.start]]
      if char_offset not in char_to_window_idx:
        char_to_window_idx[char_offset] = len(tokens)
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
    tokens.append(tokenizer.eos)

    # add background tokens or extra context
    annotation = example.document_annotation
    if annotation is not None and FLAGS.max_background_tokens > 0:
      background_sub_tokens = []
      segment_char_begin = example.word_to_char_offset[0][subtoken_to_doc_token[
          doc_span.start]]
      segment_char_end = example.word_to_char_offset[1][subtoken_to_doc_token[
          doc_span.start + doc_span.length - 1]]
      tokenize_fn = lambda x: tokenizer.tokenize(x)[0]

      if FLAGS.background_type.endswith("ngram"):
        background_sub_tokens = background.get_sentence_contexts_by_overlap(example.question_text,\
                                                        segment_char_begin, segment_char_end, \
                                                        example.document_annotation, \
                                                        tokenizer, \
                                                        FLAGS.annotation_confidence_threshold,\
                                                        FLAGS.max_background_tokens - 1, \
                                                        FLAGS.max_single_source_tokens, \
                                                        corpus, local_cache)
      elif FLAGS.background_type == "following":
        background_sub_tokens = background.get_following_contexts(all_doc_tokens, tokenize_fn,\
                                                                         split_token_index + 1, \
                                                                         min(FLAGS.max_background_tokens - 1, \
                                                                         len(all_doc_tokens) - split_token_index - 1))
      if background_sub_tokens:
        tokens += background_sub_tokens
        tokens.append(tokenizer.eos)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    query_ngrams = util.get_ngrams(tokens[:qlen], 3)
    context_ngrams = util.get_ngrams(tokens[qlen:], 3)
    score = len(set(context_ngrams).intersection(query_ngrams)) / max(
        1, len(query_ngrams))

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < FLAGS.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)

    assert len(input_ids) == FLAGS.max_seq_length, (len(input_ids), input_ids)
    assert len(input_mask) == FLAGS.max_seq_length

    start_position = None
    end_position = None

    if is_training:
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      out_of_span = False
      if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
        out_of_span = True
      if out_of_span:
        if (FLAGS.include_unknowns < 0 or
            random.random() > FLAGS.include_unknowns):
          continue
        start_position = 0
        end_position = 0
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset
    if idx != -1 and idx < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("example_id: %s" % (example_id))
      tf.logging.info("doc_span_index: %s" % (doc_span_index))
      tf.logging.info("tokens: %s" % " ".join(tokens))
      tf.logging.info(
          "token_to_orig_map: %s" %
          " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
      tf.logging.info("token_is_max_context: %s" % " ".join(
          ["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      if is_training:
        answer_text = " ".join(tokens[start_position:(end_position + 1)])
        tf.logging.info("start_position: %d" % (start_position))
        tf.logging.info("end_position: %d" % (end_position))
        tf.logging.info("answer: %s" % (answer_text))

    features.append(
        InputFeatures(
            unique_id=-1,
            example_index=example_id,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            start_position=start_position,
            end_position=end_position,
            match_score=score))

  return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""
  tokens, _ = tokenizer.tokenize(orig_answer_text)
  tok_answer_text = tokenizer.tokenized_to_original(tokens)

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = tokenizer.tokenized_to_original(
          doc_tokens[new_start:(new_end + 1)]).strip()
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
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


def create_model(bert_config, is_training, input_ids, input_mask,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/nq/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    example_index = features["example_index"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
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
    if mode == tf_estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      # Computes the loss for positions.
      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "example_index": example_index,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "example_index": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

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
    d = tf.data.Dataset.list_files(input_file, shuffle=True)
    d = d.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=50, sloppy=is_training))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "example_index", "start_logits", "end_logits"])
PrelimPrediction = collections.namedtuple("PrelimPrediction", [
    "feature_index", "start_index", "end_index", "start_logit", "end_logit",
    "doc_span_index"
])
NbestPrediction = collections.namedtuple(
    "NbestPrediction",
    ["text", "start_logit", "end_logit", "example_index", "doc_span_index"])


def make_predictions(all_examples, all_features, all_results):
  """Create prediction dict from results."""
  tokenizer = tokenization.get_tokenizer(
      FLAGS.tokenizer_type, FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index = feature.example_index
    example_index_to_features[example_index].append(feature)
  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[str(result.unique_id) + "_" +
                        str(result.example_index)] = result

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  max_context_index = FLAGS.max_seq_length if FLAGS.unconstrained_predictions else (
      FLAGS.max_seq_length - 2 - FLAGS.max_background_tokens)
  for example in all_examples:
    example_index = example.int_id
    features = example_index_to_features[example_index]
    prelim_predictions = []
    for (feature_index, feature) in enumerate(features):
      unique_id = str(feature.unique_id) + "_" + str(example_index)
      result = unique_id_to_result[unique_id]
      start_indexes = _get_best_indexes(result.start_logits, FLAGS.n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, FLAGS.n_best_size)
      for start_index in start_indexes:
        for end_index in end_indexes:
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          if start_index > max_context_index or end_index > max_context_index:
            continue
          length = end_index - start_index + 1
          if length > FLAGS.max_answer_length:
            continue
          prelim_predictions.append(
              PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index],
                  doc_span_index=feature.doc_span_index))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= FLAGS.n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = tokenizer.tokenized_to_original(tok_tokens).strip()
        orig_text = " ".join(orig_tokens)
        final_text = get_final_text(tok_text, orig_text)
        if final_text in seen_predictions:
          continue
        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              example_index=example_index,
              doc_span_index=pred.doc_span_index))

    if not nbest:
      nbest.append(
          NbestPrediction(
              text="empty",
              start_logit=0.0,
              end_logit=0.0,
              example_index=-1,
              doc_span_index=-1))
    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)
    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["example_index"] = entry.example_index
      output["doc_span_index"] = entry.doc_span_index
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    all_predictions[example.qas_id] = nbest_json[0]["text"]
    all_nbest_json[example.qas_id] = nbest_json

  return all_predictions, all_nbest_json


def get_final_text(pred_text, orig_text):
  """Project the tokenized prediction back to the original text."""

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

  tokenizer = bert_tokenization.BasicTokenizer(
      do_lower_case=FLAGS.do_lower_case)
  tok_text = " ".join(tokenizer.tokenize(orig_text))
  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info("Final: Unable to find text: '%s' in '%s'" %
                      (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Final: Length not equal after stripping spaces: '%s' vs '%s'",
          orig_ns_text, tok_ns_text)
    return orig_text

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
      tf.logging.info("Final: Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Final: Couldn't map end position")
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


def get_raw_scores(dataset, predictions):
  """Calculate eval metrics."""
  answers = {}
  get_answers = (triviaqa_evaluation.get_ground_truths
                ) if FLAGS.triviaqa_eval else (lambda x: x)
  for example in dataset:
    for qa in example["qas"]:
      answers[qa["qid"]] = get_answers(qa["answers"])
  exact_scores = {}
  f1_scores = {}
  scoring_class = triviaqa_evaluation if FLAGS.triviaqa_eval else mrqa_official_eval
  for qid, ground_truths in answers.items():
    if qid not in predictions:
      print("Missing prediction for %s" % qid)
      continue
    prediction = predictions[qid]
    exact_scores[qid] = scoring_class.metric_max_over_ground_truths(
        scoring_class.exact_match_score, prediction,
        ground_truths) if ground_truths else 0
    f1_scores[qid] = scoring_class.metric_max_over_ground_truths(
        scoring_class.f1_score, prediction,
        ground_truths) if ground_truths else 0
  return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  """Make final evaluation dict containing EM and F1."""
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ("exact", 100.0 * sum(exact_scores.values()) / total),
        ("f1", 100.0 * sum(f1_scores.values()) / total),
        ("total", total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ("total", total),
    ])


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_precomputed_file:
      raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

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
  parse_prefix()
  set_train_num_precomputed()
  validate_flags_or_throw(bert_config)
  if FLAGS.output_dir is not None:
    tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.get_tokenizer(
      FLAGS.tokenizer_type, FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  save_checkpoints_steps = FLAGS.save_checkpoints_steps if FLAGS.num_evals_per_epoch == -1 else (
      FLAGS.train_num_precomputed /
      (FLAGS.train_batch_size * FLAGS.num_evals_per_epoch))
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          experimental_host_call_every_n_steps=1000,
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_features = FLAGS.train_num_precomputed
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # read eval files
  tf.logging.info("Reading eval data")

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))
    else:
      return tf.gfile.Open(path, "r")

  eval_dataset = []
  for predict_file in tf.gfile.Glob(FLAGS.predict_file):
    with _open(predict_file) as reader:
      for line in reader:
        line = line.strip()
        if not line:
          continue
        line_json = json.loads(line)
        if "header" not in line_json:
          eval_dataset.append(line_json)
  tf.logging.info("Read {} lines".format(len(eval_dataset)))
  eval_examples = read_mrqa_entries(eval_dataset, is_training=False)
  eval_features = convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      is_training=False,
      features_file=FLAGS.eval_features_file)
  tf.logging.info("Done reading eval data")

  if FLAGS.do_train:
    tf.logging.info("***** Running training on precomputed features *****")
    tf.logging.info("  Num split examples = %d", num_train_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_filename = FLAGS.train_precomputed_file
    train_input_fn = input_fn_builder(
        input_file=train_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    evaluate_checkpoints(estimator, eval_examples, eval_dataset, eval_features,
                         FLAGS.eval_tf_filename)

  if FLAGS.do_predict:
    if not FLAGS.output_prediction_file:
      raise ValueError(
          "--output_prediction_file must be defined in predict mode.")
    tf.logging.info("Using {} filenames matching {}".format(
        len(tf.gfile.Glob(FLAGS.eval_tf_filename)), FLAGS.eval_tf_filename))
    predict_input_fn = input_fn_builder(
        input_file=FLAGS.eval_tf_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    tf.logging.info('Existing "best" checkpoints {}'.format(
        tf.gfile.Glob(get_best_checkpoint(estimator) + "*")))
    if not tf.gfile.Glob(get_best_checkpoint(estimator) + "*"):
      evaluate_checkpoints(estimator, eval_examples, eval_dataset,
                           eval_features, FLAGS.eval_tf_filename)
    for result in estimator.predict(
        predict_input_fn,
        yield_single_examples=True,
        checkpoint_path=get_best_checkpoint(estimator)):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      example_index = int(result["example_index"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              example_index=example_index,
              start_logits=start_logits,
              end_logits=end_logits))

    result, preds, n_best_preds = compute_pred_dict(eval_examples,
                                                    eval_features, all_results,
                                                    eval_dataset)
    with tf.gfile.Open(FLAGS.output_prediction_file, "w") as f:
      json.dump(preds, f, indent=4)
    if result is not None and FLAGS.metrics_file is not None:
      with tf.gfile.Open(FLAGS.metrics_file, "w") as f:
        json.dump(result, f, indent=4)
    if FLAGS.nbest_file is not None:
      with tf.gfile.Open(FLAGS.nbest_file, "w") as f:
        json.dump(n_best_preds, f, indent=4)


def get_best_checkpoint(estimator):
  return os.path.join(estimator.model_dir, "model.ckpt-best")


def evaluate_checkpoints(estimator, eval_examples, eval_dataset, eval_features,
                         eval_tf_filename):
  """Evaluate all checkpoints, and keep only the best one."""
  best_result, best_checkpoint = None, None
  checkpoints = tf.train.get_checkpoint_state(
      estimator.model_dir).all_model_checkpoint_paths

  predict_input_fn = input_fn_builder(
      input_file=eval_tf_filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)
  tf.logging.info("Iterating checkpoints {}".format(checkpoints))
  for checkpoint in checkpoints:
    all_results = []
    for result in estimator.predict(
        predict_input_fn,
        yield_single_examples=True,
        checkpoint_path=checkpoint):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      example_index = int(result["example_index"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              example_index=example_index,
              start_logits=start_logits,
              end_logits=end_logits))
    result, _, _ = compute_pred_dict(eval_examples, eval_features, all_results,
                                     eval_dataset)
    if (best_result is None) or (result[FLAGS.eval_metric] >
                                 best_result[FLAGS.eval_metric]):
      best_result = result
      best_checkpoint = checkpoint
      tf.logging.info(
          "!!! Best dev %s %.2f %s" %
          (str(checkpoint), result[FLAGS.eval_metric], FLAGS.eval_metric))
  for fname in tf.io.gfile.glob(best_checkpoint + "*"):
    ext = os.path.splitext(fname)[-1]
    tf.io.gfile.rename(
        fname, get_best_checkpoint(estimator) + ext, overwrite=True)
  with tf.gfile.Open(
      os.path.join(estimator.model_dir, "org_checkpoint_name.txt"), "w") as f:
    f.write(best_checkpoint)
  tf.logging.info("Deleting checkpoints {}".format(checkpoints))
  for checkpoint in checkpoints:
    for fname in tf.io.gfile.glob(checkpoint + "*"):
      if tf.io.gfile.exists(fname):
        tf.gfile.Remove(fname)


def compute_pred_dict(eval_examples, eval_features, all_results, eval_dataset):
  """Compute predictions and evaluate."""
  preds, nbest_preds = make_predictions(eval_examples, eval_features,
                                        all_results)
  result = None
  if FLAGS.metrics_file is not None:
    exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
    result = make_eval_dict(exact_raw, f1_raw)
  if FLAGS.verbose_logging:
    tf.logging.info("***** Eval results *****")
    for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))
  return result, preds, nbest_preds


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()
