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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import random
from language.tek_representations import tokenization
from language.tek_representations.utils import util
from nltk import tokenize
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("background_corpus_file", None,
                    "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("tokenizer_type", "roberta", "roberta | bert")
flags.DEFINE_string("mask_type", "span", "span based masking")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool("mask_only_context", False,
                  "Whether to mask only the context segment")

flags.DEFINE_bool("mask_prop_from_context", False,
                  "Whether to mask x% of context len")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
flags.DEFINE_bool("whole_document_entities", False,
                  "If true take enitity pages from all of the document")
flags.DEFINE_integer("max_background_len", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer(
    "long_sentence_threshold", 70,
    "Penalize longer sentences. Use -1 to disregard sentence len.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.0,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
flags.DEFINE_float("background_probability", 1.0,
                   "Probability of including background segment")

flags.DEFINE_integer("lower", 1, "min number of subwords to be masked")
flags.DEFINE_integer("upper", 10, "max number of sybwords to be masked")
flags.DEFINE_float("geo_p", 0.2, "Geometric distribution param")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, positions, masked_lm_positions,
               masked_lm_labels, is_background_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.positions = positions
    self.is_background_next = is_background_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "positions: %s\n" % (" ".join([str(x) for x in self.positions]))
    s += "is_background_next: %s\n" % self.is_background_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_tf_example(instance, tokenizer, max_seq_length,
                      max_predictions_per_seq):
  """Create features for each instance."""
  input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
  input_mask = [1] * len(input_ids)
  segment_ids = list(instance.segment_ids)
  positions = list(instance.positions)
  assert len(input_ids) <= max_seq_length

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    positions.append(positions[-1] + 1)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(positions) == max_seq_length

  masked_lm_positions = list(instance.masked_lm_positions)
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
  masked_lm_weights = [1.0] * len(masked_lm_ids)

  while len(masked_lm_positions) < max_predictions_per_seq:
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  next_sentence_label = 1 if instance.is_background_next else 0

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  features["positions"] = create_int_feature(positions)
  features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
  features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
  features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
  features["next_sentence_labels"] = create_int_feature([next_sentence_label])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example, features


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    tf_example, features = create_tf_example(instance, tokenizer,
                                             max_seq_length,
                                             max_predictions_per_seq)

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      num_masked = len([x for x in instance.tokens if x == tokenizer.mask])
      tf.logging.info("Num masked: {} of {}".format(num_masked,
                                                    len(instance.tokens)))
      tf.logging.info("tokens: {}".format(" ".join([
          tokenization.printable_text(
              x, strip_roberta_space=(FLAGS.tokenizer_type == "roberta"))
          for x in instance.tokens
      ])))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info("%s: %s" %
                        (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def split_into_sentences(text, doc_annotations, tokenizer):
  """Split into sentences and return bookkeeping info."""
  sentences = []
  sentences_starts = []
  sentence_annotations = []
  doc_annotations = sorted(doc_annotations, key=lambda x: x[2])
  annotation_idx = 0
  sentences_text = tokenize.sent_tokenize(text)
  token_idx = 0
  for sentence_text in sentences_text:
    sub_tokens, word_starts = tokenizer.tokenize(sentence_text)
    sentences.append(sub_tokens)
    sentences_starts.append(word_starts)
    sentence_annotations.append([])
    token_idx += len(sentence_text.split(" "))
    while annotation_idx < len(
        doc_annotations) and doc_annotations[annotation_idx][2] < token_idx:
      sentence_annotations[-1].append(doc_annotations[annotation_idx])
      annotation_idx += 1
  return sentences, sentences_starts, sentence_annotations


def create_training_instances(input_files, background_corpus_file, tokenizer,
                              max_seq_length, max_background_len, dupe_factor,
                              masked_lm_prob, max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  input_files = input_files[0] if len(input_files) == 1 else input_files
  corpus = util.get_corpus(str(input_files))
  for idx, (raw_id, raw_doc) in enumerate(corpus.iteritems()):
    qid = tokenization.convert_to_unicode(raw_id)
    doc_json = json.loads(tokenization.convert_to_unicode(raw_doc))
    sentences, starts, all_sent_annotations = split_into_sentences(
        doc_json["text"], doc_json["wiki_links"], tokenizer)
    for sentence, starts_for_sent, sent_annotations in zip(
        sentences, starts, all_sent_annotations):
      all_documents[-1].append(
          (qid, sentence, starts_for_sent, sent_annotations))
    all_documents.append([])
    if idx > 5:
      break
  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = tokenizer.vocab
  instances = []
  corpus = util.get_corpus(background_corpus_file)
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              corpus, tokenizer, all_documents[document_index], max_seq_length,
              max_background_len, masked_lm_prob, max_predictions_per_seq,
              vocab_words, rng, FLAGS.mask_only_context,
              FLAGS.background_probability, FLAGS.mask_prop_from_context,
              FLAGS.long_sentence_threshold))

  return instances


def create_instances_from_document(corpus, tokenizer, document, max_seq_length,
                                   max_background_len, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng,
                                   mask_only_context, background_probability,
                                   mask_prop_from_context,
                                   long_sentence_threshold):
  """Creates `TrainingInstance`s for a single document."""

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3
  target_seq_length = max_num_tokens - max_background_len
  instances = []
  current_chunk = []
  current_length = 0
  current_entity_set = set()

  i = 0
  while i < len(document):
    qid, segment, _, sent_annotations = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    segment_entities = set([
        annotation[0] for annotation in sent_annotations if annotation[0] != qid
    ])
    for entity in segment_entities:
      current_entity_set.add(entity)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = len(current_chunk)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Background next
        is_background_next = False
        # Select background sentences if...
        if max_background_len > 0 and current_entity_set and (
            i == len(document) - 1 or rng.random() < background_probability):
          is_background_next = True
          target_b_length = max(max_num_tokens - len(tokens_a),
                                max_background_len)
          # Fill tokens b until target_b_len is reached
          add_to_background(tokens_a, current_entity_set, corpus, tokenizer,
                            tokens_b, target_b_length, long_sentence_threshold)
        # Actual next
        else:
          is_background_next = False
          i += 1
          while i < len(document):
            _, segment, _, _ = document[i]
            for token in segment:
              if len(tokens_a) + len(tokens_b) < max_seq_length:
                tokens_b.append(token)
            i += 1
          i -= 1
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
        assert len(tokens_a) >= 1

        tokens = []
        segment_ids = []
        positions = []
        # segment a -- context
        a_segment_id = 1
        tokens.append(tokenizer.bos)
        segment_ids.append(a_segment_id)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(a_segment_id)
        tokens.append(tokenizer.eos)
        segment_ids.append(a_segment_id)
        positions = list(range(len(tokens)))
        context_len = len(tokens)

        # segment b -- background
        if tokens_b:
          b_seg_id = 2
          for token in tokens_b:
            tokens.append(token)
            segment_ids.append(b_seg_id)
          tokens.append(tokenizer.eos)
          segment_ids.append(b_seg_id)
          positions += list(range(len(tokens) - context_len))

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, tokenizer, max_predictions_per_seq,
             vocab_words, rng, context_len, mask_only_context,
             mask_prop_from_context)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            positions=positions,
            is_background_next=is_background_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
      current_entity_set = current_entity_set if FLAGS.whole_document_entities else set(
      )
    i += 1

  return instances


def add_to_background(query_tokens, doc_ids, corpus, tokenizer,
                      background_tokens, max_sentence_len,
                      long_sentence_threshold):
  """Sort and add sentences to the background."""
  sentence_scores = score_sentences(query_tokens, doc_ids, corpus, tokenizer,
                                    long_sentence_threshold)
  sentence_scores = sorted(sentence_scores, key=lambda x: x[-1], reverse=True)
  previous_entity = None
  for i, (name, tokens, _, _) in enumerate(sentence_scores):
    if name == previous_entity:
      tokens_to_add = tokens
    else:
      if i > 0:
        background_tokens.append(tokenizer.eos)
      tokenized_entity, _ = tokenizer.tokenize(name)
      tokens_to_add = tokenized_entity + [tokenizer.entity_separator] + tokens
    if len(background_tokens) >= max_sentence_len:
      break
    background_tokens += tokens_to_add
    previous_entity = name
  return sentence_scores


def score_sentences(query_tokens,
                    doc_ids,
                    corpus,
                    tokenizer,
                    long_sentence_threshold,
                    n=3):
  """Score sentences with respect to the query."""
  query_ngrams = util.get_ngrams(query_tokens, n)
  sentence_scores = []
  for doc_id in doc_ids:
    try:
      doc_json = json.loads(
          tokenization.convert_to_unicode(corpus[doc_id.encode("utf-8")]))
    except KeyError:
      continue
    sentences, starts_text, _ = split_into_sentences(doc_json["text"], [],
                                                     tokenizer)
    name_tokens, _ = tokenizer.tokenize(doc_json["name"])
    for sentence, starts_sentence in zip(sentences, starts_text):
      tokens = name_tokens + [":"] + sentence
      # starts = starts_name + [True] + starts_sentence
      sentence_ngrams = util.get_ngrams(tokens, n)
      long_sent_penalty = len(tokens) if (
          long_sentence_threshold != -1 and
          len(tokens) > long_sentence_threshold) else 1
      score = len(set(sentence_ngrams).intersection(query_ngrams)) / (
          max(1, len(query_ngrams)) * long_sent_penalty)
      sentence_scores.append(
          (doc_json["name"], sentence, starts_sentence, score))
  return sentence_scores


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def get_span_len(lower, upper, geo_p):
  lens = list(range(lower, upper + 1))
  len_distrib = [
      geo_p * (1 - geo_p)**(i - lower) for i in range(lower, upper + 1)
  ]
  len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
  span_len = np.random.choice(lens, p=len_distrib)
  return span_len


def create_masked_lm_predictions(tokens, masked_lm_prob, tokenizer,
                                 max_predictions_per_seq, vocab_words, rng,
                                 context_len, mask_only_context,
                                 mask_prop_from_context):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  num_subwords_in_span = 0
  for (i, token) in enumerate(tokens):
    if token == tokenizer.bos or token == tokenizer.eos:
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.mask_type == "span" and len(cand_indexes) >= 1 and
        num_subwords_in_span > 0):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])
      if num_subwords_in_span == 0:
        num_subwords_in_span = get_span_len(FLAGS.lower, FLAGS.upper,
                                            FLAGS.geo_p)
    num_subwords_in_span -= 1

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)
  if mask_only_context and mask_prop_from_context:
    to_predict = int(round(context_len * masked_lm_prob))
  else:
    to_predict = int(round(len(tokens) * masked_lm_prob))

  num_to_predict = min(max_predictions_per_seq, max(1, to_predict))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes or (mask_only_context and
                                      index > context_len):
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = tokenizer.mask
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    if FLAGS.max_background_len > 0:
      trunc_tokens = tokens_a if (
          len(tokens_a) > len(tokens_b) and
          len(tokens_b) < FLAGS.max_background_len) else tokens_b
    else:
      trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

    assert len(trunc_tokens) >= 1
    trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tokenizer = tokenization.get_tokenizer(
      FLAGS.tokenizer_type, FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(input_files,
                                        FLAGS.background_corpus_file, tokenizer,
                                        FLAGS.max_seq_length,
                                        FLAGS.max_background_len,
                                        FLAGS.dupe_factor, FLAGS.masked_lm_prob,
                                        FLAGS.max_predictions_per_seq, rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
