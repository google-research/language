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
"""A script to create TFExamples for model training and evaluation.

Starting from lists of nearest neighbors retrieved from a fast model, exports
TFExamples for training a reranking model with either full cross-attention
or multi-vector factored representation.
"""

import collections
import json
import os
import random

from absl import app
from absl import flags
from bert import tokenization
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

flags.DEFINE_string("neighbors_path", None, "Path to neighbors sstable.")

flags.DEFINE_string("passages_path", None, "Path to passage input sstable.")
flags.DEFINE_string("encoded_passages_path", None,
                    "Path to encoded passage input sstable.")
flags.DEFINE_string("queries_path", None, "Path to query input sstable.")
flags.DEFINE_string("encoded_queries_path", None,
                    "Path to encoded query input sstable.")
flags.DEFINE_integer("num_candidates", "20",
                     "Number of candidates to learn to rank.")
flags.DEFINE_string("examples_path", None, "Path examples tfrecord.")

flags.DEFINE_string("bert_hub_module_path",
                    "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
                    "Path to the BERT TF-Hub module.")

flags.DEFINE_bool(
    "factored_model", True, "Whether to factor the pair sequence"
    "models into ones using two independent BERT encodings.")

flags.DEFINE_integer(
    "max_seq_length", 260,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 30,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_string("answers_path", None, "Path to answers sstable.")

flags.DEFINE_bool(
    "single_per_query", True,
    "Whether to construct a single example per query for training.")

flags.DEFINE_integer(
    "train_records_per_query", 1,
    "The number of training records (each with a fixed number of candidates).")

flags.DEFINE_integer(
    "max_neighbors", 100,
    "The maximum number of neighbors to consider from the retrieved passages.")

flags.DEFINE_bool("add_gold_to_eval", False, "whether to add gold for the dev"
                  "and test set.")
flags.DEFINE_float("fraction_dev", 0, "fraction of examples in dev.")
flags.DEFINE_float("fraction_test", 0, "fraction of examples in test.")

flags.DEFINE_bool("add_random", True, "whether to add random negatives.")

flags.DEFINE_string("record_suffix", "",
                    "a suffix to append to each tf-record file.")
flags.DEFINE_integer(
    "rand_seed", 34567,
    "a random seed to controll the random negatives selection.")

FLAGS = flags.FLAGS

QueryInfo = collections.namedtuple("QueryInfo", ["id", "text"])

PassageInfo = collections.namedtuple("PassageInfo",
                                     ["id", "text", "label", "score"])


def have_neighbors(q_id, neighbors):
  return q_id in neighbors


def get_tokenization_info(module_handle):
  with tf.Graph().as_default():
    bert_module = hub.Module(module_handle)
    with tf.Session() as sess:
      return sess.run(bert_module(signature="tokenization_info", as_dict=True))


def get_tokenizer(module_handle):
  tokenization_info = get_tokenization_info(module_handle)
  return tokenization.FullTokenizer(
      vocab_file=tokenization_info["vocab_file"],
      do_lower_case=tokenization_info["do_lower_case"])


def augmented_neighbors_list(q_id,
                             neighbors,
                             is_training,
                             processor,
                             train_eval=False):
  """Retrieve and convert the neighbors to a list.

  Args:
    q_id: a question id
    neighbors: a table mapping q_id to a list of top candidates
    is_training: True for training set examples
    processor: Helper object
    train_eval: If this is on, we have a sub-set of the training set for which
      we don't add the gold answer if it is not in the neighbors list

  Returns:
     lists of passage ids, list of corresponding labels, list of scores,
     and the index of the first random negative


  """
  n_pb = neighbors[q_id]
  n_list = []
  n_labels = []
  n_scores = []  # the higher, the better

  n_positive = 0
  answers = processor.get_answers(q_id)

  for n in range(len(n_pb)):
    if n >= FLAGS.max_neighbors:
      break  # ignore any later neighbors
    next_n = n_pb[n]

    if processor.answer_match(q_id, next_n[0], answers):
      n_list.append(next_n[0])
      n_labels.append(1)
      n_scores.append(-next_n[1])
      n_positive += 1

    else:
      # see if we keep it
      n_list.append(next_n[0])
      n_labels.append(0)
      n_scores.append(-next_n[1])
  if not n_positive:
    if (is_training or FLAGS.add_gold_to_eval):
      gold_p_id = processor.get_gold_passage_id(q_id)
      if gold_p_id is None and is_training:
        print("Did not find answer matches.")
        return [], [], [], 0
      if gold_p_id is not None:
        n_list.append(gold_p_id)
        n_labels.append(1)
        prior_gold = 0
        n_scores.append(prior_gold)
        n_positive += 1
    else:
      if is_training:
        print("Did not find answer matches.")
        return [], [], [], 0

  # add the same number of random examples as we have neighbors
  # we should add about
  # (FLAGS.num_candidates -1) * FLAGS. train_records_per_query/2 random
  index_rand_start = len(n_list)
  num_random = index_rand_start
  if is_training and not train_eval:  # getting fewer random for speed
    num_random = (int)(
        (FLAGS.num_candidates - 1) * FLAGS.train_records_per_query / 2)

  if FLAGS.add_random:
    random_passages = processor.get_random(num_random)
    random_labels = []
    random_scores = [0] * num_random
    for r in range(len(random_passages)):
      n_scores.append(random_scores[r])
      if processor.answer_match(q_id, random_passages[r], answers):
        random_labels.append(1)
      else:
        random_labels.append(0)
    n_list.extend(random_passages)
    n_labels.extend(random_labels)

  return n_list, n_labels, n_scores, index_rand_start


def sub_select_examples(n_list, n_labels, n_scores, index_rand_start,
                        is_training):
  """Select a sub-set of the examples if we have a limit on examples."""
  # if we are selecting negatives for training , we need to have about
  num_negatives_total = (FLAGS.num_candidates -
                         1) * FLAGS.train_records_per_query
  # we need to select this many negatives out of the total
  positives = []
  first_stage_negatives = []
  rand_negatives = []
  index = 0
  for p_id, label, score in zip(n_list, n_labels, n_scores):
    if label > 0:
      positives.append((p_id, label, score))
    else:
      if index < index_rand_start:
        first_stage_negatives.append((p_id, label, score))
      else:
        rand_negatives.append((p_id, label, score))
    index += 1

  if not is_training:
    first_stage_negatives.extend(rand_negatives)
    return positives, first_stage_negatives

  # now sub-select negatives from the first-stage ones
  n_negatives_select = num_negatives_total - len(rand_negatives)
  selected_negatives_first = select_random(first_stage_negatives,
                                           n_negatives_select)
  # we need to make lists of the positives and an interleaved merge of the
  # negatives
  merged_negatives = []
  index_rand = 0
  for elem in selected_negatives_first:
    merged_negatives.append(elem)
    if index_rand < len(rand_negatives):
      merged_negatives.append(rand_negatives[index_rand])
      index_rand += 1
  return positives, merged_negatives


def select_random(items_list, n_to_select):
  """Selecting the given number of items at random."""
  selected = set()
  if n_to_select >= len(items_list):
    return items_list

  while len(selected) < n_to_select:
    j = random.randint(0, len(items_list) - 1)
    if items_list[j] not in selected:
      selected.add(items_list[j])

  return selected


def passage_texts(pid_list, passages):
  result = []
  for pid in pid_list:
    result.append(passages[pid])
  return result


def get_table(table_path):
  with open(table_path) as fid:
    return json.load(fid)


def generate_samples_v2(queries,
                        passages,
                        neighbors_path,
                        is_training,
                        processor,
                        q_ids,
                        train_eval=False):
  """Enum. (query, List<passage>, List<passage>, List<label>, List<score>)."""

  n_processed = 0
  n_found = 0
  seq_qids = set(q_ids)
  with open(neighbors_path) as fid:
    neighbors = json.load(fid)

  for q_id, q_text in queries.items():
    # get the retrieved neighbors
    if q_id not in seq_qids:
      continue
    if n_processed % 10000 == 0:
      print("processed " + str(n_processed) + " found " + str(n_found))
    n_processed = n_processed + 1
    if not have_neighbors(q_id, neighbors):
      print("neighbors for " + str(q_id) + " not found.")
      continue
    cand_list_ids, cand_list_labels, cand_list_scores, rand_start \
        = augmented_neighbors_list(
            q_id, neighbors, is_training, processor, train_eval=train_eval)
    # get the text of the neigbors
    if not cand_list_ids:
      continue
    if 1 in cand_list_labels:
      n_found = n_found + 1
    # sub-select

    positive, negative = sub_select_examples(cand_list_ids, cand_list_labels,
                                             cand_list_scores, rand_start,
                                             is_training)

    # get information for the positive passages

    passage_infos_positive = wrap_with_passage_texts(positive, passages)
    passage_infos_negative = wrap_with_passage_texts(negative, passages)
    yield QueryInfo(q_id, q_text), passage_infos_positive, \
        passage_infos_negative


def wrap_with_passage_texts(cand_triples, passages):
  """Add passage text to the records."""
  cand_list_ids = [x[0] for x in cand_triples]
  cand_texts = passage_texts(cand_list_ids, passages)
  passage_infos = []
  for i in range(len(cand_list_ids)):
    cid = cand_list_ids[i]
    cl = cand_triples[i][1]
    ctext = cand_texts[i]
    cscore = cand_triples[i][2]
    passage_infos.append(PassageInfo(cid, ctext, cl, cscore))
  return passage_infos


class InputExample(object):
  """A single training/test example for ranking."""

  def __init__(self,
               guid,
               text_a,
               text_b_list=None,
               text_b_guids=None,
               labels=None,
               scores=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. This is the
        query.
      text_b_list : list of string. A list of the untokenized texts of the
        second sequences. Only must be specified for sequence pair tasks.
      text_b_guids: indices of candidate passages.
      labels: (Optional) list of string. The relevance of each instance. This
        should be specified for train and dev examples, but not for test
        examples.
      scores: list of floats of scores according to a first-pass model
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b_list = text_b_list
    self.text_b_guids = text_b_guids
    self.labels = labels
    self.scores = scores


class InputFeatures(object):
  """A single set of features of data which are for lists of paired candidates."""

  def __init__(self, unique_ids, input_ids, input_masks, segment_ids, label_ids,
               scores):
    self.input_ids = input_ids
    self.input_masks = input_masks
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.unique_ids = unique_ids  # the unique ids are strings qid-pid
    self.scores = scores


def _truncate_seq(tokens_a, max_length):
  """Truncates a sequence  in place to the maximum length."""

  # This is a simple heuristic which will always truncate the sequence
  # one token at a time.
  while True:
    total_length = len(tokens_a)
    if total_length <= max_length:
      break
    tokens_a.pop()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the second sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    tokens_b.pop()


def convert_single_example(ex_index, example, max_seq_length, tokenizer,
                           label_list):
  """Converts a single `InputExample` into one or more `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  _truncate_seq(tokens_a, FLAGS.max_query_length)

  assert (len(example.text_b_list)) == FLAGS.num_candidates
  tokens_b_list = []
  for text_b in example.text_b_list:
    tokens_b_list.append(tokenizer.tokenize(text_b))
  for tokens_b in tokens_b_list:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  all_segment_ids = []
  all_input_ids = []
  all_input_mask = []
  all_unique_ids = []
  all_label_ids = []
  all_tokens = []
  for cand_index, tokens_b in enumerate(tokens_b_list):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = int(example.labels[cand_index])
    all_input_ids.extend(input_ids)
    all_input_mask.extend(input_mask)
    all_unique_ids.append(example.guid + "-" + example.text_b_guids[cand_index])
    all_label_ids.append(label_id)
    all_segment_ids.extend(segment_ids)
    all_tokens.extend(tokens)
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info(
        "tokens: %s" %
        " ".join([tokenization.printable_text(x) for x in all_tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in all_input_ids]))
    tf.logging.info("input_mask: %s" %
                    " ".join([str(x) for x in all_input_mask]))
    tf.logging.info("segment_ids: %s" %
                    " ".join([str(x) for x in all_segment_ids]))
    tf.logging.info("labels ids: %s" %
                    " ".join([str(x) for x in all_label_ids]))
    tf.logging.info("labels str: %s" %
                    " ".join([str(x) for x in example.labels]))
    tf.logging.info("prior scores: %s" %
                    " ".join([str(x) for x in example.scores]))

  feature = InputFeatures(
      unique_ids=all_unique_ids,
      input_ids=all_input_ids,
      input_masks=all_input_mask,
      segment_ids=all_segment_ids,
      label_ids=all_label_ids,
      scores=example.scores)
  return feature


def convert_single_example_dual(ex_index, example, max_seq_length, tokenizer,
                                label_list):
  """Converts a single `InputExample` into one or more `DualInputFeatures`.

  Args:
    ex_index: index of the example
    example: InputExample
    max_seq_length: maximal sequence length of query + one passage in tokens
    tokenizer: mapping a string to a list of tokens
    label_list: labls for all passage candidates

  Returns:
    DualInputFeatures representing the records

  This flattens out the tokens of all candidates in a single sequence for the
  passages
  The query is only listed once and has form [CLS] q [SEP]
  Passages are as many as candidates and have form [CLS] p [SEP]
  """
  max_seq_length_passage = max_seq_length - FLAGS.max_query_length
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  _truncate_seq(tokens_a, FLAGS.max_query_length - 2)

  assert (len(example.text_b_list)) == FLAGS.num_candidates
  tokens_b_list = []
  for text_b in example.text_b_list:
    tokens_b_list.append(tokenizer.tokenize(text_b))
  for tokens_b in tokens_b_list:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
    _truncate_seq(tokens_b, max_seq_length_passage - 2)

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  all_segment_ids = []
  all_input_ids = []
  all_input_mask = []
  all_unique_ids = []
  all_label_ids = []
  all_tokens = []
  query_tokens = []
  query_input_ids = []
  query_input_mask = []
  query_segment_ids = []
  # first process the query tokens
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  while len(input_ids) < FLAGS.max_query_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == FLAGS.max_query_length
  assert len(input_mask) == FLAGS.max_query_length
  assert len(segment_ids) == FLAGS.max_query_length
  # copy to query variables
  query_tokens.extend(tokens)
  query_input_ids.extend(input_ids)
  query_segment_ids.extend(segment_ids)
  query_input_mask.extend(input_mask)

  for cand_index, tokens_b in enumerate(tokens_b_list):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(1)
    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length_passage:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(1)

    assert len(input_ids) == max_seq_length_passage
    assert len(input_mask) == max_seq_length_passage
    assert len(segment_ids) == max_seq_length_passage

    label_id = int(example.labels[cand_index])
    all_input_ids.extend(input_ids)
    all_input_mask.extend(input_mask)
    all_unique_ids.append(example.guid + "-" + example.text_b_guids[cand_index])
    all_label_ids.append(label_id)
    all_segment_ids.extend(segment_ids)
    all_tokens.extend(tokens)
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    # query info
    tf.logging.info(
        "query tokens: %s" %
        " ".join([tokenization.printable_text(x) for x in query_tokens]))
    tf.logging.info("query input_ids: %s" %
                    " ".join([str(x) for x in query_input_ids]))
    tf.logging.info("query input_mask: %s" %
                    " ".join([str(x) for x in query_input_mask]))
    tf.logging.info("query segment_ids: %s" %
                    " ".join([str(x) for x in query_segment_ids]))
    tf.logging.info(
        "tokens: %s" %
        " ".join([tokenization.printable_text(x) for x in all_tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in all_input_ids]))
    tf.logging.info("input_mask: %s" %
                    " ".join([str(x) for x in all_input_mask]))
    tf.logging.info("segment_ids: %s" %
                    " ".join([str(x) for x in all_segment_ids]))
    tf.logging.info("labels ids: %s" %
                    " ".join([str(x) for x in all_label_ids]))
    tf.logging.info("prior scores: %s" %
                    " ".join([str(x) for x in example.scores]))
    tf.logging.info("labels str: %s" %
                    " ".join([str(x) for x in example.labels]))

  feature = DualInputFeatures(
      input_ids_1=query_input_ids,
      input_mask_1=query_input_mask,
      segment_ids_1=query_segment_ids,
      input_ids_2=all_input_ids,
      input_masks_2=all_input_mask,
      segment_ids_2=all_segment_ids,
      label_ids=all_label_ids,
      unique_ids=all_unique_ids,
      scores=example.scores)
  return feature


class DualInputFeatures(object):
  """A single set of features of data which are for query and a cand list."""

  def __init__(self, unique_ids, input_ids_1, input_mask_1, segment_ids_1,
               input_ids_2, input_masks_2, segment_ids_2, label_ids, scores):
    self.input_ids_1 = input_ids_1
    self.input_mask_1 = input_mask_1
    self.segment_ids_1 = segment_ids_1
    self.input_ids_2 = input_ids_2
    self.input_masks_2 = input_masks_2
    self.segment_ids_2 = segment_ids_2
    self.label_ids = label_ids
    self.unique_ids = unique_ids
    self.scores = scores


class RetrievalProcessor():
  """Processor for a retrieval dataset to tokenize."""

  def __init__(self,
               answers_path,
               queries_table,
               passages_table,
               encoded_queries_path=None,
               encoded_passages_path=None):
    self.answers_table = None
    self.passages_table = passages_table
    self.queries_table = queries_table
    self.num_passages = len(passages_table)
    self.encoded_queries_path = encoded_queries_path
    self.encoded_passages_path = encoded_passages_path
    print("Num passages " + str(self.num_passages))
    self.answers_table = get_table(answers_path)
    self.init_tables()

  def init_tables(self):

    if self.encoded_queries_path and self.encoded_passages_path:
      # we will need the tables of encoded elements
      self.encoded_queries = get_table(self.encoded_queries_path)
      self.encoded_passages = get_table(self.encoded_passages_path)

  def get_random(self, num_random):
    to_return = []
    for _ in range(num_random):
      to_return.append(str(random.randint(0, self.num_passages - 1)))
    return to_return

  def get_oracle_neighbors(self, q_id):
    """Look-up the neighbors in the oracle table."""
    result = []
    if not self.oracle_neighbors:
      return result
    neighbors = self.oracle_neighbors[q_id]

    for n in range(len(neighbors)):
      next_n = neighbors[n]
      result.append(next_n[0])
    return result

  def dict_dot_product(self, dict1, dict2):
    result = 0.0
    for k, v in dict1.iteritems():
      if k not in dict2:
        continue
      result += v * dict2[k]
    return result

  def get_gold_passage_id(self, q_key):
    """Get the id of a gold passage."""
    answers = self.answers_table[str(q_key)].split("\n")
    return answers[0]

  def get_answers(self, q_key):
    if str(q_key) not in self.answers_table:
      print("No answers")
      return []
    else:
      return self.answers_table[str(q_key)].split("\n")

  def answer_match(self, q_key, p_key, answers=None):
    """Compute whether the query matches the passage."""
    if not answers:
      if str(q_key) not in self.answers_table:
        answers = []
      else:
        answers = self.answers_table[str(q_key)].split("\n")
    return p_key in answers

  def emit(self, query_info, passage_infos):
    """Make an InputExample from the query+caniddates."""
    text_b_list = []
    text_b_guids = []
    labels = []
    scores = []
    for p in passage_infos:
      text_b_list.append(p.text)
      text_b_guids.append(p.id)
      labels.append(float(p.label))
      scores.append(float(p.score))

    return InputExample(query_info.id, query_info.text, text_b_list,
                        text_b_guids, labels, scores)

  @staticmethod
  def get_index_correct(passage_infos):
    for i, p in enumerate(passage_infos):
      if int(p.label) == 1:
        return i
    return -1

  def get_examples_v2(self, is_train, max_cand, query_info,
                      passage_infos_positive, passage_infos_negative):
    """Generating examples by grouping the canidates into records."""
    examples = []
    if is_train and not passage_infos_positive:
      return examples
    for input_example in self.get_examples_v2_iterate(is_train, max_cand,
                                                      query_info,
                                                      passage_infos_positive,
                                                      passage_infos_negative):
      examples.append(input_example)
    return examples

  def get_examples_v2_iterate(self, is_train, max_cand, query_info,
                              passage_infos_positive, passage_infos_negative):
    """Get examples as a sequence of grouped candidates."""
    # we need to iterate over the candidates in groups of max_cand
    # for training, each example needs to have a positive and we have
    # a limit of train_records_per_query
    # for eval, we just list them out in any order
    if is_train:
      random.shuffle(passage_infos_positive)
      random.shuffle(passage_infos_negative)
      n_available = len(passage_infos_negative)
      n_instances = FLAGS.train_records_per_query
      if n_instances * (max_cand - 1) > n_available:
        n_instances = (int)(n_available / (max_cand - 1))
        print("reduced candidates " + str(n_instances))
      # start generating the instances
      instances = []
      for _ in range(n_instances):
        # initialize with empty lists
        instances.append([])
      # first add the positives
      pos_index = 0
      max_pos_index = 0  # the maximum index of a positive element added
      for i in range(n_instances):
        instances[i].append(passage_infos_positive[pos_index])
        if pos_index > max_pos_index:
          max_pos_index = pos_index
        pos_index += 1
        if pos_index >= len(passage_infos_positive):
          pos_index = 0

      # see if we have posititives left over
      if max_pos_index < len(passage_infos_positive) - 1:
        # there are some positives that have not been added
        print("Extra positives! " + str(len(passage_infos_positive)))
        have_room = True
        n_added = 0
        pos_index = max_pos_index + 1
        while have_room and pos_index < len(passage_infos_positive):
          for i in range(n_instances):
            if len(instances[i]) > max_cand * .5:
              have_room = False
              break
            instances[i].append(passage_infos_positive[pos_index])
            n_added += 1
            pos_index += 1
            if pos_index >= len(passage_infos_positive):
              break
        print("Total positives added " + str(pos_index) + " out of " +
              str(len(passage_infos_positive)))

      # positives have been added, fill in negatives
      pos_index = 0
      for i in range(n_instances):
        n_needed = max_cand - len(instances[i])
        instances[i].extend(passage_infos_negative[pos_index:pos_index +
                                                   n_needed])
        yield self.emit(query_info, instances[i])
        pos_index += n_needed

    else:
      # this is for eval, just emit all of them
      rotate_group = []
      rotate_group.extend(passage_infos_positive)
      rotate_group.extend(passage_infos_negative)
      start = 0
      while start < len(rotate_group):
        end = start + max_cand
        if end > len(rotate_group):
          move = end - len(rotate_group)
          start = start - move
          end = end - move
        passages = rotate_group[start:end]
        yield self.emit(query_info, passages)
        start = end

  def group_iterate(self,
                    query_info,
                    passage_list,
                    max_cand,
                    is_training,
                    train_eval=False):
    """Iterates over the passage candidates in groups of specified number."""
    # first shuffle the list
    if is_training:
      random.shuffle(passage_list)  # These are PassageInfos
    index_correct = RetrievalProcessor.get_index_correct(passage_list)

    rotate_group = passage_list
    # for training this becomes all passages except for one singled as correct

    if is_training:
      rotate_group = []
      max_cand = max_cand - 1
      for (i, p) in enumerate(passage_list):
        if i == index_correct:
          continue
        rotate_group.append(p)
    start = 0

    while start < len(rotate_group):
      end = start + max_cand
      if end > len(rotate_group):
        move = end - len(rotate_group)
        start = start - move
        end = end - move

      passages = rotate_group[start:end]

      if is_training:
        passages.insert(0, passage_list[index_correct])

      yield self.emit(query_info, passages)
      if is_training and FLAGS.single_per_query and not train_eval:
        break  # we want each query to appear only once in training except when
        # we generate a special set for training eval
      start = end

  def get_labels(self):
    """See base class."""
    return ["0", "1"]


def generate_split(queries_path, fraction_dev, fraction_test):
  """Place queries in sets of training, dev, test, and heldout train.

  Args:
     queries_path : str name of file containing json table of queries
     fraction_dev: str 1 iff exporting dev set, otherwise 0
     fraction_test : str 1 iff exporting test set, otherwise 0

  Returns:
     sets of train, dev, and test queries

  """
  train_qids = {}
  dev_qids = {}
  test_qids = {}
  train_eval_qids = {}
  queries = get_table(queries_path)
  answers = get_table(FLAGS.answers_path)
  num_examples = 0
  n_queries = len(queries)
  assert (fraction_dev == 0 or fraction_dev == 1)
  assert (fraction_test == 0 or fraction_test == 1)
  assert fraction_test * fraction_dev == 0

  # actually iterate over the neighbors
  print("Splitting " + str(n_queries) + "queries")
  for q_id, _ in queries.items():

    num_examples = num_examples + 1

    if fraction_dev == 1:
      dev_qids[q_id] = 1
      continue
    if fraction_test == 1:
      test_qids[q_id] = 1
      continue

    if q_id in answers:
      train_qids[q_id] = 1

  tf.logging.info(
      "Created examples train {} eval {} test {} train eval {}".format(
          len(train_qids), len(dev_qids), len(test_qids), len(train_eval_qids)))
  return train_qids, dev_qids, test_qids, train_eval_qids


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file):
  if not FLAGS.factored_model:
    return file_based_convert_examples_to_features_single(
        examples, label_list, max_seq_length, tokenizer, output_file)
  else:
    return file_based_convert_examples_to_features_dual(examples, label_list,
                                                        max_seq_length,
                                                        tokenizer, output_file)


def file_based_convert_examples_to_features_single(examples, label_list,
                                                   max_seq_length, tokenizer,
                                                   output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  ex_index = 0
  for example in examples:
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example " + str(ex_index))

    feature = convert_single_example(ex_index, example, max_seq_length,
                                     tokenizer, label_list)
    ex_index = ex_index + 1

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()

    p_ids = []
    q_ids = []

    for qp in feature.unique_ids:
      elems = qp.split("-")
      q_id = int(elems[0])
      p_id = int(elems[1])
      p_ids.append(p_id)
      q_ids.append(q_id)

    features["q_ids"] = create_int_feature(q_ids)
    features["cand_nums"] = create_int_feature(p_ids)
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_masks)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["scores"] = create_float_feature(feature.scores)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
  print("Wrote " + str(ex_index) + " examples " + output_file)


def file_based_convert_examples_to_features_dual(examples, label_list,
                                                 max_seq_length, tokenizer,
                                                 output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)
  num_written = 0

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example " + str(ex_index))
    num_written = num_written + 1

    feature = convert_single_example_dual(ex_index, example, max_seq_length,
                                          tokenizer, label_list)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    p_ids = []
    q_ids = []

    for qp in feature.unique_ids:
      elems = qp.split("-")
      q_id = int(elems[0])
      p_id = int(elems[1])
      p_ids.append(p_id)
      q_ids.append(q_id)
    features["q_ids"] = create_int_feature(q_ids)
    features["cand_nums"] = create_int_feature(p_ids)
    features["input_ids_1"] = create_int_feature(feature.input_ids_1)
    features["input_mask_1"] = create_int_feature(feature.input_mask_1)
    features["segment_ids_1"] = create_int_feature(feature.segment_ids_1)
    features["input_ids_2"] = create_int_feature(feature.input_ids_2)
    features["input_masks_2"] = create_int_feature(feature.input_masks_2)
    features["segment_ids_2"] = create_int_feature(feature.segment_ids_2)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["scores"] = create_float_feature(feature.scores)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  print("Wrote " + str(num_written) + " in " + output_file)


def save_tf_examples(queries_path, passages_path, neighbors_path, examples_path,
                     max_cand):
  """Saves examples as TFRecords."""
  examples_path = os.path.join(examples_path, str(FLAGS.factored_model))
  os.makedirs(examples_path, exist_ok=True)

  train_qids, dev_qids, test_qids, train_eval_qids = generate_split(
      queries_path, FLAGS.fraction_dev, FLAGS.fraction_test)
  suffix = ""

  shard_suffix = ".tf_record"
  shard_suffix = FLAGS.record_suffix + shard_suffix
  train_file = os.path.join(examples_path, suffix + "train" + shard_suffix)
  dev_file = os.path.join(examples_path, suffix + "eval" + shard_suffix)
  test_file = os.path.join(examples_path, suffix + "test" + shard_suffix)
  train_eval_file = os.path.join(examples_path,
                                 suffix + "train_eval" + shard_suffix)

  queries = get_table(queries_path)
  passages = get_table(passages_path)
  if train_qids:
    print("Saving train examples in " + train_file)
    save_examples(queries, passages, neighbors_path, train_qids, max_cand,
                  train_file, True)
  if dev_qids:
    print("Saving dev examples in " + dev_file)
    save_examples(queries, passages, neighbors_path, dev_qids, max_cand,
                  dev_file, False)
  if test_qids:
    print("Saving test examples in " + test_file)
    save_examples(queries, passages, neighbors_path, test_qids, max_cand,
                  test_file, False)
  if train_eval_qids:
    print("Saving train eval examples in " + train_eval_file + " number " +
          str(len(train_eval_qids)))
    save_examples(
        queries,
        passages,
        neighbors_path,
        train_eval_qids,
        max_cand,
        train_eval_file,
        True,
        train_eval=True)


def get_grouped_candidates_v2(processor,
                              queries,
                              passages,
                              neighbors_path,
                              q_ids,
                              max_cand,
                              is_training,
                              train_eval=False):
  """For each query, selecting positive and negative candidates."""
  for q_info, p_infos_positive, p_infos_negative in generate_samples_v2(
      queries,
      passages,
      neighbors_path,
      is_training,
      processor,
      q_ids,
      train_eval=train_eval):
    if q_info.id not in q_ids:
      continue
    for example in processor.get_examples_v2(is_training, max_cand, q_info,
                                             p_infos_positive,
                                             p_infos_negative):
      yield example


def save_examples(queries,
                  passages,
                  neighbors_path,
                  q_ids,
                  max_cand,
                  examples_path,
                  is_training,
                  train_eval=False):
  """Save examples for the given q_ids as TFRecords."""

  encoded_passages_path = None
  encoded_queries_path = None
  encoded_passages_path = FLAGS.encoded_passages_path
  encoded_queries_path = FLAGS.encoded_queries_path
  processor = RetrievalProcessor(FLAGS.answers_path, queries, passages,
                                 encoded_queries_path, encoded_passages_path)
  tokenizer = get_tokenizer(FLAGS.bert_hub_module_path)
  max_seq_length = FLAGS.max_seq_length
  function = get_grouped_candidates_v2

  examples = function(
      processor,
      queries,
      passages,
      neighbors_path,
      q_ids,
      max_cand,
      is_training,
      train_eval=train_eval)
  file_based_convert_examples_to_features(examples, processor.get_labels(),
                                          max_seq_length, tokenizer,
                                          examples_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  print("Saving examples.")
  random.seed(FLAGS.rand_seed)  # setting random seed for stability
  save_tf_examples(FLAGS.queries_path, FLAGS.passages_path,
                   FLAGS.neighbors_path, FLAGS.examples_path,
                   FLAGS.num_candidates)
  return


if __name__ == "__main__":
  tf.disable_eager_execution()
  app.run(main)
