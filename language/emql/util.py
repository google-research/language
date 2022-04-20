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
"""Utility functions for EmQL models.
"""


from bert import tokenization
import numpy as np
import tensorflow.compat.v1 as tf

app = tf.app
DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 128

VOCAB_FILE = './preprocess/bert_vocab.txt'



def topk_search_fn(query_embs, embeddings_mat,
                   k):
  """Compute topk of inner product search.

  Args:
    query_embs: batch_size, hidden_size
    embeddings_mat: num_vocab, hidden_size
    k: top k

  Returns:
    topk_fact_logits, topk_fact_ids
  """
  fact_logits = tf.matmul(
      query_embs, embeddings_mat, transpose_b=True)
  # batch_size, num_facts
  fact_logits /= tf.sqrt(tf.cast(tf.shape(query_embs)[1], tf.float32))
  # batch_size, num_facts
  topk_fact_logits, topk_fact_ids = tf.nn.top_k(fact_logits, k=k)
  # batch_size, intermediate_top_k
  return topk_fact_logits, topk_fact_ids


def compute_hits_at_k(
    logits, labels, k):
  """Compute hits@1.

  Args:
    logits: batch_size, num_candidate
    labels: batch_size, num_candidate
    k: top k

  Returns:
    hits@k
  """
  assert logits.shape.as_list() == labels.shape.as_list()
  num_dimensions = len(logits.shape)
  _, topk = tf.nn.top_k(logits, k=k)
  hits_at_k = tf.reduce_sum(
      tf.gather(labels, topk, batch_dims=num_dimensions - 1),
      axis=1) > 0
  hits_at_k = tf.cast(hits_at_k, tf.float32)
  return hits_at_k


def compute_recall_at_k(
    logits, labels, k):
  """Compute hits@1.

  Args:
    logits: batch_size, num_candidate
    labels: batch_size, num_candidate
    k: top k

  Returns:
    recall at k
  """
  assert logits.shape.as_list() == labels.shape.as_list()
  num_dimensions = len(logits.shape)
  _, topk = tf.nn.top_k(logits, k=k)
  recall_at_k = tf.reduce_sum(
      tf.gather(labels, topk, batch_dims=num_dimensions - 1),
      axis=1) / tf.reduce_sum(labels, axis=1)
  return recall_at_k


def compute_average_precision_at_k(
    logits, labels, k):
  """Compute average precision at k.

  Args:
    logits: batch_size, nun_candidate
    labels: batch_size, num_candidate
    k: scalar

  Returns:
    average_precision_at_k
  """

  _, topk = tf.nn.top_k(logits, k)
  # batch_size, k
  true_positives = tf.gather(labels, topk, batch_dims=1)
  # batch_size, k
  # e.g. [[0, 1, 1, 0], [1, 0, 0, 1]]
  upper_triangle_matrix = tf.constant(
      np.triu(np.ones([k, k])), dtype=tf.float32)
  # k, k
  # e.g. [[1,1,1,1], [0,1,1,1], [0,0,1,1], [0,0,0,1]]
  upper_triangle_matrix /= tf.reduce_sum(
      upper_triangle_matrix, axis=0, keepdims=True)
  # k, k
  # e.g. [[1,1/2,1/3,1/4], [0,1/2,1/3,1/4], [0,0,1/3,1/4], [0,0,0,1/4]]
  recall_at_k = tf.matmul(true_positives, upper_triangle_matrix)
  # batch_size, k
  # e.g. [[0, 1/2, 2/3, 2/4], [1, 1/2, 1/3, 2/4]]
  positive_recall_at_k = true_positives * recall_at_k
  # batch_size, k
  # e.g. [[0, 1/2, 2/3, 0], [1, 0, 0, 2/4]]
  num_true_positive = tf.reduce_sum(true_positives, axis=1)
  # batch_size
  # e.g. [2, 2]
  num_true_positive_replace_0_with_1 = tf.where(
      num_true_positive > 0, num_true_positive,
      tf.ones(tf.shape(num_true_positive), dtype=tf.float32))
  average_precision_at_k = \
      tf.reduce_sum(positive_recall_at_k, axis=1) \
      / num_true_positive_replace_0_with_1
  # batch_size
  # e.g. [(1/2 + 2/3) / 2, (1 + 2/4) / 2]
  return average_precision_at_k


def get_nonzero_ids(v_hot, k):
  """Get k ids of nonzero values in the v_hot vector.

  If less than k nonzero values exist, pad with -1.

  Args:
    v_hot: v hot vector (batch_size, dim)
    k: top k

  Returns:
    k ids of nonzero values
  """
  if k == -1:
    k = tf.shape(v_hot)[-1]
  label_k, idx_k = tf.nn.top_k(v_hot, k)
  idx_k = tf.where(
      tf.not_equal(label_k, 0), idx_k, tf.fill(tf.shape(idx_k), value=-1))
  return idx_k


def embedding_lookup_with_padding(embeddings_mat,
                                  ids,
                                  padding = -1):
  """Wrapper of tf.nn.embedding_lookup with padding.

  Args:
    embeddings_mat: embedding table
    ids: ids of elements
    padding: padded value

  Returns:
    embeddings of elements
  """
  mask = tf.cast(tf.not_equal(ids, padding), dtype=tf.int32)
  # batch_size,
  embs = tf.nn.embedding_lookup(embeddings_mat, ids * mask)
  # batch_size, hidden_size
  mask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)
  # batch_size, 1
  embs *= mask
  # batch_size, hidden_size
  return embs


def compute_x_in_set(x, s):
  """Check if elements in tensor x are in set s.

  Args:
    x: batch_size, num_candidate
    s: batch_size, k (padded with -1)

  Returns:
    boolean tensor with shape of x
  """
  s = tf.expand_dims(s, axis=2)
  # batch_size, k, 1
  k = tf.shape(s)[1]
  x = tf.tile(tf.expand_dims(x, axis=1), [1, k, 1])
  # batch_size, k, num_candidate
  x_to_s_map = tf.equal(x, s)
  # batch_size, k, num_candidate
  x_in_s = tf.reduce_any(x_to_s_map, axis=1)
  # batch_size, num_candidate
  return x_to_s_map, x_in_s


def get_var_shape(checkpoint_dir, var_name):
  checkpoint_var2shape = {
      name: shape for name, shape in tf.train.list_variables(checkpoint_dir)
  }
  return checkpoint_var2shape[var_name]


def load_db_checkpoint(var_name,
                       checkpoint_dir,
                       dtype = tf.float32,
                       cpu = True,
                       trainable = False,
                       is_local_variable = True):
  """Load weights for variables from checkpoint.

  Args:
    var_name: variable name
    checkpoint_dir: checkpoint dir
    dtype: data type
    cpu: if load to cpu or not
    trainable: train loaded variable or not
    is_local_variable: local or global variable

  Returns:
    variables loaded with weights from checkpoint
  """
  shape = get_var_shape(checkpoint_dir, var_name)
  get_variable_func = (
      tf.get_local_variable if is_local_variable else tf.get_variable)

  if cpu:
    with tf.device('/cpu:0'):
      tf_db = get_variable_func(
          var_name,
          shape=shape,
          dtype=dtype,
          trainable=trainable,
          use_resource=True)
  else:
    tf_db = get_variable_func(
        var_name,
        shape=shape,
        dtype=dtype,
        trainable=trainable,
        use_resource=True)

  assignment_map = {var_name: tf_db}
  tf.train.init_from_checkpoint(checkpoint_dir, assignment_map)

  return tf_db


class BertTokenizer(object):
  """Tokenizer for BERT.

  Returned token_ids, seg_ids, and input_mask are padded to max_seq_length.

  """

  def __init__(self,
               vocab_file = VOCAB_FILE,
               max_seq_length = MAX_SEQ_LENGTH):
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=DO_LOWER_CASE)
    self.max_seq_length = max_seq_length

  def tokenize(self, text):
    """Tokenize text.

    Args:
      text: text in natural language

    Returns:
      a list of tokens, segment_ids, and input mask
    """

    text = tokenization.convert_to_unicode(text)
    tokens = self.tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    seg_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)
    assert self.max_seq_length > len(tokens)
    pad_len = self.max_seq_length - len(tokens)
    token_ids = np.array(token_ids + [0] * pad_len, dtype='int32')
    seg_ids = np.array(seg_ids + [0] * pad_len, dtype='int32')
    input_mask = np.array(input_mask + [0] * pad_len, dtype='int32')
    return tokens, (token_ids, seg_ids, input_mask)
