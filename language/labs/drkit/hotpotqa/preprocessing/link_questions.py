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
"""Command-line version of the wiki neighbors demo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import urllib.parse

from absl import app
from absl import flags
from absl import logging
from bert import tokenization
from language.labs.drkit import search_utils
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import tensorflow.compat.v1 as tf
from tqdm import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string("hotpotqa_file", None, "Path to HotpotQA dataset file.")

flags.DEFINE_string("entity_dir", None,
                    "Path to Entity co-occurrence directory.")

flags.DEFINE_string("vocab_file", None, "Path to vocab for tokenizer.")

flags.DEFINE_string("output_file", None, "Path to Output file.")


def tfidf_linking(questions, base_dir, tokenizer, top_k, batch_size=100):
  """Match questions to entities via Tf-IDF."""
  # Load entity ids and masks.
  tf.reset_default_graph()
  id_ckpt = os.path.join(base_dir, "entity_ids")
  entity_ids = search_utils.load_database(
      "entity_ids", None, id_ckpt, dtype=tf.int32)
  mask_ckpt = os.path.join(base_dir, "entity_mask")
  entity_mask = search_utils.load_database("entity_mask", None, mask_ckpt)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.logging.info("Loading entity ids and masks...")
    np_ent_ids, np_ent_mask = sess.run([entity_ids, entity_mask])
  tf.logging.info("Building entity count matrix...")
  entity_count_matrix = search_utils.build_count_matrix(np_ent_ids, np_ent_mask)

  # Tokenize questions and build count matrix.
  tf.logging.info("Tokenizing questions...")
  ques_toks, ques_masks = [], []
  for question in questions:
    toks = tokenizer.tokenize(question["question"])
    tok_ids = tokenizer.convert_tokens_to_ids(toks)
    ques_toks.append(tok_ids)
    ques_masks.append([1 for _ in tok_ids])
  tf.logging.info("Building question count matrix...")
  question_count_matrix = search_utils.build_count_matrix(ques_toks, ques_masks)

  # Tf-IDF.
  tf.logging.info("Computing IDFs...")
  idfs = search_utils.counts_to_idfs(entity_count_matrix, cutoff=1e-5)
  tf.logging.info("Computing entity Tf-IDFs...")
  ent_tfidfs = search_utils.counts_to_tfidf(entity_count_matrix, idfs)
  ent_tfidfs = normalize(ent_tfidfs, norm="l2", axis=0)
  tf.logging.info("Computing question TF-IDFs...")
  qry_tfidfs = search_utils.counts_to_tfidf(question_count_matrix, idfs)
  qry_tfidfs = normalize(qry_tfidfs, norm="l2", axis=0)
  tf.logging.info("Searching...")
  top_doc_indices = np.empty((len(questions), top_k), dtype=np.int32)
  top_doc_distances = np.empty((len(questions), top_k), dtype=np.float32)
  # distances = qry_tfidfs.transpose().dot(ent_tfidfs)
  num_batches = len(questions) // batch_size
  tf.logging.info("Computing distances in %d batches of size %d",
                  num_batches + 1, batch_size)
  for nb in tqdm(range(num_batches + 1)):
    min_ = nb * batch_size
    max_ = (nb + 1) * batch_size
    if min_ >= len(questions):
      break
    if max_ > len(questions):
      max_ = len(questions)
    distances = qry_tfidfs[:, min_:max_].transpose().dot(ent_tfidfs).tocsr()
    for ii in range(min_, max_):
      my_distances = distances[ii - min_, :].tocsr()
      if len(my_distances.data) <= top_k:
        o_sort = np.argsort(-my_distances.data)
        top_doc_indices[ii, :len(o_sort)] = my_distances.indices[o_sort]
        top_doc_distances[ii, :len(o_sort)] = my_distances.data[o_sort]
        top_doc_indices[ii, len(o_sort):] = 0
        top_doc_distances[ii, len(o_sort):] = 0
      else:
        o_sort = np.argpartition(-my_distances.data, top_k)[:top_k]
        top_doc_indices[ii, :] = my_distances.indices[o_sort]
        top_doc_distances[ii, :] = my_distances.data[o_sort]

  # Load entity metadata and conver to kb_id.
  metadata_file = os.path.join(base_dir, "entities.json")
  entity2id, entity2name = json.load(tf.gfile.Open(metadata_file))
  id2entity = {i: e for e, i in entity2id.items()}
  id2name = {i: entity2name[e] for e, i in entity2id.items()}
  mentions = []
  for ii in range(len(questions)):
    my_mentions = []
    for m in range(top_k):
      my_mentions.append({
          "kb_id": id2entity[top_doc_indices[ii, m]],
          "score": str(top_doc_distances[ii, m]),
          "name": id2name[top_doc_indices[ii, m]],
      })
    mentions.append(my_mentions)

  return mentions


def load_entity_matrices(base_dir):
  """Load entity co-occurrence and co-reference matrices."""
  cooccur_ckpt = os.path.join(base_dir, "ent2ment.npz")
  coref_ckpt = os.path.join(base_dir, "coref.npz")

  tf.reset_default_graph()
  co_data, co_indices, co_rowsplits = search_utils.load_ragged_matrix(
      "ent2ment", cooccur_ckpt)
  coref_map = search_utils.load_database(
      "coref", None, coref_ckpt, dtype=tf.int32)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.logging.info("Loading ragged matrix...")
    np_data, np_indices, np_indptr = sess.run(
        [co_data, co_indices, co_rowsplits])
    tf.logging.info("Loading coref map...")
    np_coref = sess.run(coref_map)
    num_entities = np_indptr.shape[0] - 1
    num_mentions = np_coref.shape[0]
    tf.logging.info("Creating sparse matrix %d x %d...", num_entities,
                    num_mentions)
    sp_cooccur = sp.csr_matrix((np_data, np_indices, np_indptr),
                               shape=(num_entities, num_mentions))
    tf.logging.info("Creating sparse matrix %d x %d...", num_mentions,
                    num_entities)
    sp_coref = sp.csr_matrix((np.ones_like(np_coref, dtype=np.int32),
                              (np.arange(np_coref.shape[0]), np_coref)),
                             shape=(num_mentions, num_entities))

  metadata_file = os.path.join(base_dir, "entities.json")
  entity2id, _ = json.load(tf.gfile.Open(metadata_file))

  return sp_cooccur, sp_coref, entity2id


def evaluate_entity_linking(questions, base_dir, num_hops):
  """Evaluate how often answers can be reached from linked entities."""

  def _check_answers(sp_vec, answers):
    found_ans = np.zeros((len(answers),), dtype=bool)
    for ii, ans in enumerate(answers):
      if sp_vec[0, ans] > 0.:
        found_ans[ii] = True
    return found_ans

  sp_cooccur, sp_coref, entity2id = load_entity_matrices(base_dir)
  num_found_ans = {i: 0. for i in range(num_hops + 1)}
  for ii, question in enumerate(questions):
    if (ii + 1) % 1000 == 0:
      tf.logging.info("Evaluated %d questions...", ii)
    subjects = [
        entity2id.get(ee["kb_id"].lower(), 0) for ee in question["entities"]
    ]
    answers = [
        entity2id.get(ee["kb_id"].lower(), 0)
        for ee in question["supporting_facts"]
    ]
    # Create the initial sparse vector.
    vals = np.ones((len(subjects),), dtype=np.float32)
    vals = vals / vals.sum()
    rows = np.zeros((len(subjects),), dtype=np.int32)
    cols = np.asarray(subjects, dtype=np.int32)
    v_st = sp.csr_matrix((vals, (rows, cols)), shape=(1, len(entity2id)))
    found_ans = _check_answers(v_st, answers)
    if found_ans.all():
      num_found_ans[0] += 1
      continue
    for i in range(num_hops):
      v_m = v_st * sp_cooccur
      v_st = v_m * sp_coref
      # print("then", v_st.getnnz(), v_st[0].nonzero()[:10])
      found_ans = np.logical_or(found_ans, _check_answers(v_st, answers))
      if found_ans.all():
        num_found_ans[i + 1] += 1
        break
  for i in range(num_hops + 1):
    num_found_ans[i] /= len(questions)
  return num_found_ans


def _process_url(url):
  """Get textual identifier from URL and quote it."""
  return urllib.parse.quote(" ".join(url.rsplit("/", 1)[-1].split("_")))


def main(_):
  logging.set_verbosity(logging.INFO)

  logging.info("Reading HotpotQA data...")
  with tf.gfile.Open(FLAGS.hotpotqa_file) as f:
    data = json.load(f)
  logging.info("Done.")

  logging.info("Entity linking %d questions...", len(data))
  num_empty_questions = 0
  recall, recall_at1 = 0., 0.
  all_questions = []
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
  linked_entities = tfidf_linking(data, FLAGS.entity_dir, tokenizer, 20)
  for ii, item in enumerate(data):
    sup_facts = list(set([title for title, _ in item["supporting_facts"]]))
    # pylint: disable=g-complex-comprehension
    all_questions.append({
        "question":
            item["question"],
        "entities":
            linked_entities[ii],
        "answer":
            item["answer"],
        "_id":
            item["_id"],
        "level":
            item["level"],
        "type":
            item["type"],
        "supporting_facts": [{
            "kb_id": urllib.parse.quote(title),
            "name": title
        } for title in sup_facts],
    })

  logging.info("Writing questions to output file...")
  f_out = tf.gfile.Open(FLAGS.output_file, "w")
  f_out.write("\n".join(json.dumps(q) for q in all_questions))
  f_out.close()
  questions_to_eval = random.sample(all_questions, 1000)
  num_found_ans = evaluate_entity_linking(questions_to_eval, FLAGS.entity_dir,
                                          3)
  logging.info("===============================================")
  logging.info("===============================================")
  logging.info("%d questions without entities (out of %d)", num_empty_questions,
               len(data))
  logging.info("recall of at least 1 supporting facts %.3f", recall / len(data))
  logging.info("recall @1 of supporting facts %.3f", recall_at1 / len(data))
  total_ans_reachable = 0.
  for i in num_found_ans:
    logging.info("answers reachable in %d hops %.3f", i, num_found_ans[i])
    total_ans_reachable += num_found_ans[i]
  logging.info("answers reachable %.3f", total_ans_reachable)
  logging.info("===============================================")
  logging.info("===============================================")


if __name__ == "__main__":
  app.run(main)
