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
"""Script to pre-process single-hop data for multi-hop path following."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import math
import os
import random

from absl import app
from absl import flags

from bert import tokenization
from language.labs.drkit import bert_utils
from language.labs.drkit import search_utils
from language.labs.drkit.preprocessing import preprocess_utils
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("paragraphs_file", None,
                    "File with paragraphs and mentions.")

flags.DEFINE_string("queries_file", None, "File with queries over paragraphs.")

flags.DEFINE_string("multihop_output_dir", None,
                    "Output directory to store generated files to.")

flags.DEFINE_string(
    "dev_entities_file", None,
    "JSON file containing a list of entities to hold out for "
    "the dev set.")

flags.DEFINE_string("pretrain_dir", None,
                    "Directory with pre-trained BERT model.")

flags.DEFINE_integer("max_paragraphs_per_entity", 50,
                     "Maximum number of paragraphs to retrieve per entity.")

flags.DEFINE_integer("max_entity_len", 15,
                     "Maximum number of tokens per entity.")

flags.DEFINE_integer("ngram_size", 2,
                     "Size of ngrams to hash for creating Tf-Idf vectors.")

flags.DEFINE_integer(
    "hash_size", 24, "Number of hash buckets over the vocabulary. "
    "The number of buckets is 2^{this value}.")

flags.DEFINE_float(
    "idf_cutoff_fraction", 0.0,
    "Fraction of lowest IDF words to remove from sparse search.")


def _get_sub_paras(para, tokenizer, max_seq_length, doc_stride, total):
  """Split paragraph object into sub-paragraphs with maximum length."""
  max_tokens_for_doc = max_seq_length - 2  # -2 for [CLS] and [SEP]
  para_tokens, para_char_to_token = bert_utils.preprocess_text(
      para["context"], tokenizer)

  # Get mention token start and ends.
  mentions = []
  for im, ment in enumerate(para["mentions"]):
    st_tok = para_char_to_token[ment["start"]][0]
    en_tok = para_char_to_token[ment["start"] + len(ment["text"]) - 1][1]
    mentions.append({
        "wikidata_id": ment["wikidata_id"],
        "name": ment["name"],
        "text": ment["text"],
        "start_token": st_tok,
        "end_token": en_tok,
        "orig_index": im,
    })

  # Get sub para spans.
  sub_paras = []
  start_offset = 0
  while start_offset < len(para_tokens):
    length = len(para_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    sub_paras.append((start_offset, length))
    if start_offset + length == len(para_tokens):
      break
    start_offset += min(length, doc_stride)

  # Assign each mention to a sub_para.
  sub_para_to_mentions = {i: [] for i in range(len(sub_paras))}
  for ment in mentions:
    best_score, best_index = None, None
    for ii, subp in enumerate(sub_paras):
      subp_end = subp[0] + subp[1] - 1
      if ment["start_token"] < subp[0] or ment["end_token"] > subp_end:
        continue
      score = min(ment["start_token"] - subp[0], subp_end - ment["end_token"])
      if best_score is None or score > best_score:
        best_score = score
        best_index = ii
    ment["start_token"] -= sub_paras[best_index][0]
    ment["end_token"] -= sub_paras[best_index][0]
    sub_para_to_mentions[best_index].append(ment)

  # Create a list of sub_para objects.
  sub_para_objects = []
  for ii, subp in enumerate(sub_paras):
    sub_para_objects.append({
        "id": total[0],
        "wikidata_id": para["wikidata_id"],
        "mentions": sub_para_to_mentions[ii],
        "tokens": para_tokens[subp[0]:subp[0] + subp[1]],
    })
    total[0] += 1

  return sub_para_objects


def main(_):
  if not tf.gfile.Exists(FLAGS.multihop_output_dir):
    tf.gfile.MakeDirs(FLAGS.multihop_output_dir)

  # Initialize tokenizer.
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # Number of buckets.
  hash_size = int(math.pow(2, FLAGS.hash_size))

  # Read paragraphs, mentions and entities.
  mentions, entity2id, entity2name = [], {}, {}
  para_rows, para_cols, para_vals = [], [], []
  ment_rows, ment_cols, ment_vals = [], [], []
  para_to_global_mention = {}
  mention2text = {}
  total_sub_paras = [0]
  all_sub_paras = []
  entity2paracount = collections.defaultdict(int)
  tf.logging.info("Reading paragraphs from %s", FLAGS.paragraphs_file)
  with tf.gfile.Open(FLAGS.paragraphs_file) as f:
    for line in tqdm(f):
      orig_para = json.loads(line.strip())
      para_id = orig_para["id"]
      sub_para_objs = _get_sub_paras(orig_para, tokenizer, FLAGS.max_seq_length,
                                     FLAGS.doc_stride, total_sub_paras)
      para_to_global_mention[para_id] = {}
      for para_obj in sub_para_objs:
        # Add mentions from this paragraph.
        for m in para_obj["mentions"]:
          if m["wikidata_id"] is None:
            para_to_global_mention[para_id][m["orig_index"]] = -1
            continue
          para_to_global_mention[para_id][m["orig_index"]] = len(mentions)
          # Create paragraph to mention sparse connections.
          ment_rows.append(para_obj["id"])
          ment_cols.append(len(mentions))
          ment_vals.append(1)
          if m["wikidata_id"] not in entity2id:
            entity2id[m["wikidata_id"]] = len(entity2id)
            if m["name"] is not None:
              entity2name[m["wikidata_id"]] = m["name"]
            else:
              entity2name[m["wikidata_id"]] = m["text"]
          mention2text[len(mentions)] = m["text"]
          mentions.append((entity2id[m["wikidata_id"]], para_obj["id"],
                           m["start_token"], m["end_token"]))
        # Create token counts for this paragraph.
        para_counts = search_utils.get_hashed_counts(para_obj["tokens"],
                                                     FLAGS.ngram_size,
                                                     hash_size)
        para_rows.extend(para_counts.keys())
        para_cols.extend([para_obj["id"]] * len(para_counts))
        para_vals.extend(para_counts.values())
        all_sub_paras.append(para_obj["tokens"])
        entity2paracount[para_obj["wikidata_id"]] += 1
      assert len(all_sub_paras) == total_sub_paras[0], (len(all_sub_paras),
                                                        total_sub_paras)
  tf.logging.info("Creating paragraph and mention sparse matrices")
  tf.logging.info("Num paragraphs = %d, Num mentions = %d", total_sub_paras[0],
                  len(mentions))
  sp_vocab2para = sp.csr_matrix((para_vals, (para_rows, para_cols)),
                                shape=(hash_size, total_sub_paras[0]),
                                dtype=np.float32)
  sp_para2ment = sp.csr_matrix((ment_vals, (ment_rows, ment_cols)),
                               shape=(total_sub_paras[0], len(mentions)),
                               dtype=np.float32)
  tf.logging.info("Saving coreference map.")
  search_utils.write_to_checkpoint(
      "coref", np.array([m[0] for m in mentions], dtype=np.int32), tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "coref.npz"))
  tf.logging.info("Saving mentions metadata.")
  np.save(
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "mentions.npy"), "w"),
      np.array(mentions, dtype=np.int64))
  json.dump(
      mention2text,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "mention2text.json"), "w"))
  tf.logging.info("Saving entities metadata.")
  json.dump([entity2id, entity2name],
            tf.gfile.Open(
                os.path.join(FLAGS.multihop_output_dir, "entities.json"), "w"))
  tf.logging.info("Saving split paragraphs.")
  json.dump(
      all_sub_paras,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "subparas.json"), "w"))

  # Compute IDF and TF-IDF
  idfs = search_utils.counts_to_idfs(
      sp_vocab2para, cutoff=FLAGS.idf_cutoff_fraction)
  tf.logging.info("Number of non-zero IDFs = %d (total = %d)", idfs.getnnz(),
                  idfs.shape[0])
  sp_paratfidf = search_utils.counts_to_tfidf(sp_vocab2para, idfs)

  # Create entity sparse vectors.
  tf.logging.info("Processing entities.")
  entity_rows, entity_cols, entity_vals = [], [], []
  entity_ids = np.zeros((len(entity2id), FLAGS.max_entity_len), dtype=np.int32)
  entity_mask = np.zeros((len(entity2id), FLAGS.max_entity_len),
                         dtype=np.float32)
  num_exceed_len = 0.
  for entity in tqdm(entity2id):
    ei = entity2id[entity]
    entity_tokens = tokenizer.tokenize(entity2name[entity])
    entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
    if len(entity_token_ids) > FLAGS.max_entity_len:
      num_exceed_len += 1
      entity_token_ids = entity_token_ids[:FLAGS.max_entity_len]
    entity_ids[ei, :len(entity_token_ids)] = entity_token_ids
    entity_mask[ei, :len(entity_token_ids)] = 1.
    entity_counts = search_utils.get_hashed_counts(entity_tokens,
                                                   FLAGS.ngram_size, hash_size)
    entity_rows.extend(entity_counts.keys())
    entity_cols.extend([entity2id[entity]] * len(entity_counts))
    entity_vals.extend(entity_counts.values())
  tf.logging.info("Saving %d entity ids and mask. %d exceed max-length of %d.",
                  len(entity2id), num_exceed_len, FLAGS.max_entity_len)
  search_utils.write_to_checkpoint(
      "entity_ids", entity_ids, tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "entity_ids"))
  search_utils.write_to_checkpoint(
      "entity_mask", entity_mask, tf.float32,
      os.path.join(FLAGS.multihop_output_dir, "entity_mask"))
  tf.logging.info("Creating entity sparse matrix (total = %d)", len(entity2id))
  sp_vocab2entity = sp.csr_matrix((entity_vals, (entity_rows, entity_cols)),
                                  shape=(hash_size, len(entity2id)),
                                  dtype=np.float32)
  sp_entitytfidf = search_utils.counts_to_tfidf(sp_vocab2entity, idfs)

  # Multiply to get entity to mention.
  tf.logging.info("Converting to entity -> mention search matrix.")
  tf.logging.info("Entities to paragraphs.")
  sp_entity2para = sp_entitytfidf.transpose().dot(sp_paratfidf).tocsr()
  tf.logging.info("Non zero entries before filtering = %d",
                  sp_entity2para.getnnz())
  tf.logging.info("Filtering top-%d paragraphs per entity.",
                  FLAGS.max_paragraphs_per_entity)
  sp_entity2para_filt = preprocess_utils.filter_sparse_rows(
      sp_entity2para, FLAGS.max_paragraphs_per_entity)
  tf.logging.info("Non zero entries after filtering = %d",
                  sp_entity2para_filt.getnnz())
  tf.logging.info("Entities to mentions.")
  sp_entity2mention = sp_entity2para_filt.dot(sp_para2ment)
  tf.logging.info("Non zero entries final = %d", sp_entity2mention.getnnz())

  tf.logging.info("Deleting arrays not needed anymore.")
  del sp_entity2para
  del sp_entity2para_filt
  del sp_entitytfidf
  del sp_paratfidf
  del sp_para2ment

  tf.logging.info("Saving sparse matrix with shape %s",
                  str(sp_entity2mention.shape))
  search_utils.write_ragged_to_checkpoint(
      "ent2ment", sp_entity2mention,
      os.path.join(FLAGS.multihop_output_dir, "ent2ment.npz"))

  # Read queries and find global index of answers.
  tf.logging.info("Reading queries from %s.", FLAGS.queries_file)
  all_queries = []
  with tf.gfile.Open(FLAGS.queries_file) as f:
    for line in tqdm(f):
      qry = json.loads(line.strip())
      if qry["object"] is None:
        continue
      if "bridge" in qry:
        subj_para_id = qry["para_id"][0]
        obj_para_id = qry["para_id"][1]
      elif "bridge_0" in qry:
        subj_para_id = qry["para_id"][0]
        bridge_para_id = qry["para_id"][1]
        obj_para_id = qry["para_id"][2]
      else:
        subj_para_id = qry["para_id"]
        obj_para_id = qry["para_id"]
      qry["subject"]["ent_id"] = entity2id.get(qry["subject"]["wikidata_id"], 0)
      qry["object"]["ent_id"] = entity2id.get(qry["object"]["wikidata_id"], 0)
      if subj_para_id in para_to_global_mention:
        qry["subject"]["global_mentions"] = []
        for m in qry["subject"]["mentions"]:
          qry["subject"]["global_mentions"].append(
              para_to_global_mention[subj_para_id][m])
      else:
        qry["subject"]["global_mentions"] = [0]
      if obj_para_id in para_to_global_mention:
        qry["object"]["global_mention"] = (
            para_to_global_mention[obj_para_id][qry["object"]["mention"]])
      else:
        qry["object"]["global_mention"] = 0
      my_mention_text = mention2text[qry["object"]["global_mention"]]
      if ("aliases" in qry["object"] and
          my_mention_text not in qry["object"]["aliases"]):
        qry["object"]["aliases"][my_mention_text] = 1
      if "bridge" in qry:
        qry["bridge"]["ent_id"] = entity2id[qry["bridge"]["wikidata_id"]]
        qry["bridge"]["global_mention_1"] = (
            para_to_global_mention[subj_para_id][qry["bridge"]["mention_1"]])
        qry["bridge"]["global_mention_2"] = []
        for m in qry["bridge"]["mention_2"]:
          qry["bridge"]["global_mention_2"].append(
              para_to_global_mention[obj_para_id][m])
      if "bridge_0" in qry:
        qry["bridge_0"]["ent_id"] = entity2id[qry["bridge_0"]["wikidata_id"]]
        qry["bridge_0"]["global_mention_1"] = (
            para_to_global_mention[subj_para_id][qry["bridge_0"]["mention_1"]])
        qry["bridge_0"]["global_mention_2"] = []
        for m in qry["bridge_0"]["mention_2"]:
          qry["bridge_0"]["global_mention_2"].append(
              para_to_global_mention[bridge_para_id][m])
      if "bridge_1" in qry:
        qry["bridge_1"]["ent_id"] = entity2id[qry["bridge_1"]["wikidata_id"]]
        qry["bridge_1"]["global_mention_1"] = (
            para_to_global_mention[bridge_para_id][qry["bridge_1"]["mention_1"]]
        )
        qry["bridge_1"]["global_mention_2"] = []
        for m in qry["bridge_1"]["mention_2"]:
          qry["bridge_1"]["global_mention_2"].append(
              para_to_global_mention[obj_para_id][m])
      all_queries.append(qry)

  # Split into train and dev sets.
  if FLAGS.dev_entities_file is not None:
    dev_entities = set(json.load(tf.gfile.Open(FLAGS.dev_entities_file)))
    tf.logging.info("Loaded %d dev set entities", len(dev_entities))
  else:
    all_entities = set([qry["subject"]["wikidata_id"] for qry in all_queries])
    dev_entities = set(
        random.sample(list(all_entities), int(1.0 * len(all_entities))))
    tf.logging.info("Sampled %d dev set entities", len(dev_entities))

  # Store.
  train_f = os.path.join(FLAGS.multihop_output_dir, "train_qrys.json")
  dev_f = os.path.join(FLAGS.multihop_output_dir, "dev_qrys.json")
  dev_queries = []
  with tf.gfile.Open(train_f, "w") as f_tr, tf.gfile.Open(dev_f, "w") as f_de:
    for qry in all_queries:
      if qry["subject"]["wikidata_id"] in dev_entities:
        f_de.write(json.dumps(qry) + "\n")
        if "bridge" in qry:
          dev_queries.append({
              "id": qry["id"],
              "query": (qry["subject"]["name"] + " . " +
                        qry["relation"][1]["text"][0]),
              "subject": qry["subject"]["ent_id"],
              "subject_id": qry["subject"]["wikidata_id"],
              "entity": qry["object"]["ent_id"],
              "mention": qry["object"]["global_mention"],
              "paracount": entity2paracount[qry["subject"]["wikidata_id"]],
          })
        elif "bridge_0" in qry:
          dev_queries.append({
              "id": qry["id"],
              "query": (qry["subject"]["name"] + " . " +
                        qry["relation"][1]["text"][0] + " . " +
                        qry["relation"][2]["text"][0]),
              "subject": qry["subject"]["ent_id"],
              "subject_id": qry["subject"]["wikidata_id"],
              "entity": qry["object"]["ent_id"],
              "mention": qry["object"]["global_mention"],
              "paracount": entity2paracount[qry["subject"]["wikidata_id"]],
          })
        else:
          dev_queries.append({
              "id": qry["id"],
              "query":
                  (qry["subject"]["name"] + " . " + qry["relation"]["text"][0]),
              "subject": qry["subject"]["ent_id"],
              "subject_id": qry["subject"]["wikidata_id"],
              "entity": qry["object"]["ent_id"],
              "mention": qry["object"]["global_mention"],
              "paracount": entity2paracount[qry["subject"]["wikidata_id"]],
          })
      else:
        f_tr.write(json.dumps(qry) + "\n")

  # Copy BERT checkpoint for future use.
  tf.logging.info("Copying BERT checkpoint.")
  bert_ckpt = tf.train.latest_checkpoint(FLAGS.pretrain_dir)
  tf.logging.info("%s.data-00000-of-00001", bert_ckpt)
  tf.gfile.Copy(
      bert_ckpt + ".data-00000-of-00001",
      os.path.join(FLAGS.multihop_output_dir, "bert_init.data-00000-of-00001"),
      overwrite=True)
  tf.logging.info("%s.index", bert_ckpt)
  tf.gfile.Copy(
      bert_ckpt + ".index",
      os.path.join(FLAGS.multihop_output_dir, "bert_init.index"),
      overwrite=True)
  tf.logging.info("%s.meta", bert_ckpt)
  tf.gfile.Copy(
      bert_ckpt + ".meta",
      os.path.join(FLAGS.multihop_output_dir, "bert_init.meta"),
      overwrite=True)

  # Get mention embeddings from BERT.
  tf.logging.info("Computing mention embeddings for %d paras.",
                  len(all_sub_paras))
  bert_predictor = bert_utils.BERTPredictor(tokenizer, bert_ckpt)
  para_emb = bert_predictor.get_doc_embeddings(all_sub_paras)
  mention_emb = np.empty((len(mentions), 2 * bert_predictor.emb_dim),
                         dtype=np.float32)
  for im, mention in enumerate(mentions):
    mention_emb[im, :] = np.concatenate([
        para_emb[mention[1], mention[2], :], para_emb[mention[1], mention[3], :]
    ])
  del para_emb
  tf.logging.info("Saving %d mention features to tensorflow checkpoint.",
                  mention_emb.shape[0])
  with tf.device("/cpu:0"):
    search_utils.write_to_checkpoint(
        "db_emb", mention_emb, tf.float32,
        os.path.join(FLAGS.multihop_output_dir, "mention_feats"))

  # Check accuracy of queries against index.
  if len(dev_queries) > 500:
    eval_queries = random.sample(dev_queries, 500)
  else:
    eval_queries = dev_queries
  tf.logging.info("Evaluating %d queries.", len(eval_queries))
  tf.logging.info("Computing dense retrieval scores.")
  qry_toks = [tokenizer.tokenize(qry["query"]) for qry in eval_queries]
  ent_toks = [
      tokenizer.tokenize(entity2name[qry["subject_id"]]) for qry in eval_queries
  ]
  qry_st, qry_en, qry_bow = bert_predictor.get_qry_embeddings(
      qry_toks, ent_toks)
  qry_vecs = np.concatenate([qry_st, qry_en], axis=1)
  rel_dots = qry_vecs.dot(mention_emb.transpose())
  bow_vecs = np.concatenate([qry_bow, qry_bow], axis=1)
  bow_dots = bow_vecs.dot(mention_emb.transpose())
  tf.logging.info("Done.")
  tf.logging.info("Computing sparse retrieval scores.")
  qry_ents = np.array([qry["subject"] for qry in eval_queries])
  sp_qry_to_entity = sp.csr_matrix(
      (np.ones(qry_ents.shape[0]), (np.arange(qry_ents.shape[0]), qry_ents)),
      shape=[qry_ents.shape[0], len(entity2id)],
      dtype=np.float32)
  sparse_dots = sp_qry_to_entity.dot(sp_entity2mention)
  tf.logging.info("Done.")
  tf.logging.info("Multiplying and obtaining answers.")
  top_ans = np.argmax(sparse_dots.multiply(bow_dots + rel_dots), axis=1)
  top_ans = np.squeeze(np.asarray(top_ans))
  mention_acc, entity_acc = 0., 0.
  all_paracounts = [qry["paracount"] for qry in eval_queries]
  sort_i = sorted(range(len(all_paracounts)), key=lambda k: all_paracounts[k])
  fourth = all_paracounts[sort_i[int(0.25 * len(sort_i))]]
  threefourth = all_paracounts[sort_i[int(0.75 * len(sort_i))]]
  bottom_correct, bottom_total = 0., 0
  middle_correct, middle_total = 0., 0
  top_correct, top_total = 0., 0
  sparse_recall = 0.
  for ii, qry in enumerate(eval_queries):
    if qry["mention"] == top_ans[ii]:
      mention_acc += 1
    if qry["entity"] == mentions[top_ans[ii]][0]:
      entity_acc += 1
      if qry["paracount"] < fourth:
        bottom_correct += 1
      elif qry["paracount"] < threefourth:
        middle_correct += 1
      else:
        top_correct += 1
    if qry["paracount"] < fourth:
      bottom_total += 1
    elif qry["paracount"] < threefourth:
      middle_total += 1
    else:
      top_total += 1
    if sparse_dots[ii, qry["mention"]] > 0.:
      sparse_recall += 1
  tf.logging.info("Mention Accuracy = %.3f Entity Accuracy = %.3f",
                  mention_acc / len(eval_queries),
                  entity_acc / len(eval_queries))
  tf.logging.info("Sparse recall = %.3f", sparse_recall / len(eval_queries))
  tf.logging.info("Fourth cutoff = %d Three fourth cutoff = %d", fourth,
                  threefourth)
  tf.logging.info("Bottom Acc = %.3f Middle Acc = %.3f Top Acc = %.3f",
                  bottom_correct / bottom_total, middle_correct / middle_total,
                  top_correct / top_total)


if __name__ == "__main__":
  app.run(main)
