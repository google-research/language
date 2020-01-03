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
"""Script to pre-process wikimovie data."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os

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

flags.DEFINE_string("data_dir", None, "Path to corpus files.")

flags.DEFINE_string("qry_dir", None, "Path to question files.")

flags.DEFINE_string("multihop_output_dir", None, "Path to output files.")

flags.DEFINE_string("pretrain_dir", None,
                    "Directory with pre-trained BERT model.")

flags.DEFINE_integer("max_paragraphs_per_entity", 50,
                     "Maximum number of paragraphs to retrieve per entity.")

flags.DEFINE_integer("max_entity_len", 15,
                     "Maximum number of tokens per entity.")


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
        "kb_id": ment["kb_id"],
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
        "mentions": sub_para_to_mentions[ii],
        "tokens": para_tokens[subp[0]:subp[0] + subp[1]],
    })
    total[0] += 1

  return sub_para_objects


def main(_):
  if not tf.gfile.Exists(FLAGS.multihop_output_dir):
    tf.gfile.MakeDirs(FLAGS.multihop_output_dir)

  # Filenames.
  paragraphs_file = os.path.join(FLAGS.data_dir, "processed_wiki.json")
  train_file = os.path.join(FLAGS.qry_dir, "train.json")
  dev_file = os.path.join(FLAGS.qry_dir, "dev.json")
  test_file = os.path.join(FLAGS.qry_dir, "test.json")
  entities_file = os.path.join(FLAGS.data_dir, "entities.txt")

  # Initialize tokenizer.
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # Read entities.
  tf.logging.info("Reading entities.")
  entity2id, entity2name = {}, {}
  with tf.gfile.Open(entities_file) as f:
    for ii, line in tqdm(enumerate(f)):
      entity = line.strip()
      entity2id[entity] = ii
      entity2name[entity] = entity

  # Read paragraphs, mentions and entities.
  mentions = []
  ent_rows, ent_cols, ent_vals = [], [], []
  para_rows, para_cols, para_vals = [], [], []
  mention2text = {}
  total_sub_paras = [0]
  all_sub_paras = []
  entity_counts = collections.defaultdict(int)
  tf.logging.info("Reading paragraphs from %s", paragraphs_file)
  with tf.gfile.Open(paragraphs_file) as f:
    for line in tqdm(f):
      orig_para = json.loads(line.strip())
      sub_para_objs = _get_sub_paras(orig_para, tokenizer, FLAGS.max_seq_length,
                                     FLAGS.doc_stride, total_sub_paras)
      for para_obj in sub_para_objs:
        # Add mentions from this paragraph.
        my_entities = []
        my_mentions = []
        for m in para_obj["mentions"]:
          # Para to mention matrix.
          para_rows.append(para_obj["id"])
          para_cols.append(len(mentions))
          para_vals.append(1.)
          # Create entity to mention sparse connections.
          my_entities.append(m["kb_id"])
          my_mentions.append(len(mentions))
          entity_counts[m["kb_id"]] += 1
          mention2text[len(mentions)] = m["text"]
          mentions.append(
              (m["kb_id"], para_obj["id"], m["start_token"], m["end_token"]))
        for entity in my_entities:
          ent_rows.append(entity)
          ent_cols.append(para_obj["id"])
          ent_vals.append(1. / len(my_mentions))
        all_sub_paras.append(para_obj["tokens"])
      assert len(all_sub_paras) == total_sub_paras[0], (len(all_sub_paras),
                                                        total_sub_paras)
  tf.logging.info("Num paragraphs = %d, Num mentions = %d", total_sub_paras[0],
                  len(mentions))
  tf.logging.info("Saving coreference map.")
  search_utils.write_to_checkpoint(
      "coref", np.array([m[0] for m in mentions], dtype=np.int32), tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "coref.npz"))
  tf.logging.info("Creating entity to mentions matrix.")
  sp_entity2para = sp.csr_matrix((ent_vals, (ent_rows, ent_cols)),
                                 shape=[len(entity2id),
                                        len(all_sub_paras)])
  sp_entity2para_filt = preprocess_utils.filter_sparse_rows(
      sp_entity2para, FLAGS.max_paragraphs_per_entity)
  sp_para2ment = sp.csr_matrix((para_vals, (para_rows, para_cols)),
                               shape=[len(all_sub_paras),
                                      len(mentions)])
  sp_entity2mention = sp_entity2para_filt.dot(sp_para2ment)
  tf.logging.info("Num nonzero = %d", sp_entity2mention.getnnz())
  tf.logging.info("Saving as ragged tensor %s.", str(sp_entity2mention.shape))
  search_utils.write_ragged_to_checkpoint(
      "ent2ment", sp_entity2mention,
      os.path.join(FLAGS.multihop_output_dir, "ent2ment.npz"))
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
  json.dump(
      entity_counts,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "entity_counts.json"), "w"))
  tf.logging.info("Saving split paragraphs.")
  json.dump(
      all_sub_paras,
      tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "subparas.json"), "w"))

  # Store entity tokens.
  tf.logging.info("Processing entities.")
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
  tf.logging.info("Saving %d entity ids and mask. %d exceed max-length of %d.",
                  len(entity2id), num_exceed_len, FLAGS.max_entity_len)
  search_utils.write_to_checkpoint(
      "entity_ids", entity_ids, tf.int32,
      os.path.join(FLAGS.multihop_output_dir, "entity_ids"))
  search_utils.write_to_checkpoint(
      "entity_mask", entity_mask, tf.float32,
      os.path.join(FLAGS.multihop_output_dir, "entity_mask"))

  # Pre-process question files.
  def _preprocess_qrys(in_file, out_file):
    tf.logging.info("Working on %s", in_file)
    with tf.gfile.Open(in_file) as f_in, tf.gfile.Open(out_file, "w") as f_out:
      for line in f_in:
        item = json.loads(line.strip())
        # Sort entities in ascending order of their frequencies.
        e_counts = [entity_counts[e["kb_id"]] for e in item["entities"]]
        sorted_i = sorted(enumerate(e_counts), key=lambda x: x[1])
        item["entities"] = [item["entities"][ii] for ii, _ in sorted_i]
        f_out.write(json.dumps(item) + "\n")

  _preprocess_qrys(train_file,
                   os.path.join(FLAGS.multihop_output_dir, "train.json"))
  _preprocess_qrys(dev_file, os.path.join(FLAGS.multihop_output_dir,
                                          "dev.json"))
  _preprocess_qrys(test_file,
                   os.path.join(FLAGS.multihop_output_dir, "test.json"))

  # Copy BERT checkpoint for future use.
  tf.logging.info("Copying BERT checkpoint.")
  if tf.gfile.Exists(os.path.join(FLAGS.pretrain_dir, "best_model.index")):
    bert_ckpt = os.path.join(FLAGS.pretrain_dir, "best_model")
  else:
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


if __name__ == "__main__":
  app.run(main)
