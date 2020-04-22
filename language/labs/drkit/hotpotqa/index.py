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
"""Script to pre-process HotpotQA data."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
# import sys

from absl import app
from absl import flags

from bert import tokenization
from language.labs.drkit import bert_utils_v2
from language.labs.drkit import search_utils
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("wiki_file", None, "Path to corpus.")

flags.DEFINE_string("multihop_output_dir", None, "Path to output files.")

flags.DEFINE_string("pretrain_dir", None,
                    "Directory with pre-trained BERT model.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_boolean("do_preprocess", None,
                     "Whether to run paragraph preprocessing.")

flags.DEFINE_boolean("do_copy", None,
                     "Whether to copy bert checkpoint.")

flags.DEFINE_boolean("do_embed", None, "Whether to run mention embedding.")

flags.DEFINE_boolean("do_combine", None,
                     "Whether to combine all shards into one.")

flags.DEFINE_integer("max_total_paragraphs", None,
                     "Maximum number of paragraphs to process.")

flags.DEFINE_integer("predict_batch_size", 32, "Batch size for embedding.")

flags.DEFINE_integer("max_mentions_per_entity", 20,
                     "Maximum number of mentions to retrieve per entity.")

flags.DEFINE_integer("max_entity_length", 15,
                     "Maximum number of tokens per entity.")

flags.DEFINE_integer("num_shards", 20,
                     "Number of shards to store mention embeddings in.")

flags.DEFINE_integer("my_shard", None,
                     "Shard number for this process to run over.")

flags.DEFINE_integer("shards_to_combine", None,
                     "Max number of shards to combine.")


def _get_sub_paras(para, tokenizer, max_seq_length, doc_stride, total):
  """Split paragraph object into sub-paragraphs with maximum length."""
  if not para["context"]:
    return []
  max_tokens_for_doc = max_seq_length - 2  # -2 for [CLS] and [SEP]
  para_tokens, para_char_to_token = bert_utils_v2.preprocess_text(
      para["context"], tokenizer)

  # Get mention token start and ends.
  mentions = []
  for im, ment in enumerate(para["mentions"]):
    if ment["start"] + len(ment["text"]) - 1 >= len(para_char_to_token):
      tf.logging.warn("Mention not within paragraph: (%s, %s)",
                      json.dumps(ment), para["context"])
      continue
    st_tok = para_char_to_token[ment["start"]][0]
    en_tok = para_char_to_token[ment["start"] + len(ment["text"]) - 1][1]
    mentions.append({
        "kb_id": ment["kb_id"],
        "text": ment["text"],
        "start_token": st_tok,
        "end_token": en_tok,
        "orig_index": im,
    })
  if not mentions:
    return []

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
  if not sub_paras:
    return []

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
    if best_index is None:
      best_index = 0
    ment["start_token"] -= sub_paras[best_index][0]
    ment["end_token"] -= sub_paras[best_index][0]
    if (ment["start_token"] < sub_paras[best_index][1] and
        ment["end_token"] < sub_paras[best_index][1]):
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

  # Initialize tokenizer.
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # Read paragraphs, mentions and entities.
  if FLAGS.do_preprocess:
    tf.logging.info("Reading all entities")
    entity2id, entity2name = {}, {}
    with tf.gfile.Open(FLAGS.wiki_file) as f:
      for ii, line in tqdm(enumerate(f)):
        orig_para = json.loads(line.strip())
        ee = orig_para["kb_id"].lower()
        if ee not in entity2id:
          entity2id[ee] = len(entity2id)
          entity2name[ee] = orig_para["title"]
    tf.logging.info("Found %d entities", len(entity2id))

    tf.logging.info("Reading paragraphs from %s", FLAGS.wiki_file)
    mentions = []
    ent_rows, ent_cols, ent_vals = [], [], []
    mention2text = {}
    total_sub_paras = [0]
    all_sub_paras = []
    num_skipped_mentions = 0.
    with tf.gfile.Open(FLAGS.wiki_file) as f:
      for ii, line in tqdm(enumerate(f)):
        if ii == FLAGS.max_total_paragraphs:
          tf.logging.info("Processed maximum number of paragraphs, breaking.")
          break
        if ii > 0 and ii % 100000 == 0:
          tf.logging.info("Skipped / Kept mentions = %.3f",
                          num_skipped_mentions / len(mentions))
        orig_para = json.loads(line.strip())
        if orig_para["kb_id"].lower() not in entity2id:
          tf.logging.warn("%s not in entities. Skipping %s para",
                          orig_para["kb_id"], orig_para["title"])
          continue
        sub_para_objs = _get_sub_paras(orig_para, tokenizer,
                                       FLAGS.max_seq_length, FLAGS.doc_stride,
                                       total_sub_paras)
        for para_obj in sub_para_objs:
          # Add mentions from this paragraph.
          local2global = {}
          title_entity_mention = None
          for im, mention in enumerate(
              para_obj["mentions"][:FLAGS.max_mentions_per_entity]):
            if mention["kb_id"].lower() not in entity2id:
              # tf.logging.warn("%s not in entities. Skipping mention %s",
              #                 mention["kb_id"], mention["text"])
              num_skipped_mentions += 1
              continue
            mention2text[len(mentions)] = mention["text"]
            local2global[im] = len(mentions)
            if mention["kb_id"] == orig_para["kb_id"]:
              title_entity_mention = len(mentions)
            mentions.append(
                (entity2id[mention["kb_id"].lower()], para_obj["id"],
                 mention["start_token"], mention["end_token"]))
          for im, gm in local2global.items():
            # entity to mention matrix.
            ent_rows.append(entity2id[orig_para["kb_id"].lower()])
            ent_cols.append(gm)
            ent_vals.append(1.)
            if title_entity_mention is not None:
              ent_rows.append(mentions[gm][0])
              ent_cols.append(title_entity_mention)
              ent_vals.append(1.)
          all_sub_paras.append(para_obj["tokens"])
        assert len(all_sub_paras) == total_sub_paras[0], (len(all_sub_paras),
                                                          total_sub_paras)
    tf.logging.info("Num paragraphs = %d, Num mentions = %d",
                    total_sub_paras[0], len(mentions))
    tf.logging.info("Saving coreference map.")
    search_utils.write_to_checkpoint(
        "coref", np.array([m[0] for m in mentions], dtype=np.int32), tf.int32,
        os.path.join(FLAGS.multihop_output_dir, "coref.npz"))
    tf.logging.info("Creating entity to mentions matrix.")
    sp_entity2mention = sp.csr_matrix((ent_vals, (ent_rows, ent_cols)),
                                      shape=[len(entity2id),
                                             len(mentions)])
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
                  os.path.join(FLAGS.multihop_output_dir, "entities.json"),
                  "w"))
    tf.logging.info("Saving split paragraphs.")
    json.dump(
        all_sub_paras,
        tf.gfile.Open(
            os.path.join(FLAGS.multihop_output_dir, "subparas.json"), "w"))

  # Store entity tokens.
  if FLAGS.do_preprocess:
    tf.logging.info("Processing entities.")
    entity_ids = np.zeros((len(entity2id), FLAGS.max_entity_length),
                          dtype=np.int32)
    entity_mask = np.zeros((len(entity2id), FLAGS.max_entity_length),
                           dtype=np.float32)
    num_exceed_len = 0.
    for entity in tqdm(entity2id):
      ei = entity2id[entity]
      entity_tokens = tokenizer.tokenize(entity2name[entity])
      entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
      if len(entity_token_ids) > FLAGS.max_entity_length:
        num_exceed_len += 1
        entity_token_ids = entity_token_ids[:FLAGS.max_entity_length]
      entity_ids[ei, :len(entity_token_ids)] = entity_token_ids
      entity_mask[ei, :len(entity_token_ids)] = 1.
    tf.logging.info("Saving %d entity ids. %d exceed max-length of %d.",
                    len(entity2id), num_exceed_len, FLAGS.max_entity_length)
    search_utils.write_to_checkpoint(
        "entity_ids", entity_ids, tf.int32,
        os.path.join(FLAGS.multihop_output_dir, "entity_ids"))
    search_utils.write_to_checkpoint(
        "entity_mask", entity_mask, tf.float32,
        os.path.join(FLAGS.multihop_output_dir, "entity_mask"))

  # Copy BERT checkpoint for future use.
  if FLAGS.do_copy:
    tf.logging.info("Copying BERT checkpoint.")
    if tf.gfile.Exists(os.path.join(FLAGS.pretrain_dir, "best_model.index")):
      bert_ckpt = os.path.join(FLAGS.pretrain_dir, "best_model")
    else:
      bert_ckpt = tf.train.latest_checkpoint(FLAGS.pretrain_dir)
    tf.logging.info("%s.data-00000-of-00001", bert_ckpt)
    tf.gfile.Copy(
        bert_ckpt + ".data-00000-of-00001",
        os.path.join(FLAGS.multihop_output_dir,
                     "bert_init.data-00000-of-00001"),
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

  if FLAGS.do_embed:
    # Get mention embeddings from BERT.
    bert_ckpt = os.path.join(FLAGS.multihop_output_dir, "bert_init")
    if not FLAGS.do_preprocess:
      with tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "mentions.npy"), "rb") as f:
        mentions = np.load(f)
      with tf.gfile.Open(
          os.path.join(FLAGS.multihop_output_dir, "subparas.json")) as f:
        all_sub_paras = json.load(f)
    tf.logging.info("Computing embeddings for %d mentions over %d paras.",
                    len(mentions), len(all_sub_paras))
    shard_size = len(mentions) // FLAGS.num_shards
    bert_predictor = bert_utils_v2.BERTPredictor(tokenizer, bert_ckpt)
    if FLAGS.my_shard is None:
      shard_range = range(FLAGS.num_shards + 1)
    else:
      shard_range = [FLAGS.my_shard]
    for ns in shard_range:
      min_ = ns * shard_size
      max_ = (ns + 1) * shard_size
      if min_ >= len(mentions):
        break
      if max_ > len(mentions):
        max_ = len(mentions)
      min_subp = mentions[min_][1]
      max_subp = mentions[max_ - 1][1]
      tf.logging.info("Processing shard %d of %d mentions and %d paras.", ns,
                      max_ - min_, max_subp - min_subp + 1)
      para_emb = bert_predictor.get_doc_embeddings(
          all_sub_paras[min_subp:max_subp + 1])
      assert para_emb.shape[2] == 2 * FLAGS.projection_dim
      mention_emb = np.empty((max_ - min_, 2 * bert_predictor.emb_dim),
                             dtype=np.float32)
      for im, mention in enumerate(mentions[min_:max_]):
        mention_emb[im, :] = np.concatenate([
            para_emb[mention[1] - min_subp, mention[2], :FLAGS.projection_dim],
            para_emb[mention[1] - min_subp, mention[3],
                     FLAGS.projection_dim:2 * FLAGS.projection_dim]
        ])
      del para_emb
      tf.logging.info("Saving %d mention features to tensorflow checkpoint.",
                      mention_emb.shape[0])
      with tf.device("/cpu:0"):
        search_utils.write_to_checkpoint(
            "db_emb_%d" % ns, mention_emb, tf.float32,
            os.path.join(FLAGS.multihop_output_dir, "mention_feats_%d" % ns))

  if FLAGS.do_combine:
    # Combine sharded DB into one.
    if FLAGS.shards_to_combine is None:
      shard_range = range(FLAGS.num_shards + 1)
    else:
      shard_range = range(FLAGS.shards_to_combine)
    with tf.device("/cpu:0"):
      all_db = []
      for i in shard_range:
        ckpt_path = os.path.join(FLAGS.multihop_output_dir,
                                 "mention_feats_%d" % i)
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        tf.logging.info("Reading %s from %s with shape %s", "db_emb_%d" % i,
                        ckpt_path, str(var_to_shape_map["db_emb_%d" % i]))
        tf_db = search_utils.load_database("db_emb_%d" % i,
                                           var_to_shape_map["db_emb_%d" % i],
                                           ckpt_path)
        all_db.append(tf_db)
      tf.logging.info("Reading all variables.")
      session = tf.Session()
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      np_db = session.run(all_db)
      tf.logging.info("Concatenating and storing.")
      np_db = np.concatenate(np_db, axis=0)
      search_utils.write_to_checkpoint(
          "db_emb", np_db, tf.float32,
          os.path.join(FLAGS.multihop_output_dir, "mention_feats"))


if __name__ == "__main__":
  app.run(main)
