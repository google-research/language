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
r"""Exports a minimal TF-Hub module for BERT models.

python export_to_tfhub \
    --bert_directory=$BERT_DIR \
    --checkpoint_path=$MODEL_DIR \
    --export_path=$HUB_DIR \
    --layer_norm \
    --num_vec_query=${NUM_QUERY_VEC} \
    --num_vec_passage=${NUM_DOC_VEC} \
    --projection_size=${PROJECTION_SIZE}

tfhub publish-module $HUB_DIR/$MODEL_NAME $HUB_HANDLE.
"""

import os

from absl import app
from absl import flags
from bert import modeling
from language.multivec.models import ranking_model_experiment_inbatch
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

flags.DEFINE_string(
    "bert_directory", None,
    "Pass in the directory for BERT_LARGE. The config will be read from there, "
    "with overwrites from the checkpoint name.")

flags.DEFINE_string(
    "checkpoint_path", None,
    "e.g. path/to/my_model; the latest checkpoint will be taken from there.")

flags.DEFINE_string(
    "export_path", None,
    "Path to the output TF-Hub module. The model name will be derived from the "
    "checkpoint.")

FLAGS = flags.FLAGS


def module_fn(is_training):
  """Module function."""
  input_ids_1 = tf.placeholder(tf.int32, [None, None], "input_ids_1")
  input_masks_1 = tf.placeholder(tf.int32, [None, None], "input_masks_1")
  segment_ids_1 = tf.placeholder(tf.int32, [None, None], "segment_ids_1")

  input_ids_2 = tf.placeholder(tf.int32, [None, None], "input_ids_2")
  input_masks_2 = tf.placeholder(tf.int32, [None, None], "input_masks_2")
  segment_ids_2 = tf.placeholder(tf.int32, [None, None], "segment_ids_2")

  bert_config = modeling.BertConfig.from_json_file(
      os.path.join(FLAGS.bert_directory, "bert_config.json"))

  output_layer_query, _ = \
      ranking_model_experiment_inbatch.encode_block(
          bert_config,
          input_ids_1,
          input_masks_1,
          segment_ids_1,
          False,
          FLAGS.num_vec_query,
          is_training)
  output_layer_passage, _ = \
      ranking_model_experiment_inbatch.encode_block(
          bert_config,
          input_ids_2,
          input_masks_2,
          segment_ids_2,
          False,
          FLAGS.num_vec_passage,
          is_training)
  vocab_file = os.path.join(FLAGS.bert_directory, "vocab.txt")
  vocab_file = tf.constant(value=vocab_file, dtype=tf.string, name="vocab_file")

  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file)

  hub.add_signature(
      name="query_output",
      inputs=dict(
          input_ids=input_ids_1,
          input_mask=input_masks_1,
          segment_ids=segment_ids_1),
      outputs=dict(output_layer=output_layer_query))

  hub.add_signature(
      name="passage_output",
      inputs=dict(
          input_ids=input_ids_2,
          input_mask=input_masks_2,
          segment_ids=segment_ids_2),
      outputs=dict(output_layer=output_layer_passage))

  hub.add_signature(
      name="tokenization_info",
      inputs={},
      outputs=dict(vocab_file=vocab_file, do_lower_case=tf.constant(True)))


def main(_):
  tf.logging.info("Running export_to_tfhub.py")
  tags_and_args = []
  for is_training in (True, False):
    tags = set()
    if is_training:
      tags.add("train")
    tags_and_args.append((tags, dict(is_training=is_training)))
  spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)

  tf.logging.info("Using checkpoint {}".format(FLAGS.checkpoint_path))
  tf.logging.info("Exporting to {}".format(FLAGS.export_path))
  spec.export(FLAGS.export_path, checkpoint_path=FLAGS.checkpoint_path)


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_directory")
  flags.mark_flag_as_required("checkpoint_path")
  flags.mark_flag_as_required("export_path")
  app.run(main)
