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
"""Encode wikipedia."""
import os
import time

from absl import app
from absl import flags

from language.orqa.utils import bert_utils
from language.orqa.utils import scann_utils

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub

flags.DEFINE_string("retriever_module_path", None,
                    "Path to the retriever TF-Hub module.")
flags.DEFINE_string("examples_path", None, "Path to tf.train.Examples")
flags.DEFINE_integer("num_blocks", 13353718, "Expected number of blocks.")
flags.DEFINE_integer("num_threads", 48, "Num threads for input reading.")
flags.DEFINE_integer("block_seq_len", 288, "Document sequence length.")
flags.DEFINE_integer("batch_size", 4096, "Batch size.")
flags.DEFINE_boolean("use_tpu", False, "Use TPU model.")
flags.DEFINE_string("master", None, "Optional master address.")
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

FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
  """Model function."""
  del labels, params
  encoder_module = hub.Module(FLAGS.retriever_module_path)
  block_emb = encoder_module(
      inputs=dict(
          input_ids=features["block_ids"],
          input_mask=features["block_mask"],
          segment_ids=features["block_segment_ids"]),
      signature="projected")
  predictions = dict(block_emb=block_emb)
  return tf_estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)


def parse_examples(serialized_example):
  """Make retrieval examples."""
  feature_spec = dict(
      title_ids=tf.FixedLenSequenceFeature([], tf.int64, True),
      token_ids=tf.FixedLenSequenceFeature([], tf.int64, True))
  features = tf.parse_single_example(serialized_example, feature_spec)
  features = {k: tf.cast(v, tf.int32) for k, v in features.items()}
  tokenizer = bert_utils.get_tokenizer(FLAGS.retriever_module_path)
  cls_id, sep_id = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
  block_ids, block_mask, block_segment_ids = bert_utils.pad_or_truncate_pair(
      token_ids_a=features["title_ids"],
      token_ids_b=features["token_ids"],
      sequence_length=FLAGS.block_seq_len,
      cls_id=cls_id,
      sep_id=sep_id)
  return dict(
      block_ids=block_ids,
      block_mask=block_mask,
      block_segment_ids=block_segment_ids)


def input_fn(params):
  """An input function satisfying the tf.estimator API."""
  compression_type = "GZIP" if FLAGS.examples_path.endswith(".gz") else ""
  examples_paths = sorted(tf.io.gfile.glob(FLAGS.examples_path))
  dataset = tf.data.TFRecordDataset(
      examples_paths,
      compression_type=compression_type,
      buffer_size=16 * 1024 * 1024)
  dataset = dataset.map(parse_examples, num_parallel_calls=FLAGS.num_threads)

  # Repeat just one extra batch to make sure we don't lose any remainder
  # blocks that might be dropped.
  dataset = dataset.concatenate(dataset.take(params["batch_size"]))

  dataset = dataset.batch(params["batch_size"], drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset


def main(_):
  encoded_path = os.path.join(FLAGS.retriever_module_path, "encoded",
                              "encoded.ckpt")
  tf.logging.info("Embeddings will be written to %s", encoded_path)

  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None
  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      tpu_config=tf_estimator.tpu.TPUConfig(iterations_per_loop=1000))
  estimator = tf_estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size)

  start_time = time.time()
  all_block_emb = None
  i = 0
  for outputs in estimator.predict(input_fn=input_fn):
    if i == 0:
      all_block_emb = np.zeros(
          shape=(FLAGS.num_blocks, outputs["block_emb"].shape[-1]),
          dtype=np.float32)
    if i >= FLAGS.num_blocks:
      break
    all_block_emb[i, :] = outputs["block_emb"]
    i += 1
    if i % 1000 == 0:
      elapse_time = time.time() - start_time
      examples_per_second = i / elapse_time
      remaining_minutes = ((FLAGS.num_blocks - i) / examples_per_second) / 60
      tf.logging.info(
          "[%d] examples/sec: %.2f, "
          "elapsed minutes: %.2f, "
          "remaining minutes: %.2f", i, examples_per_second, elapse_time / 60,
          remaining_minutes)
  tf.logging.info("Expected %d rows, found %d rows", FLAGS.num_blocks, i)
  tf.logging.info("Saving block embedding to %s...", encoded_path)
  scann_utils.write_array_to_checkpoint("block_emb", all_block_emb,
                                        encoded_path)
  tf.logging.info("Done saving block embeddings.")


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
