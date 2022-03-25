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
"""Encode wikipedia."""
import time

from absl import app
from absl import flags
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_hub as hub

flags.DEFINE_string("retriever_module_path", None,
                    "Path to the retriever TF-Hub module.")
flags.DEFINE_string("suffix", "", "Output file name suffix.")
flags.DEFINE_string("output_path", None, "Path to outputs.")
flags.DEFINE_string("examples_path", None, "Path to tf.train.Examples")
flags.DEFINE_integer(
    "num_blocks", 8841823, "Expected number of MSMARCO blocks: "
    "8841823 passages, 6980 dev queries and 502939 train queries.")
flags.DEFINE_integer("num_threads", 48, "Num threads for input reading.")
flags.DEFINE_integer("num_vec_per_block", 1, "Num of vectors per block.")
flags.DEFINE_integer("block_seq_len", 288, "Document sequence length.")
flags.DEFINE_integer("batch_size", 4096, "Batch size.")
flags.DEFINE_boolean("use_tpu", False, "Use TPU model.")
flags.DEFINE_boolean("encode_query", True,
                     "True to encode query, otherwise encode passage")
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


def model_fn_query(features, labels, mode, params):
  """Model function."""
  del labels, params
  encoder_module = hub.Module(FLAGS.retriever_module_path)
  block_emb = encoder_module(
      inputs=dict(
          input_ids=features["block_ids"],
          input_mask=features["block_mask"],
          segment_ids=features["block_segment_ids"],
      ),
      signature="query_output",
      as_dict=True)["output_layer"]
  predictions = dict(block_emb=block_emb, key=features["key"])
  return tf_estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)


def model_fn_passage(features, labels, mode, params):
  """Model function."""
  del labels, params
  encoder_module = hub.Module(FLAGS.retriever_module_path)
  block_emb = encoder_module(
      inputs=dict(
          input_ids=features["block_ids"],
          input_mask=features["block_mask"],
          segment_ids=features["block_segment_ids"],
      ),
      signature="passage_output",
      as_dict=True)["output_layer"]
  predictions = dict(block_emb=block_emb, key=features["key"])
  return tf_estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)


def pad_or_truncate_pair(token_ids, sequence_length):
  """Pad or truncate pair."""
  token_ids = token_ids[:sequence_length]
  truncated_len = tf.size(token_ids)
  padding = tf.zeros([sequence_length - truncated_len], tf.int32)
  token_ids = tf.concat([token_ids, padding], 0)
  mask = tf.concat([tf.ones([truncated_len], tf.int32), padding], 0)
  if FLAGS.encode_query:
    segment_ids = tf.concat([tf.zeros([truncated_len], tf.int32), padding], 0)
  else:
    segment_ids = tf.concat([tf.ones([truncated_len], tf.int32), padding], 0)
  token_ids = tf.ensure_shape(token_ids, [sequence_length])
  mask = tf.ensure_shape(mask, [sequence_length])
  segment_ids = tf.ensure_shape(segment_ids, [sequence_length])

  return token_ids, mask, segment_ids


def parse_examples(serialized_example):
  """Make retrieval examples."""
  feature_spec = dict(
      input_ids=tf.FixedLenSequenceFeature([], tf.int64, True),
      key=tf.FixedLenSequenceFeature([], tf.int64, True))
  features = tf.parse_single_example(serialized_example, feature_spec)
  features = {k: tf.cast(v, tf.int32) for k, v in features.items()}
  block_ids, block_mask, block_segment_ids = pad_or_truncate_pair(
      token_ids=features["input_ids"], sequence_length=FLAGS.block_seq_len)
  key = tf.ensure_shape(features["key"], [1])
  return dict(
      block_ids=block_ids,
      block_mask=block_mask,
      block_segment_ids=block_segment_ids,
      key=key)


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


def save_to_h5py(outfn, output_array, field_name):
  h5f = h5py.File(outfn, "w")
  h5f.create_dataset(field_name, data=output_array)
  h5f.close()


def main(_):
  if FLAGS.encode_query:
    encoding_path = FLAGS.output_path + "/queries_" + FLAGS.suffix + "_encodings.h5py"
    passage_id_path = FLAGS.output_path + "/queries_" + FLAGS.suffix + "_ids.h5py"
  else:
    encoding_path = FLAGS.output_path + "/passage_encodings.h5py"
    passage_id_path = FLAGS.output_path + "/passage_ids.h5py"
  if not tf.io.gfile.exists(FLAGS.output_path):
    tf.io.gfile.makedirs(FLAGS.output_path)
  tf.logging.info("Embeddings will be written to %s", encoding_path)
  tf.logging.info("Passage ID will be written to %s", passage_id_path)

  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None
  run_config = tf_estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      tpu_config=tf_estimator.tpu.TPUConfig(iterations_per_loop=1000))
  if FLAGS.encode_query:
    model_fn = model_fn_query
  else:
    model_fn = model_fn_passage
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
  num_vec = FLAGS.num_blocks * FLAGS.num_vec_per_block
  for outputs in estimator.predict(input_fn=input_fn):
    if i == 0:
      all_block_emb = np.zeros(
          shape=(num_vec, outputs["block_emb"].shape[-1]), dtype=np.float32)
      all_paragraph_ids = np.zeros(shape=(num_vec), dtype=np.int32)
    if i >= num_vec:
      break
    for k in range(FLAGS.num_vec_per_block):
      all_block_emb[i, :] = outputs["block_emb"][k, :]
      all_paragraph_ids[i] = outputs["key"][0]
      i += 1
    if i % 1000 == 0:
      elapse_time = time.time() - start_time
      examples_per_second = i / elapse_time
      remaining_minutes = ((num_vec - i) / examples_per_second) / 60
      tf.logging.info(
          "[%d] examples/sec: %.2f, "
          "elapsed minutes: %.2f, "
          "remaining minutes: %.2f", i, examples_per_second, elapse_time / 60,
          remaining_minutes)
  tf.logging.info("Expected %d rows, found %d rows", FLAGS.num_blocks, i)
  tf.logging.info("Saving block embedding to %s...", encoding_path)
  save_to_h5py(encoding_path, all_block_emb, "encodings")
  tf.logging.info("Saving passage ids to %s...", passage_id_path)
  save_to_h5py(passage_id_path, all_paragraph_ids, "ids")
  tf.logging.info("Done saving block embeddings and passage ids.")


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
