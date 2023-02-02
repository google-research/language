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
"""Continuously refresh document embeddings."""
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from language.common.utils import export_utils
from language.realm import featurization
from language.realm import retrieval
import tensorflow.compat.v1 as tf

flags.DEFINE_string("model_dir", None, "Model directory.")

flags.DEFINE_boolean("use_tpu", True, "Whether to use TPU workers.")

flags.DEFINE_string("tpu_workers", None,
                    "Comma-separated list of TPU worker addresses.")

flags.DEFINE_integer("tpu_cores", None, "Number of TPU cores per worker.")

flags.DEFINE_integer("predict_batch_size", 256,
                     "Embed documents in batches of this size.")

flags.DEFINE_integer("start_at_step", 0,
                     "Minimum training step to start refreshing embeddings.")

flags.DEFINE_string("retrieval_corpus_path", None,
                    "Glob path to the sharded retrieval corpus.")

FLAGS = flags.FLAGS

# Amount of time before checking for a new config.
SLEEP_INTERVAL_S = 30.0


def move_directory(from_dir, to_dir):
  tf.gfile.MakeDirs(os.path.dirname(to_dir))
  tf.gfile.Rename(from_dir, to_dir)


def cleanup_encoded_modules():
  export_utils.clean_tfhub_exports(
      FLAGS.model_dir, hub_prefix="encoded", exports_to_keep=3)


def is_training_done():
  """Checks if model training has completed."""
  # Existence of this file indicates that training is done.
  training_done_filename = os.path.join(FLAGS.model_dir, "TRAINING_DONE")
  return tf.gfile.Exists(training_done_filename)


def load_featurizer():
  """Loads a featurizer from hyperparams specified in model_dir."""
  params_path = os.path.join(FLAGS.model_dir, "estimator_params.json")
  with tf.gfile.GFile(params_path) as f:
    params = json.load(f)

  tokenizer = featurization.Tokenizer(
      vocab_path=params["vocab_path"], do_lower_case=params["do_lower_case"])

  featurizer = featurization.Featurizer(
      query_seq_len=params["query_seq_len"],
      candidate_seq_len=params["candidate_seq_len"],
      num_candidates=params["num_candidates"],
      max_masks=params["max_masks"],
      tokenizer=tokenizer,
      separate_candidate_segments=params["separate_candidate_segments"])

  logging.info("Loaded featurizer.")
  return featurizer


def write_array_to_checkpoint(var_name, array, checkpoint_path):
  """Writes Numpy array to TF checkpoint."""
  with tf.Graph().as_default():
    init_value = tf.py_func(lambda: array, [], tf.float32)
    init_value.set_shape(array.shape)
    var = tf.get_variable(var_name, initializer=init_value)
    saver = tf.train.Saver([var])
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      saver.save(session, checkpoint_path)


def main(_):
  if FLAGS.use_tpu:
    tpu_workers = FLAGS.tpu_workers.split(",")

  doc_shard_paths = sorted(tf.gfile.Glob(FLAGS.retrieval_corpus_path))
  doc_shard_sizes = retrieval.count_tf_records_parallel(
      doc_shard_paths, num_processes=12)

  previous_export_path = None
  while True:
    try:
      current_export_path = export_utils.best_export_path(
          FLAGS.model_dir, best_prefix="tf_hub")
    except tf.errors.NotFoundError as e:
      logging.warn("An error occurred while looking for an exported module: %s",
                   e)
      current_export_path = None

    # If there is no Hub module, or it hasn't changed, try this loop again.
    if (current_export_path is None or
        previous_export_path == current_export_path):
      continue

    # Read the step counter file. This file is written after the directory path
    # has been finalized, so it may appear after current_export_path changes.
    # Therefore, wait a bit, and behave gracefully if it's still not there.
    time.sleep(2.0)
    try:
      step_count_filename = os.path.join(current_export_path or "",
                                         "global_step.txt")
      with tf.gfile.GFile(step_count_filename) as f:
        step_count = int(f.read())
    except (OSError, ValueError):
      step_count = -1  # Check `is_training_done()` below.

    if step_count >= FLAGS.start_at_step:
      # Move the newly exported model to a staging area.
      # Move the entire export directory so that both the "bert" encoder and the
      # query / document "embedder" get moved around together.
      staging_export_path = current_export_path.replace("tf_hub", "temp")

      tf.logging.info("Found new export: %s", current_export_path)
      tf.logging.info("Staging path: %s", staging_export_path)
      move_directory(current_export_path, staging_export_path)

      hub_module_spec = os.path.join(staging_export_path, "embedder")

      featurizer = load_featurizer()
      if FLAGS.use_tpu:
        encoded = retrieval.embed_documents_using_multiple_tpu_workers(  # pytype: disable=wrong-arg-types
            shard_paths=doc_shard_paths,
            shard_sizes=doc_shard_sizes,
            hub_module_spec=hub_module_spec,
            featurizer=featurizer,
            tpu_workers=tpu_workers,
            batch_size=FLAGS.predict_batch_size,
            num_tpu_cores_per_worker=FLAGS.tpu_cores,
        )
      else:
        encoded = retrieval.embed_documents(  # pytype: disable=wrong-arg-types
            shard_paths=doc_shard_paths,
            shard_sizes=doc_shard_sizes,
            hub_module_spec=hub_module_spec,
            featurizer=featurizer,
            batch_size=FLAGS.predict_batch_size,
            tpu_run_config=None,
        )

      # Save the document embeddings to disk.
      logging.info("Encoded shape: %s, dtype: %s", encoded.shape, encoded.dtype)
      encoded_path = os.path.join(hub_module_spec, "encoded", "encoded.ckpt")
      tf.gfile.MakeDirs(os.path.dirname(encoded_path))
      logging.info("Writing to checkpoint %s", encoded_path)
      write_array_to_checkpoint("block_emb", encoded, encoded_path)
      logging.info("Done writing checkpoint.")

      # After new embeddings have been written to disk, move from staging area
      # to permanent commit location.
      commit_export_path = staging_export_path.replace("temp", "encoded")
      tf.logging.info("Commit path: %s", commit_export_path)
      move_directory(staging_export_path, commit_export_path)
      cleanup_encoded_modules()
      previous_export_path = current_export_path
    elif is_training_done():
      # If training is done, no need to re-embed documents.
      return
    else:
      time.sleep(SLEEP_INTERVAL_S)


# Note: internal version of the code overrides this function.
def run_main():
  app.run(main)



if __name__ == "__main__":
  run_main()
