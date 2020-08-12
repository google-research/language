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
"""Scann utils."""
import tensorflow.compat.v1 as tf

# pylint: disable=g-import-not-at-top
try:
  from scann import ScannBuilder
except ImportError:
  tf.logging.warn("scann is not available.")
# pylint: enable=g-import-not-at-top


def write_array_to_checkpoint(var_name, np_db, checkpoint_path):
  """Write np array to checkpoint."""
  with tf.Graph().as_default():
    init_value = tf.py_func(lambda: np_db, [], tf.float32)
    init_value.set_shape(np_db.shape)
    tf_db = tf.get_variable(var_name, initializer=init_value)
    saver = tf.train.Saver([tf_db])
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      saver.save(session, checkpoint_path)


def load_scann_searcher(var_name,
                        checkpoint_path,
                        num_neighbors,
                        dimensions_per_block=2,
                        num_leaves=1000,
                        num_leaves_to_search=100,
                        training_sample_size=10000):
  """Load scann searcher from checkpoint."""
  with tf.device("/cpu:0"):
    np_db = tf.train.load_checkpoint(checkpoint_path).get_tensor(var_name)
    init_db = tf.py_func(lambda: np_db, [], tf.float32)
    init_db.set_shape(np_db.shape)
    tf_db = tf.get_local_variable(var_name, initializer=init_db)

    builder = ScannBuilder(
        db=tf_db,
        num_neighbors=num_neighbors,
        distance_measure="dot_product")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=training_sample_size)
    builder = builder.score_ah(dimensions_per_block=dimensions_per_block)
    searcher = builder.create_tf()
  return tf_db, searcher
