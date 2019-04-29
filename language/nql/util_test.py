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
"""Tests for util."""

import os
import tempfile

from language.nql import dataset
from language.nql import nql
from language.nql import util
import numpy as np
import tensorflow as tf


def tabline(s):
  return "\t".join(s.split(" ")) + "\n"


TRIPPY_KG_LINES = [
    tabline("feature t1 purple"),
    tabline("feature t1 green"),
    tabline("feature t1 red"),
    tabline("feature t2 purple"),
    tabline("feature t2 red"),
    tabline("feature t3 red"),
    tabline("feature t3 black"),
    tabline("feature b1 black"),
    tabline("feature b1 tan"),
    tabline("feature b2 white"),
    tabline("feature b2 grey"),
    tabline("feature b3 black"),
    tabline("feature b3 white"),
    tabline("feature b3 tan"),
    tabline("feature u1 purple"),
    tabline("feature t1 green"),
    tabline("feature u2 green"),
    tabline("feature t2 red"),
    tabline("feature c1 black"),
    tabline("feature b1 grey"),
    tabline("feature c2 tan"),
    tabline("feature c2 grey")
]

TRAIN_DATA_LINES = [
    "t1|trippy", "t2|trippy", "t3|trippy", "b1|boring", "b2|boring", "b3|boring"
]

TEST_DATA_LINES = ["u1|trippy", "u2|trippy", "c1|boring", "c2|boring"]


def simple_tf_dataset(context,
                      tuple_input,
                      x_type,
                      y_type,
                      normalize_outputs=False,
                      batch_size=1,
                      shuffle_buffer_size=1000,
                      feature_key=None,
                      field_separator="\t"):
  """A dataset with just two columns, x and y.

  Args:
    context: a NeuralQueryContext
    tuple_input: passed to util.tuple_dataset
    x_type: type of entities x
    y_type: type of entities y1,...,yk
    normalize_outputs: make the encoding of {y1,...,yk} sum to 1
    batch_size: size of minibatches
    shuffle_buffer_size: if zero, do not shuffle the dataset. Otherwise, this is
      passed in as argument to shuffle
    feature_key: if not None, wrap the x part of the minibatch in a dictionary
      with the given key
    field_separator: passed in to dataset.tuple_dataset

  Returns:
    a tf.data.Dataset formed by wrapping the generator
  """
  dset = dataset.tuple_dataset(
      context,
      tuple_input, [x_type, y_type],
      normalize_outputs=normalize_outputs,
      field_separator=field_separator)
  if shuffle_buffer_size > 0:
    dset = dset.shuffle(shuffle_buffer_size)
  dset = dset.batch(batch_size)
  if feature_key is None:
    return dset
  else:
    wrap_x_in_dict = lambda x, y: ({feature_key: x}, y)
    return dset.map(wrap_x_in_dict)


class TrippyBuilder(util.ModelBuilder):

  def config_context(self, context, params=None):
    context.declare_relation("feature", "instance_t", "feature_t")
    context.declare_relation(
        "indicates", "feature_t", "label_t", trainable=True)
    context.extend_type("label_t", ["trippy", "boring"])
    context.load_kg(lines=TRIPPY_KG_LINES)
    context.set_initial_value(
        "indicates", np.ones(context.get_shape("indicates"), dtype="float32"))

  def config_model_prediction(self, model, feature_ph_dict, params=None):
    model.x = model.context.as_nql(feature_ph_dict["x"], "instance_t")
    model.score = model.x.feature().indicates()
    model.predicted_y = model.score.tf_op(nql.nonneg_softmax)
    model.predictions = {"y": model.predicted_y}

  def config_model_training(self, model, labels_ph, params=None):
    model.labels = model.context.as_tf(labels_ph)
    model.loss = nql.nonneg_crossentropy(model.predicted_y.tf, model.labels)
    optimizer = tf.train.AdagradOptimizer(1.0)
    model.train_op = optimizer.minimize(
        loss=model.loss, global_step=tf.train.get_global_step())

  def config_model_evaluation(self, model, labels_ph, params=None):
    model.accuracy = tf.metrics.accuracy(
        tf.argmax(input=model.labels, axis=1),
        tf.argmax(input=model.predicted_y.tf, axis=1))
    model.top_labels = util.labels_of_top_ranked_predictions_in_batch(
        model.labels, model.predicted_y.tf)
    model.precision_at_one = tf.metrics.mean(model.top_labels)
    model.evaluations = {
        "accuracy": model.accuracy,
        "precision@1": model.precision_at_one
    }


class BaseTester(tf.test.TestCase):

  def setUp(self):
    super(BaseTester, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.context = TrippyBuilder().build_context()

  def make_train_dset(self, num_epochs):
    # need to specify a non-default field separator
    # because tabs are disallowed in test input files
    return simple_tf_dataset(
        self.context,
        TRAIN_DATA_LINES,
        "instance_t",
        "label_t",
        feature_key="x",
        field_separator="|").repeat(num_epochs)

  def make_test_dset(self):
    return simple_tf_dataset(
        self.context,
        TEST_DATA_LINES,
        "instance_t",
        "label_t",
        shuffle_buffer_size=0,
        feature_key="x",
        field_separator="|")


class TestModelBuilder(BaseTester):

  def setUp(self):
    super(TestModelBuilder, self).setUp()
    self.session = tf.Session()

  def check_one_hot(self, m, i, typename):
    self.assertEqual(m.shape, (self.context.get_max_id(typename),))
    self.assertEqual(np.sum(m), 1.0)
    self.assertEqual(m[i], 1.0)

  def test_tf_dataset(self):
    dset1 = simple_tf_dataset(
        self.context,
        TRAIN_DATA_LINES,
        "instance_t",
        "label_t",
        shuffle_buffer_size=0,
        field_separator="|")
    x, y = self.session.run(
        tf.data.make_one_shot_iterator(dset1).get_next())
    self.check_batch(x, 0, "instance_t")
    self.check_batch(y, 0, "label_t")

  def check_batch(self, m, i, typename):
    self.assertEqual(m.shape, (1, self.context.get_max_id(typename)))
    self.assertEqual(np.sum(m), 1.0)
    self.assertEqual(m[0, i], 1.0)

  def test_tf_minibatch_dataset(self):
    dset2 = simple_tf_dataset(
        self.context,
        TRAIN_DATA_LINES,
        "instance_t",
        "label_t",
        batch_size=2,
        shuffle_buffer_size=0,
        field_separator="|")
    x, y = self.session.run(
        tf.data.make_one_shot_iterator(dset2).get_next())
    # check that this is a minibatch containing the first two instances
    self.assertEqual(x.shape[0], 2)
    self.assertEqual(y.shape[0], 2)
    self.assertEqual(x.shape[1], self.context.get_max_id("instance_t"))
    self.assertEqual(y.shape[1], self.context.get_max_id("label_t"))
    self.assertEqual(np.sum(x), 2.0)
    self.assertEqual(np.sum(y), 2.0)
    self.assertEqual(x[0, 0], 1.0)
    self.assertEqual(x[1, 1], 1.0)
    # both of the first two instances are negative
    self.assertEqual(y[0, 0], 1.0)
    self.assertEqual(y[1, 0], 1.0)

  def test_ph_learn(self):

    # build model
    feature_ph_dict = {"x": self.context.placeholder("x", "instance_t")}
    labels_ph = self.context.placeholder("y", "label_t")
    builder = TrippyBuilder()
    model = builder.build_model(feature_ph_dict, labels_ph)
    trainer = util.Trainer(self.session, model, feature_ph_dict, labels_ph)

    # train
    trainer.train(self.make_train_dset(5))

    # check the model fits the train data
    evaluation = trainer.evaluate(self.make_train_dset(1))
    self.assertEqual(evaluation["accuracy"], 1.0)
    self.assertEqual(evaluation["precision@1"], 1.0)

    # try running the model on something
    for inst_name in ["u1", "u2", "c1", "c2"]:
      x = model.context.one_hot_numpy_array(inst_name, "instance_t")
      x_ph = feature_ph_dict["x"]
      fd = {x_ph.name: x}
      y_dict = model.predicted_y.eval(self.session, feed_dict=fd)
      # the u's are class trippy
      if inst_name[0] == "u":
        self.assertGreater(y_dict["trippy"], y_dict["boring"])
      # the c's are class boring but c1 is hard to get
      elif inst_name == "c2":
        self.assertLess(y_dict["trippy"], y_dict["boring"])

    # test the model
    evaluation = trainer.evaluate(self.make_test_dset())
    self.assertGreaterEqual(evaluation["accuracy"], 0.7)
    self.assertGreaterEqual(evaluation["precision@1"], 0.7)

    # test callback
    cb_model = builder.build_model(feature_ph_dict, labels_ph)
    cb_model.loss_history = []

    def my_callback(fd, loss, secs):
      del fd, secs  # unused
      cb_model.loss_history.append(loss)
      return None

    cb_model.training_callback = my_callback
    with tf.Session() as session:
      cb_trainer = util.Trainer(session, cb_model, feature_ph_dict, labels_ph)
      cb_trainer.train(self.make_train_dset(5))
      self.assertEqual(len(cb_model.loss_history), 30)
      self.assertLess(cb_model.loss_history[-1], 0.05)

  def test_estimator_learn(self):

    def train_input_fn():
      return self.make_train_dset(5)

    def test_input_fn():
      return self.make_test_dset()

    estimator = TrippyBuilder().build_estimator()
    estimator.train(input_fn=train_input_fn)
    evaluation = estimator.evaluate(input_fn=train_input_fn)
    self.assertEqual(evaluation["accuracy"], 1.0)
    self.assertEqual(evaluation["global_step"], 30)
    evaluation = estimator.evaluate(input_fn=test_input_fn)
    self.assertGreater(evaluation["accuracy"], 0.7)
    self.assertGreaterEqual(evaluation["precision@1"], 0.7)


class TestSaveRestore(BaseTester):

  def setUp(self):
    super(TestSaveRestore, self).setUp()
    tmp_dir = tempfile.mkdtemp("util_test")
    self.checkpoint_location_a = os.path.join(tmp_dir, "trippy.ckpt")
    self.checkpoint_location_b = os.path.join(tmp_dir, "trippy2.ckpt")

  def test_est(self):

    def train_input_fn():
      return self.make_train_dset(5)

    def test_input_fn():
      return self.make_test_dset()

    estimator = TrippyBuilder().build_estimator(
        model_dir=self.checkpoint_location_a)
    estimator.train(input_fn=train_input_fn)
    evaluation = estimator.evaluate(input_fn=test_input_fn)
    self.assertGreater(evaluation["accuracy"], 0.7)
    self.assertGreaterEqual(evaluation["precision@1"], 0.7)

  def test_ph(self):

    def try_model_on_test_instances(model, sess, feature_ph_dict):
      trial = {}
      for inst_name in ["u1", "u2", "c1", "c2"]:
        x = model.context.one_hot_numpy_array(inst_name, "instance_t")
        x_ph = feature_ph_dict["x"]
        fd = {x_ph.name: x}
        y_dict = model.predicted_y.eval(sess, feed_dict=fd)
        trial[inst_name] = y_dict["boring"]
      return trial

    # Train and save.
    with tf.Graph().as_default():
      with tf.Session() as sess1:
        builder1 = TrippyBuilder()
        context1 = builder1.build_context()
        feature_ph_dict1 = {"x": context1.placeholder("x", "instance_t")}
        labels_ph1 = context1.placeholder("y", "label_t")
        model1 = builder1.build_model(feature_ph_dict1, labels_ph1)

        trainer1 = util.Trainer(sess1, model1, feature_ph_dict1, labels_ph1)
        trainer1.train(self.make_train_dset(5))
        trial1a = try_model_on_test_instances(model1, sess1, feature_ph_dict1)
        saver1 = tf.train.Saver()
        saver1.save(sess1, self.checkpoint_location_a)

    # Restore, evaluate, train, and save.
    with tf.Graph().as_default():
      with tf.Session() as sess2:
        builder2 = TrippyBuilder()
        context2 = builder2.build_context()
        feature_ph_dict2 = {"x": context2.placeholder("x", "instance_t")}
        labels_ph2 = context2.placeholder("y", "label_t")
        model2 = builder2.build_model(feature_ph_dict2, labels_ph2)
        saver2 = tf.train.Saver()

        trainer2 = util.Trainer(sess2, model2, feature_ph_dict2, labels_ph2)
        saver2.restore(sess2, self.checkpoint_location_a)
        trainer2.evaluate(self.make_test_dset())
        trial2a = try_model_on_test_instances(model2, sess2, feature_ph_dict2)
        self.assertDictEqual(trial1a, trial2a)

        trainer2.train(self.make_train_dset(5))
        saver2.save(sess2, self.checkpoint_location_b)
        trial2b = try_model_on_test_instances(model2, sess2, feature_ph_dict2)
        with self.assertRaises(tf.test.TestCase.failureException):
          self.assertDictEqual(trial2a, trial2b)

    # Restore and evaluate.
    with tf.Graph().as_default():
      with tf.Session() as sess3:
        builder3 = TrippyBuilder()
        context3 = builder3.build_context()
        feature_ph_dict3 = {"x": context3.placeholder("x", "instance_t")}
        labels_ph3 = context3.placeholder("y", "label_t")
        model3 = builder3.build_model(feature_ph_dict3, labels_ph3)
        saver3 = tf.train.Saver()

        trainer3 = util.Trainer(sess3, model3, feature_ph_dict3, labels_ph3)
        saver3.restore(sess3, self.checkpoint_location_b)
        trainer3.evaluate(self.make_test_dset())
        trial3b = try_model_on_test_instances(model3, sess3, feature_ph_dict3)
        self.assertDictEqual(trial2b, trial3b)


if __name__ == "__main__":
  tf.test.main()
