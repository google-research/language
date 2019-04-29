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
"""Misc utilities for NQL."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import time

from language.nql import nql
import tensorflow as tf



class Model(object):
  """Help for building Estimator-friendly NQL models.

  To use this, subclass it, and implement config_* methods, so as to define a
  procedure for building an NQL model that can be used with tf.Estimators.

  Specifically you implement

  - config_context, which will configure a freshly-created NeuralQueryContext
  object.  Usually this means declaring relations and types, loading KG
  triples, and etc.

  - config_model_prediction, which will configure a freshly-created Model.
  Usually this means adding attributes to the model that correspond to NQL
  or Tensorflow operations to perform the inference done by the model.
  You can assume that model.context points to the appropriate
  NeuralQueryContext,  configured by config_context.

  - config_model_training, which adds additional attributes corresponding to a
  loss function and optimization step.  You can assume here that
  config_model_prediction has been run on this model.

  - config_model_evaluation, which adds additional attributes for
  measures like accuracy, etc.  You can assume all the other configuration
  has been done. The metrics should be from the tf.metrics package
  (or be compatible - i.e., be pairs of metric_uodate_op, metric).

  Each config_model_* step adds attributes to the Model instance indicating
  the final outputs of this phase: specifically config_model_prediction
  adds the attribute model.predictions, config_model_evaluation
  adds the attribute model.evaluations, and config_model_training
  adds model.loss and model.train_op.  (By 'attribute' above we mean
  instance variables of the Model object.)
  """

  def __init__(self):
    # these should all be set to real values in a functioning instance
    self.context = None
    self.predictions = None
    self.train_op = self.loss = None
    self.evaluations = None
    # default progress counters used in training
    self.num_batches = 0
    self.num_examples = 0
    self.total_loss = 0

    def default_training_callback(fd,
                                  loss, elapsed_time):
      """Called after each training step is executed.

      Arguments:
        fd: feed dictionary used for the training step.
        loss: model's loss on the minibatch just used.
        elapsed_time: time that has elapsed since training started.

      Returns:
        string that is a status update.
      """
      # All batches have the same shape. Pick the first one.
      minibatch_size = list(fd.values())[0].shape[0]
      self.num_examples += minibatch_size
      self.num_batches += 1
      self.total_loss += loss
      return ('%d examples in %.2f sec batch loss %.4f avg loss %.4f' %
              (self.num_examples, elapsed_time, loss,
               self.total_loss / self.num_batches))

    self.training_callback = default_training_callback


class ModelBuilder(object):
  """Help for building Estimator-friendly NQL models.

  To use this, subclass it, and implement config_* methods, so as to define a
  procedure for building an NQL model that can be used with tf.Estimators.

  Specifically you implement

  - config_context, which will configure a freshly-created NeuralQueryContext
  object.  Usually this means declaring relations and types, loading KG
  triples, and etc.

  - config_model_prediction, which will configure a freshly-created model.
  Usually this means adding attributes to the model that correspond to NQL
  or Tensorflow operations to perform the inference done by the model.
  You can assume that model.context points to the appropriate context,
  configured by config_context.

  - config_model_training, which adds additional attributes corresponding to a
  loss function and optimization step.  You can assume here that
  config_model_prediction has been run on this model.

  - config_model_evaluation, which adds additional attributes for
  measures like accuracy, etc.  You can assume all the other configuration
  has been done. The metrics should be from the tf.metrics package
  (or be compatible).

  Each of the config_model_* steps adds some attributes that indicate
  the final outputs of this phase: specifically config_model_prediction
  adds the attribute model.predictions, config_model_evaluation
  adds the attribute model.evaluations, and config_model_training
  adds model.loss and model.train_op.
  """

  def build_context(self, params=None):
    """Create a new NeuralQueryContext and configure it.

    Args:
      params: optional parameters to be passed to config_context

    Returns:
      The newly configured context.
    """
    context = nql.NeuralQueryContext()
    self.config_context(context, params)
    return context

  def build_model(self, feature_ph_dict, labels_ph, params=None, context=None):
    """Construct and return a Model.

    Args:
      feature_ph_dict: maps feature names to placeholders that will hold the
        corresponding inputs.
      labels_ph: a placeholder that will hold the target labels
      params: optional parameters to be passed to the config_* methods.
      context: if provided, use instead of building a fresh context

    Returns:
      a fully configured Model, where model.context is a
      freshly-built context produced by self.build_context().
    """
    model = Model()
    model.context = context or self.build_context(params=params)
    self.config_model(model, feature_ph_dict, labels_ph, params=params)
    self.check_model_completeness(model)
    return model

  def build_model_fn(self):
    """Return a function suitable for use in creating an Estimator.

    Will call self.build_model

    Returns:
      a Python function
    """

    def model_fn(features, labels, mode, params):
      """Estimator model_fn produced by ModelBuilder.

      Args:
        features: passed to config_model_prediction
        labels: passed to config_model_[training|evaluation]
        mode: from tf.estimator.ModeKeys
        params: dict of options passed to config_* methods
      """
      # initialize and partially configure the model
      m = Model()
      m.context = self.build_context(params=params)

      self.config_model_prediction(m, features, params=params)
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=m.predictions)

      self.config_model_training(m, labels, params)
      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, train_op=m.train_op, loss=m.loss)

      self.config_model_evaluation(m, labels, params)
      if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=m.loss, eval_metric_ops=m.evaluations)

      raise ValueError('illegal mode %r' % mode)

    return model_fn

  def build_estimator(self, model_dir=None, params=None):
    """Produce an Estimator for this Model.

    Args:
      model_dir: passed in to Estimator - location of tmp files
        used by Estimator to checkpoint models
      params: passed in to estimator - dict of model_fn parameters

    Returns:
      a tf.estimator.Estimator
    """
    return tf.estimator.Estimator(
        model_fn=self.build_model_fn(), model_dir=model_dir, params=params)

  def check_model_completeness(self, model):
    """Verify that the model has been fully configured.

    Args:
      model: a fully-configured Model
    """
    for attr in ['context', 'predictions', 'train_op', 'loss', 'evaluations']:
      if getattr(model, attr, None) is None:
        raise ValueError('model has no %s set' % attr)

  # these are the abstract routines to specify when you subclass a model

  def config_model(self, model, feature_ph_dict, labels_ph, params=None):
    """Configure an existing Model.

    model.context should be already be set when this is called.

    Args:
      model: model to configure
      feature_ph_dict: maps feature names to placeholders that will hold the
        corresponding inputs.
      labels_ph: a placeholder that will hold the target labels
      params: optional parameters to be passed to the config_* methods.
    """

    self.config_model_prediction(model, feature_ph_dict, params=params)
    self.config_model_training(model, labels_ph, params=params)
    self.config_model_evaluation(model, labels_ph, params=params)

  def config_context(self, context, params=None):
    """Configure a context object, to use to help build the model.

    Subclass this for a particular modeling task.

    Args:
      context: a NeuralQueryContext
      params: optional parameters
    """
    raise NotImplementedError

  def config_model_prediction(self, model, feature_ph_dict, params=None):
    """Add additional attributes to model which make predictions.

    The model's inputs will be tf.Placeholders named in the feature_ph_dict
    dictionary, probably coerced into nql.  It will produce some output values,
    which are are specified by the setting model.predictions to dictionary
    mapping strings to model attributes.

    Args:
      model: model to configure
      feature_ph_dict: maps feature names to placeholders that will hold the
        corresponding inputs.
      params: optional parameters
    """
    raise NotImplementedError

  def config_model_training(self, model, labels_ph, params=None):
    """Add additional attributes to the model which allow training.

    These should include at least model.loss and model.train_op.  For use with
    Estimators, model.train_op should include the option
    global_step=tf.train.get_global_step()).

    Args:
      model: model to configure
      labels_ph: a placeholder that will hold the target labels
      params: optional parameters
    """
    raise NotImplementedError

  def config_model_evaluation(self, model, labels_ph, params=None):
    """Add additional attributes to the model which allow evaluation.

    This should also set model.evaluations to an appropriate dictionary.
    Args:
      model: a partly Model for which prediction has been configured.
      labels_ph: a placeholder that will hold the target labels
      params: optional parameters
    """
    raise NotImplementedError


class Trainer(object):
  """Collects methods for training and testing Models."""

  def __init__(self,
               session,
               model,
               feature_ph_dict,
               labels_ph,
               initialize=True):
    """Create a Trainer object for a task.

    Args:
      session: a tf.Session, used to run the dset's iterator
      model: a Model
      feature_ph_dict: maps feature names to placeholders that will hold the
        corresponding inputs.
      labels_ph: a placeholder that will hold the target labels
      initialize: If true, run initializers that erase current model parameters.
    """
    self.session = session
    self.model = model
    self.feature_ph_dict = feature_ph_dict
    self.labels_ph = labels_ph
    if initialize:
      session.run([
          tf.global_variables_initializer(),
          tf.local_variables_initializer(),
          tf.tables_initializer()
      ])

  def as_read_head(self, dset):
    """Get the next minibatch from a dataset.

    Arguments:
      dset: a tf.data.Dataset

    Returns:
      a TF expression that evaluates to the next minibatch.
    """
    return tf.data.make_one_shot_iterator(dset).get_next()

  def feed_dict_iterator(self, dset):
    """Iterator over feed_dict dictionaries.

    Args:
      dset: a tf.data.Dataset

    Yields:
      for each value produced by the datasets read's head, a dictionary mapping
      names of all placeholders in feature_ph_dict or labels_ph to appropriate
      alues.
    """
    read_head = self.as_read_head(dset)
    try:
      while True:
        (feature_val_dict, labels_val) = self.session.run(read_head)
        feed_dict = {self.labels_ph.name: labels_val}
        for feature_name, feature_val in feature_val_dict.items():
          feature_ph = self.feature_ph_dict[feature_name]
          feed_dict[feature_ph.name] = feature_val
        yield feed_dict
    except tf.errors.OutOfRangeError:
      pass

  def train(self, dset):
    """Train the model on this dataset over the examples in a dataset.

    Args:
      dset: a tf.data.Dataset
    """
    start_time = time.time()
    for fd in self.feed_dict_iterator(dset):
      _, latest_loss = self.session.run([self.model.train_op, self.model.loss],
                                        feed_dict=fd)
      if self.model.training_callback is not None:
        status = self.model.training_callback(fd, latest_loss,
                                              time.time() - start_time)
        if status:
          tf.logging.info(status)

  def evaluate(self, dset):
    """Test the model on this dataset over the examples in a dataset.

    Args:
      dset: a tf.data.Dataset

    Returns:
      a dictionary of results from model.evaluations
    """
    named_metrics = sorted(self.model.evaluations.items())
    metrics = [metric for (_, metric) in named_metrics]
    for fd in self.feed_dict_iterator(dset):
      self.session.run(metrics, feed_dict=fd)
    # tf metrics are pairs: current_value,update_op
    result = {}
    for (name, metric) in named_metrics:
      result[name] = self.session.run(metric[0])
    return result


def labels_of_top_ranked_predictions_in_batch(labels, predictions):
  """Applying tf.metrics.mean to this gives precision at 1.

  Args:
    labels: minibatch of dense 0/1 labels, shape [batch_size rows, num_classes]
    predictions: minibatch of predictions of the same shape

  Returns:
    one-dimension tensor top_labels, where top_labels[i]=1.0 iff the
    top-scoring prediction for batch element i has label 1.0
  """
  indices_of_top_preds = tf.cast(tf.argmax(input=predictions, axis=1), tf.int32)
  batch_size = tf.reduce_sum(input_tensor=tf.ones_like(indices_of_top_preds))
  row_indices = tf.range(batch_size)
  thresholded_labels = tf.where(
      labels > 0.0,
      tf.ones_like(labels),
      tf.zeros_like(labels))
  label_indices_to_gather = tf.transpose(
      a=tf.stack([row_indices, indices_of_top_preds]))
  return tf.gather_nd(thresholded_labels, label_indices_to_gather)
