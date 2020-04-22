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
r"""Use NQL to learn models for the nell995 dataset.

python nell995.py --rootdir=nell995/ --task=concept_athletehomestadium --num_
hops=5 --epochs=50

You should specify the Nell995 sub task in the command line. num_hops is
a hyper-parameter that determines the length of inferential chain. You may
also want to change the number of epochs to train. Set epochs to be large
if there are very few training examples for the sub task.

all tasks:
concept_agentbelongstoorganization
concept_athletehomestadium
concept_athleteplaysforteam
concept_athleteplaysinleague
concept_athleteplayssport
concept_organizationheadquarteredincity
concept_organizationhiredperson
concept_personborninlocation
concept_personleadsorganization
concept_teamplaysinleague
concept_teamplayssport
concept_worksfor
"""

import time

import nql
from nql import dataset
from nql import util
import tensorflow.compat.v1 as tf

flags = tf.flags
logging = tf.logging
tf.compat.v1.disable_eager_execution()

DEFAULT_DATA_DIR = None
SEED = 48749210
FLAGS = flags.FLAGS

flags.DEFINE_string('rootdir', DEFAULT_DATA_DIR, 'path to data directory')
flags.DEFINE_string(
    'action', 'expt_ph',
    'Must be "train_ph", "test_ph", "expt_ph", default "expt_ph"')
flags.DEFINE_string('task', 'concept_athletehomestadium', 'nell995 task name')
flags.DEFINE_string('kg_file', 'kb.cfacts',
                    'File in rootdir containing the knowledge graph')
flags.DEFINE_string('train_file', 'train.exam',
                    'File in rootdir containing the training data')
flags.DEFINE_string('test_file', 'test.exam',
                    'File in rootdir containing the test data')
# we randomly sample a small number of examples at training for online eval
flags.DEFINE_string('dev_file', 'test.exam',
                    'File in rootdir containing the dev data')
flags.DEFINE_string('relation_file', 'rels.txt',
                    'File in rootdir containing relations')
flags.DEFINE_string('query_file', 'queries.txt',
                    'File in rootdir containing queries of interest')

flags.DEFINE_integer('num_hops', 5, '# hops to use in model')
flags.DEFINE_integer('epochs', 10, 'num epochs to run')
flags.DEFINE_integer('num_train', 2000, 'train examples to use')
flags.DEFINE_integer('online_eval_size', 500,
                     'size of small set test on every few minibatches')
flags.DEFINE_integer(
    'steps_between_evals', 20,
    'for action=expt_ph, number of minibatches to train on'
    ' between evaluation on test data')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('gradient_clip', 5.0, 'if nonzero use clip_by_global_norm')
flags.DEFINE_integer('minibatch_size', 10, 'minibatch_size')

flags.DEFINE_string('checkpoint_dir', '/tmp/metaqa_chkpts',
                    'where to save training checkpoints')
flags.DEFINE_string('module_handle',
                    'https://tfhub.dev/google/nnlm-en-dim128/1',
                    'text embedding module')
flags.DEFINE_integer('num_text_dims', 128,
                     'dimension of output of text embedding module')

##############################################################################
# building the model
##############################################################################


class Nell995Builder(util.ModelBuilder):
  """A ModelBuilder for this task."""

  def config_context(self, context, params=None):

    # declare the KG relations
    relation_filename = '%s/%s/%s' % (FLAGS.rootdir, FLAGS.task,
                                      FLAGS.relation_file)

    for line in tf.io.gfile.GFile(relation_filename):
      line = line.strip()
      rel, subj_type, obj_type = line.split('\t')
      context.declare_relation(rel, subj_type, obj_type)

    # we will also use NQL for a distance flag, which indicates if the relation
    # is followed forward or backward
    context.extend_type('distance_t', [str(h) for h in range(FLAGS.num_hops)])

    # load the lines from the KG
    start_time = time.time()
    kg_filename = '%s/%s/%s' % (FLAGS.rootdir, FLAGS.task, FLAGS.kg_file)
    logging.info('loading KG from %s', kg_filename)
    with tf.gfile.GFile(kg_filename) as fp:
      context.load_kg(files=fp)
    logging.info('loaded kg in %.3f sec', (time.time() - start_time))

    # finally extend the KG to allow us to use relation names as variables
    context.construct_relation_group('rel_ent2ent_g', 'entity_t', 'entity_t')

  def config_model_prediction(self, model, feature_ph_dict, params=None):
    # we use the context a bit, so let's be brief
    c = model.context

    q_emb = tf.get_variable('q_emb', [1, FLAGS.num_text_dims])
    batch_size = tf.shape(feature_ph_dict['seeds'])[0]
    model.query_encoding = tf.tile(q_emb, multiples=[batch_size, 1])

    # we will want to be able to map the encoded text to a set of entities in a
    # given type. This function returns an NQL expression, over the specified
    # type, which is formed by running the text encoding through a learned
    # linear map to get the number of dimensions right, and then applying a
    # softmax
    def linear_text_remapper(type_name):
      num_input_dims = FLAGS.num_text_dims
      num_output_dims = c.get_max_id(type_name)
      initializer = tf.glorot_uniform_initializer()(
          [num_input_dims, num_output_dims])
      weight_matrix = tf.Variable(initializer)
      remapped_text = tf.matmul(model.query_encoding, weight_matrix)
      return c.as_nql(remapped_text, type_name)

    # the seeds, i.e. entities in the question, are the other
    # input to the model.  by convention inputs are passed in
    # in tensorflow format, so we'll wrap them as NQL
    model.seeds = c.as_nql(feature_ph_dict['seeds'], 'entity_t')
    # finally we define the NQL part of the model
    model.rels = [
        linear_text_remapper('rel_ent2ent_g') for _ in range(FLAGS.num_hops)
    ]
    model.dists = [
        linear_text_remapper('distance_t') for _ in range(FLAGS.num_hops)
    ]

    # start with seeds and build a model that follows exactly num_hops hops
    # num_hops is a hyper-parameter to tune for each task
    model.raw_y = [model.seeds]
    for h in range(FLAGS.num_hops):
      prev_raw_y = model.raw_y[-1]
      model.raw_y.append(
          prev_raw_y.follow(model.rels[h], +1).if_any(
              model.dists[h] & c.one(str(h), 'distance_t'))
          | prev_raw_y)

    # we add another rule to accomodate for shorter paths
    second_num_hops = FLAGS.num_hops - 2
    model.raw_y2 = [model.seeds]
    for h in range(second_num_hops):
      prev_raw_y = model.raw_y2[-1]
      model.raw_y2.append(
          prev_raw_y.follow(model.rels[h], +1).if_any(
              model.dists[h] & c.one(str(h), 'distance_t'))
          | prev_raw_y)

    # predictions from two paths are combined as final prediction
    model.predicted_y = nql.nonneg_softmax(model.raw_y[-1].tf +
                                           model.raw_y2[-1].tf)
    model.predictions = dict([('answer', model.raw_y[-1])])

  def config_model_training(self, model, labels_ph, params=None):
    model.loss = nql.nonneg_crossentropy(model.predicted_y, labels_ph)
    logging.info('learning rate %f', FLAGS.learning_rate)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    if FLAGS.gradient_clip > 0:
      logging.info('clipping gradients to %f', FLAGS.gradient_clip)
      gradients, variables = zip(*optimizer.compute_gradients(loss=model.loss))
      gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      model.train_op = optimizer.apply_gradients(
          zip(gradients, variables), global_step=tf.train.get_global_step())
    else:
      logging.info('no gradient clipping')
      model.train_op = optimizer.minimize(
          loss=model.loss, global_step=tf.train.get_global_step())

  def config_model_evaluation(self, model, labels_ph, params=None):
    model.accuracy = tf.metrics.accuracy(
        tf.argmax(labels_ph, axis=1), tf.argmax(model.predicted_y, axis=1))
    model.top_labels = util.labels_of_top_ranked_predictions_in_batch(
        labels_ph, model.predicted_y)
    model.precision_at_one = tf.metrics.mean(model.top_labels)
    model.loss_as_metric = tf.metrics.mean(model.loss)
    model.evaluations = {
        'precision@1': model.precision_at_one,
        'loss': model.loss_as_metric,
    }


def make_dset_fn(context, filename, shuffle=False, epochs=1, n_take=0):
  """Return a data_fn usable for an estimator.

  Args:
    context: a NeuralQueryContext
    filename: name of a file in FLAGS.rootdir
    shuffle: if True, shuffle the data
    epochs: number of times to repeat the data
    n_take: size of subset of examples for training/testing

  Returns:
    a function f so that f() returns a tf.data.Dataset
  """
  data_specs = [str, str, 'entity_t', 'entity_t']
  full_filename = '%s/%s/%s' % (FLAGS.rootdir, FLAGS.task, filename)
  logging.info('dataset from %s minibatch %d epochs %d shuffle %r take %d',
               filename, FLAGS.minibatch_size, epochs, shuffle, n_take)

  def dset_fn():
    dset = dataset.tuple_dataset(context, full_filename, data_specs)
    if shuffle:
      dset = dset.shuffle(1000, seed=SEED)
    if n_take > 0:
      dset = dset.take(n_take)
    dset = dset.batch(FLAGS.minibatch_size)
    dset = dset.repeat(epochs)
    return dset.map(lambda _, q, s, ans: ({'seeds': s}, ans))

  return dset_fn


def make_feature_label_ph_pair(context):
  """Return a dict of feature placeholders paired with label placeholder.

  Args:
    context: a NeuralQueryContext

  Returns:
    a pair feature_ph_dict, labels_ph
  """
  feature_ph_dict = {'seeds': dataset.placeholder_for_type(context, 'entity_t')}
  labels_ph = dataset.placeholder_for_type(context, 'entity_t')
  return feature_ph_dict, labels_ph


def main(unused_args):
  logging.set_verbosity(logging.INFO)
  tf.random.set_random_seed(SEED)

  logging.info('eps %f train %s test %s dev %s', FLAGS.learning_rate,
               FLAGS.train_file, FLAGS.test_file, FLAGS.dev_file)

  # set up the builder, context, and datasets
  builder = Nell995Builder()
  context = builder.build_context()

  train_dset_fn = make_dset_fn(
      context,
      FLAGS.train_file,
      shuffle=True,
      epochs=FLAGS.epochs,
      n_take=FLAGS.num_train)
  test_dset_fn = make_dset_fn(context, FLAGS.test_file, shuffle=True, epochs=1)

  if FLAGS.action == 'train_ph':
    feature_ph_dict, labels_ph = make_feature_label_ph_pair(context)
    model = builder.build_model(feature_ph_dict, labels_ph, context=context)
    with tf.Session() as session:
      trainer = util.Trainer(session, model, feature_ph_dict, labels_ph)
      trainer.train(train_dset_fn())
      tf.train.Saver().save(session, FLAGS.checkpoint_dir + '/final.chpt')

  elif FLAGS.action == 'test_ph':
    feature_ph_dict, labels_ph = make_feature_label_ph_pair(context)
    model = builder.build_model(feature_ph_dict, labels_ph, context=context)
    with tf.Session() as session:
      tf.train.Saver().restore(session, FLAGS.checkpoint_dir + '/final.chpt')
      trainer = util.Trainer(session, model, feature_ph_dict, labels_ph)
      evaluation = trainer.evaluate(test_dset_fn())
      logging.info('evaluation: %s', evaluation)

  elif FLAGS.action == 'expt_ph':
    feature_ph_dict, labels_ph = make_feature_label_ph_pair(context)
    model = builder.build_model(feature_ph_dict, labels_ph, context=context)

    callback_dset_fn = make_dset_fn(
        context,
        FLAGS.test_file,
        shuffle=True,
        epochs=1,
        n_take=FLAGS.online_eval_size)

    with tf.Session() as session:
      trainer = util.Trainer(session, model, feature_ph_dict, labels_ph)

      evaluation = trainer.evaluate(callback_dset_fn())
      tf.logging.info('before training: %s', evaluation)

      # set up callback to evaluate on dev every so often
      model.num_steps = 0
      old_callback = model.training_callback

      def eval_periodically_callback(fd, latest_loss, elapsed_time):
        # default callback increments model.num_examples
        status = old_callback(fd, latest_loss, elapsed_time)
        model.num_steps += 1
        if (model.num_steps % FLAGS.steps_between_evals) == 0:
          tf.logging.info('running eval on heldout dev set...')
          evaluation = trainer.evaluate(callback_dset_fn())
          tf.logging.info('after %d examples: %s', model.num_examples,
                          evaluation)
        return status

      model.training_callback = eval_periodically_callback

      trainer.train(train_dset_fn())
      evaluation = trainer.evaluate(test_dset_fn())
      tf.logging.info('final evaluation %s', evaluation)
      try:
        tf.train.Saver().save(session, FLAGS.checkpoint_dir + '/final.chpt')
      except ValueError:
        tf.logging.error('fail to save model at %s',
                         FLAGS.checkpoint_dir + '/final.chpt')

  else:
    raise ValueError('illegal action')


if __name__ == '__main__':
  tf.app.run()
