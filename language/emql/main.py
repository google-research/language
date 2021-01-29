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
"""Main script to run EmQL experiments."""

from absl import app
from absl import flags
from language.emql.data_loader import DataLoader
from language.emql.eval import Query2BoxMetrics
from language.emql.model import build_model_fn
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS


ROOT_DIR = './datasets/'


# files and folders
flags.DEFINE_string('root_dir', ROOT_DIR, 'data dir')
flags.DEFINE_string('checkpoint_dir', None, 'checkpoint folder')
flags.DEFINE_string('kb_index', None, 'scam index embedding table')
flags.DEFINE_string('bert_handle',
                    'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                    'bert module')

flags.DEFINE_string('model_name', None, 'model name')
flags.DEFINE_string('load_model_dir', None, 'load model folder')
flags.DEFINE_string('kb_file', 'kb.txt', 'kb filename')
flags.DEFINE_string('vocab_file', None, 'vocab filename')
# model parameters
flags.DEFINE_bool('use_cm_sketch', True, 'if use countmin sketch')
flags.DEFINE_bool('train_entity_emb', False, 'True if train entity emb')
flags.DEFINE_bool('train_relation_emb', False, 'True if train relation emb')
flags.DEFINE_bool('train_bert', False, 'True if train BERT emb')
flags.DEFINE_integer('cm_depth', 20, 'depth of countmin sketch')
flags.DEFINE_integer('cm_width', 200, 'width of countmin sketch')
flags.DEFINE_integer('intermediate_top_k', 20, 'top_k at each step')
# training parameters
flags.DEFINE_integer('epochs', 50, 'number of epochs to run')
flags.DEFINE_integer('batch_size', 64, 'mini-batch size')
flags.DEFINE_integer('relation_emb_size', 64, 'hidden state size')
flags.DEFINE_integer('entity_emb_size', 64, 'hidden state size')
flags.DEFINE_integer('vocab_emb_size', 64, 'hidden state size')

flags.DEFINE_integer('checkpoint_step', 100, 'frequency to save checkpoint')
flags.DEFINE_integer('eval_time', 120, 'frequency to save checkpoint')
flags.DEFINE_integer('max_set', 1000, 'max size of a natural set')
flags.DEFINE_integer('num_online_eval', 100, 'num of batches for online eval')
flags.DEFINE_integer('num_eval', -1, 'num of example to predict')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('gradient_clip', 5.0, 'learning rate')
flags.DEFINE_string('mode', 'train', 'train, eval, pred')
flags.DEFINE_enum('name', None, [
    'membership', 'intersection', 'union', 'set_follow', 'mixture', 'metaqa2',
    'metaqa3', 'webqsp', 'query2box_1c', 'query2box_2c', 'query2box_3c',
    'query2box_2i', 'query2box_3i', 'query2box_ic', 'query2box_ci',
    'query2box_2u', 'query2box_uc'
], 'name of training task')
flags.DEFINE_string('eval_name', None,
                    'name of evaluation task, *only for the mixture task*: '
                    'membership, intersection, union, follow, set_follow')
flags.DEFINE_integer('eval_metric_at_k', 1000, 'precision, recall, F1 at k')


def get_root_dir(name):
  """Return the corresponding root dir given taks name.

  Args:
    name: task name

  Returns:
    root dir
  """
  root_dir = FLAGS.root_dir
  if name == 'metaqa2':
    root_dir += 'MetaQA/2hop/'
  elif name == 'metaqa3':
    root_dir += 'MetaQA/3hop/'
  elif name == 'webqsp':
    root_dir += 'WebQSP/'
  elif name in ['membership', 'intersection', 'union', 'set_follow', 'mixture']:
    root_dir += '/'
  elif name.startswith('query2box'):
    root_dir += '/'
  else:
    raise ValueError('name not recognized')
  return root_dir


def run_model():
  """Run experiment with tf.estimator.

  """
  params = {
      'kb_index': FLAGS.kb_index,
      'cm_width': FLAGS.cm_width,
      'cm_depth': FLAGS.cm_depth,
      'entity_emb_size': FLAGS.entity_emb_size,
      'relation_emb_size': FLAGS.relation_emb_size,
      'vocab_emb_size': FLAGS.vocab_emb_size,
      'max_set': FLAGS.max_set,
      'learning_rate': FLAGS.learning_rate,
      'gradient_clip': FLAGS.gradient_clip,
      'intermediate_top_k': FLAGS.intermediate_top_k,
      'use_cm_sketch': FLAGS.use_cm_sketch,
      'train_entity_emb': FLAGS.train_entity_emb,
      'train_relation_emb': FLAGS.train_relation_emb,
      'bert_handle': FLAGS.bert_handle,
      'train_bert': FLAGS.train_bert,
  }

  data_loader = DataLoader(params,
                           FLAGS.name,
                           get_root_dir(FLAGS.name),
                           FLAGS.kb_file,
                           FLAGS.vocab_file)

  estimator_config = tf.estimator.RunConfig(
      save_checkpoints_steps=FLAGS.checkpoint_step)

  warm_start_settings = tf.compat.v1.estimator.WarmStartSettings(  # pylint: disable=g-long-ternary
      ckpt_to_initialize_from=FLAGS.load_model_dir,
      vars_to_warm_start=['embeddings_mat/entity_embeddings_mat',
                          'embeddings_mat/relation_embeddings_mat'],
      ) if FLAGS.load_model_dir is not None else None

  estimator = tf.estimator.Estimator(
      model_fn=build_model_fn(FLAGS.name, data_loader, FLAGS.eval_name,
                              FLAGS.eval_metric_at_k),
      model_dir=FLAGS.checkpoint_dir + FLAGS.model_name,
      config=estimator_config, params=params,
      warm_start_from=warm_start_settings)

  if FLAGS.mode == 'train':
    train_input_fn = data_loader.build_input_fn(
        name=FLAGS.name, batch_size=FLAGS.batch_size,
        mode='train', epochs=FLAGS.epochs, n_take=-1, shuffle=True)

  eval_input_fn = data_loader.build_input_fn(
      name=FLAGS.name, batch_size=FLAGS.batch_size,
      mode='eval', epochs=1, n_take=FLAGS.num_eval, shuffle=False)

  # Define mode-specific operations
  if FLAGS.mode == 'train':
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    # Busy waiting for evaluation until new checkpoint comes out
    test_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=FLAGS.num_online_eval,
        start_delay_secs=0, throttle_secs=FLAGS.eval_time)
    tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)

  elif FLAGS.mode == 'eval':
    tf_evaluation = estimator.evaluate(eval_input_fn)
    print(tf_evaluation)

  elif FLAGS.mode == 'pred':
    tf_predictions = estimator.predict(eval_input_fn)

    if FLAGS.name.startswith('query2box'):
      task = FLAGS.name.split('_')[-1]
      metrics = Query2BoxMetrics(task, FLAGS.root_dir, data_loader)
    else:
      raise NotImplementedError()

    for tf_prediction in tqdm(tf_predictions):
      metrics.eval(tf_prediction)
    metrics.print_metrics()

  else:
    raise ValueError('mode not recognized: %s' % FLAGS.mode)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_model()


if __name__ == '__main__':
  app.run(main)
