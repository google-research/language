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
"""Scalability test for NQL.

Creates a KG that is an n-by-n 2d grid, which will have n^2 nodes and about 4n^2
edges.  The runs a bunch of n-hop follow commands, eg

x.follow(r).follow(r)....follow(r)
"""

import collections
import random
import sys
import time

from nql import nql
import numpy as np
import scipy
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('vary_n', '',
                       'run for multiple n values in comma-sep list')
tf.flags.DEFINE_string(
    'vary_extra_rels', '',
    'run for multiple num_extra_rels values in comma-sep list')
tf.flags.DEFINE_integer('n', 100, 'size of grid')
tf.flags.DEFINE_integer('max_hops', 2, 'max number of hops to evaluate')
tf.flags.DEFINE_integer('num_extra_rels', 0, 'num extra rels')
tf.flags.DEFINE_integer('num_trials', 10, 'average over this number of trials')
tf.flags.DEFINE_integer('minibatch_size', 128, 'minibatch size')
tf.flags.DEFINE_boolean('details', False,
                        'show all timing/sizes, not just averages')
tf.flags.DEFINE_string('variant', '', 'empty string|mix|sum')
tf.flags.DEFINE_integer('dir', +1, 'follow forward (+1) or backward (-1)')
tf.flags.DEFINE_string('table_tag', 'default_',
                       'added to any tsv tables produced')


class MixtureNQExpr(nql.NeuralQueryExpression):
  """Implements x.follow(r) as sum_i r[i] x.dot(M_i).

  Here r[i] is scalar weight of relation i in vector r, M_i is sparse matrix for
  relation i, and x.dot(M_i) is vector-matrix product.

  This is the 'late mixing' method.
  """

  def _follow_relation_set(self, rel_expr, inverted):
    if not self.context.is_group(rel_expr.type_name):
      raise nql.RelationNameError(rel_expr.type_name,
                                  'Expression type is not a relation group.')
    scope_qualifier = '' if inverted else '_inverse'
    scope = 'follow_group_%s_%s' % (rel_expr.type_name, scope_qualifier)
    with tf.name_scope(scope):
      mixture = None
      for r_id in range(self.context.get_max_id(rel_expr.type_name)):
        r_name = self.context.get_entity_name(r_id, rel_expr.type_name)
        addend = self._follow_named_rel(r_name, inverted) * rel_expr.tf[:, r_id]
        if mixture is None:
          mixture = addend
        else:
          mixture = mixture + addend
      g = self.context.get_group(rel_expr.type_name)
      output_type = self.context.get_range(g.object_rel)
      return self.context.as_nql(mixture, output_type)


class HierarchicalMixtureNQExpr(nql.NeuralQueryExpression):
  """Implements x.follow(r) as sum_i r[i] x.dot(M_i).

  Here r[i] is scalar weight of relation i in vector r, M_i is sparse matrix for
  relation i, and x.dot(M_i) is vector-matrix product.

  This is the 'late mixing' method, but using a tree of operations in the sum_i
  instead of a long list of operations.  Not published because it's not any
  faster.
  """

  def _follow_relation_set(self, rel_expr, inverted):
    if not self.context.is_group(rel_expr.type_name):
      raise nql.RelationNameError(rel_expr.type_name,
                                  'Expression type is not a relation group.')
    scope_qualifier = '' if inverted else '_inverse'
    scope = 'follow_group_%s_%s' % (rel_expr.type_name, scope_qualifier)
    with tf.name_scope(scope):
      # build a binary tree of subexpressions
      mixture_buf = []
      for r_id in range(self.context.get_max_id(rel_expr.type_name)):
        # construct the the next individual relation-follower
        r_name = self.context.get_entity_name(r_id, rel_expr.type_name)
        addend = self._follow_named_rel(r_name, inverted) * rel_expr.tf[:, r_id]
        # add it to the buffer
        mixture_buf.append(addend)
        # clean up the buffer if needed
        n = r_id + 1
        while n % 2 == 0:
          # combine the last two things in the buffer
          a = mixture_buf.pop()
          b = mixture_buf.pop()
          mixture_buf.append(a + b)
          n = n / 2

      # we now have at most log(#relations) items left over in the buffer
      mixture = None
      for addend in mixture_buf:
        if mixture is None:
          mixture = addend
        else:
          mixture = mixture + addend
      # construct and return the final tree
      g = self.context.get_group(rel_expr.type_name)
      output_type = self.context.get_range(g.object_rel)
      return self.context.as_nql(mixture, output_type)


class MatSumNQExpr(nql.NeuralQueryExpression):
  """Implements x.follow(r) as x.dot(sum_i r[i] M_i).

  This is the 'early mixing' method.  It is only correct for minibatches of size
  1.
  """

  def _follow_relation_set(self, rel_expr, inverted):

    if not self.context.is_group(rel_expr.type_name):
      raise nql.RelationNameError(rel_expr.type_name,
                                  'Expression type is not a relation group.')

    scope_qualifier = '' if inverted else '_inverse'
    scope = 'follow_group_%s_%s' % (rel_expr.type_name, scope_qualifier)
    m_is_sparse = True
    with tf.name_scope(scope):
      matsum = None
      for r_id in range(self.context.get_max_id(rel_expr.type_name)):
        r_name = self.context.get_entity_name(r_id, rel_expr.type_name)
        addend = self.context.get_tf_tensor(r_name) * rel_expr.tf[0, r_id]
        if matsum is None:
          matsum = addend
        else:
          matsum = tf.sparse_add(matsum, addend)
        m_is_sparse = m_is_sparse and (not self.context.is_dense(r_name))

      g = self.context.get_group(rel_expr.type_name)
      output_type = self.context.get_range(g.object_rel)
      scope_qualifier = '' if inverted else '_inverse'
      transpose_m = (inverted == -1)
      output_expr = nql.matmul_any_tensor_dense_tensor(
          matsum, self.tf, a_is_sparse=m_is_sparse, transpose_a=transpose_m)
      return self.context.as_nql(output_expr, output_type)


##############################################################################
# running experiments
##############################################################################


def local_flag_settings(as_dict=False):
  """Get the current tf.flags.FLAGS settings."""
  module_dict = FLAGS.flags_by_module_dict()
  d = dict((x.name, x.value) for x in module_dict[sys.argv[0]])
  if as_dict:
    return d
  else:
    return dict2string(d)


def dict2string(d):
  """Convert a dictionary to a printable string."""
  return ' '.join([('%s=%r' % pair) for pair in sorted(d.items())])


def cell(i, j):
  """String name for a cell in the grid."""
  return 'cell_%d_%d' % (i, j)


def make_grid_context(n):
  """Create a KG encoding a 2-d grid with four+ relations."""

  # set up the context object
  if FLAGS.variant == 'mix':
    tf.logging.info('using MixtureNQExpr')
    context = nql.NeuralQueryContext()
    context.expression_factory_class = MixtureNQExpr
  elif FLAGS.variant == 'sum':
    tf.logging.info('using MatSumNQExpr')
    context = nql.NeuralQueryContext()
    context.expression_factory_class = MatSumNQExpr
    if FLAGS.minibatch_size != 1:
      raise NotImplementedError('sum not implemented for minibatch_size!=1')
  else:
    context = nql.NeuralQueryContext()
    tf.logging.info('using join/std follow operation')

  # declare the basic directions
  context.declare_relation('n', 'place_t', 'place_t')
  context.declare_relation('s', 'place_t', 'place_t')
  context.declare_relation('e', 'place_t', 'place_t')
  context.declare_relation('w', 'place_t', 'place_t')

  # add in any extra relations
  extra_rels = [('r%d' % i) for i in range(FLAGS.num_extra_rels)]
  tf.logging.info('extra_rels %r' % extra_rels[-10:])
  for r in extra_rels:
    context.declare_relation(r, 'place_t', 'place_t')

  # build a list of facts to put in the KB, one fact per line in format
  #    rel <TAB> subj <TAB> obj
  kg_lines = []
  dij = {'n': (-1, 0), 's': (+1, 0), 'e': (0, +1), 'w': (0, -1)}
  t0 = time.time()
  tf.logging.info('creating kg lines...')
  for i in range(0, n):
    for j in range(0, n):
      for direction, (di, dj) in dij.items():
        if extra_rels:
          direction = extra_rels.pop()
        if (0 <= i + di < n) and (0 <= j + dj < n):
          kg_lines.append(
              '\t'.join([direction, cell(i, j),
                         cell(i + di, j + dj)]) + '\n')

  tf.logging.info('created lines in %f sec' % (time.time() - t0))
  tf.logging.info('%d extra rels remain of %d' %
                  (len(extra_rels), FLAGS.num_extra_rels))
  tf.logging.info('loading %d lines' % len(kg_lines))
  for i in range(min(20, len(kg_lines))):
    tf.logging.info('kg_line %02d: %s' % ((i + 1), kg_lines[i].strip()))

  # load the facts into the KB
  t0 = time.time()
  context.load_kg(lines=kg_lines)
  tf.logging.info('loaded in %f sec' % (time.time() - t0))

  # define the KB type for relations
  context.construct_relation_group('dir_g', 'place_t', 'place_t')

  # return the context holding the KB
  return context


def total_size(context):
  """Return the total size of all relations in their initial encoding."""
  tot = 0
  for r in context.get_relation_names():
    try:
      m = context.get_initial_value(r)
      tot += numpy_size(r, m)
    except KeyError:
      tf.logging.info('skipping uninitialized relation %r' % r)
  return tot


def numpy_size(msg, m):
  """Size in bytes of a numpy object.

  Only works for a couple of numpy types.

  Args:
    msg: A string used in error messages.
    m: numpy array or scipy COO or CSR matrix.

  Returns:
    size in bytes
  """
  if isinstance(m, scipy.sparse.coo.coo_matrix):
    # assumes data, i, j all have 32 bits/4 bytes
    size = m.data.shape[0] * 3 * 4
  elif isinstance(m, np.ndarray):
    size = m.shape[0] * m.shape[1] * 4
  elif isinstance(m, scipy.sparse.csr_matrix):
    size += numpy_size(msg, m.indices) + numpy_size(msg, m.values)
  else:
    tf.logging.warn('uncounted object %s of type %r' % (msg, type(m)))
    size = 0
  return size


def show_size(s):
  """String that translates size in bytes to kB, Mb, etc."""
  return '%d %.2f kB %.2f Mb %.2f Gb' % (s, s / 1024., s / (1024. * 1024.), s /
                                         (1024. * 1024. * 1024.))


def expt():
  """Run an experiment."""

  # clean up last expt
  tf.compat.v1.reset_default_graph()

  # make the context object
  tf.logging.info('FLAGS.num_trials %d' % FLAGS.num_trials)
  tf.logging.info('making grid of dim %d with %d entities' %
                  (FLAGS.n, FLAGS.n * FLAGS.n))
  c = make_grid_context(FLAGS.n)
  tf.logging.info('total context size %s' % show_size(total_size(c)))

  # build all the expressions we'll want to evaluate

  # r is a set of relations
  r_ph = c.placeholder('r', 'dir_g')

  # define nql_expr[h] = output of an h-hop inference
  input_ph = c.placeholder('input', 'place_t')
  nql_expr = {0: input_ph}
  for h in range(1, FLAGS.max_hops + 1):
    nql_expr[h] = nql_expr[h - 1].follow(r_ph, FLAGS.dir)

  experiment_names = ['minibatch', 'singleton']

  # do some computations and store timing/size results
  with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    def test_run(msg, tf_expr, fd):
      """Evaluate a tf expression and time that evaluation."""
      t0 = time.time()
      tf_result = session.run(tf_expr, feed_dict=fd)
      elapsed = time.time() - t0
      result_size = numpy_size(msg, tf_result)
      return elapsed, result_size

    # store statistics - detail is trial-by-trial, total is sum over all trials
    size_detail = {}
    time_detail = {}
    atime_detail = {}  # time divided by minibatch size
    size_total = collections.defaultdict(float)
    time_total = collections.defaultdict(float)
    atime_total = collections.defaultdict(float)

    for t in range(FLAGS.num_trials + 1):
      for xname in experiment_names:  # minibatch, singleton starting points
        for n_hops in sorted(nql_expr):

          # construct an input - to keep TF from caching and optimizing
          # we need to have a unique relation and input each time around

          r_shape = (1, c.get_max_id('dir_g'))
          rw = np.ones(
              r_shape,
              dtype=np.float32) + 0.001 * np.random.rand(1, r_shape[1])

          if xname == 'singleton':
            i = random.randrange(FLAGS.n)
            j = random.randrange(FLAGS.n)
            a = c.one_hot_numpy_array(cell(i, j), 'place_t')
            fd = {input_ph.name: a, r_ph.name: rw}
          else:
            cells = []
            for k in range(FLAGS.minibatch_size):
              i = random.randrange(FLAGS.n)
              j = random.randrange(FLAGS.n)
              cells.append(cell(i, j))
            a = np.vstack([c.one_hot_numpy_array(x, 'place_t') for x in cells])
            fd = {input_ph.name: a, r_ph.name: rw}

          # run an inference step

          tm, sz = test_run('%s %d-hop' % (xname, n_hops), nql_expr[n_hops].tf,
                            fd)
          batch_size = 1 if xname == 'singleton' else FLAGS.minibatch_size
          atm = tm / batch_size
          tf.logging.info(
              'trial %d %s hops %d time(sec) %f adjusted %f qps %f size %s' %
              (t, xname, n_hops, tm, atm, (1.0 / atm), show_size(sz)))

          # record the results

          if t > 0:
            # don't record the time for first round
            detail_key = (t, xname, n_hops)
            size_detail[detail_key] = sz
            time_detail[detail_key] = tm
            atime_detail[detail_key] = atm
            total_key = (xname, n_hops)
            size_total[total_key] += sz
            time_total[total_key] += tm
            atime_total[total_key] += atm

    # print a bunch of logging info
    tf.logging.info('local flags %r' % local_flag_settings())

    if FLAGS.details:
      # very details logging info on each trial
      tf.logging.info('\t'.join(
          't sing/mini num_hop time atime qps size'.split()))
      for k in sorted(time_detail):
        sz, tm, atm = size_detail[k], time_detail[k], atime_detail[k]
        qps = 1.0 / atm
        str_k = map(str, k)
        str_stats = ['%.2f' % x for x in [tm, atm, qps]]
        tf.logging.info('\t'.join(str_k + str_stats + [show_size(sz)]))
    tf.logging.info('total context size %s' % show_size(total_size(c)))
    tf.logging.info('\t'.join('sing/mini num_hop time atime qps size'.split()))

    # log averages and extract qps for 2-hops as the summary statistic
    for k in sorted(time_total.keys()):
      nt = FLAGS.num_trials
      sz, tm, atm = size_total[k] / nt, time_total[k] / nt, atime_total[k] / nt
      qps = 1.0 / atm
      str_k = [str(x) for x in k]
      times = [('%.2f' % x) for x in [tm, atm, qps]]
      tf.logging.info('\t'.join(str_k + times + [show_size(sz / nt)]))
      # save avg mininbatch time for a summary table
      (xname, n_hops) = k
      if xname == 'minibatch' and n_hops == FLAGS.max_hops:
        summary_stat = qps

    # return a few selected values in a dictionary
    d = local_flag_settings(as_dict=True)
    d['_mb-%d-hops' % FLAGS.max_hops] = summary_stat
    d['_sz'] = sz
    return d


def save_tsv_table(filename, results):
  """Save dict of expt() outputs in a TSV file."""
  filename = FLAGS.table_tag + filename
  tf.logging.info('writing to %r' % filename)
  with tf.io.gfile.GFile(filename, 'w') as fp:
    cols = None
    for k, d in sorted(results.items()):
      if cols is None:
        d_keys = sorted(d.keys())
        cols = ['key'] + d_keys
        fp.write('\t'.join(cols) + '\n')
      fp.write('\t'.join([repr(k)] + [repr(d[dk]) for dk in d_keys]) + '\n')


def main(unused_args):
  """Run the appropriate experiment."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  tf.logging.info('tf version is %r' % tf.version)
  if FLAGS.vary_n:
    # vary grid size n
    all_ns = map(int, FLAGS.vary_n.split(','))
    tf.logging.info('varying n among %r' % all_ns)
    results = {}
    for i in all_ns:
      FLAGS.n = i
      d = expt()
      results[i] = d
    save_tsv_table('vary_n.tsv', results)
  elif FLAGS.vary_extra_rels:
    # vary number of relations
    all_ns = map(int, FLAGS.vary_extra_rels.split(','))
    tf.logging.info('varying num_extra_rels among %r' % all_ns)
    results = {}
    for i in all_ns:
      FLAGS.num_extra_rels = i
      d = expt()
      results[i] = d
    save_tsv_table('vary_extra_rels.tsv', results)
  else:
    # run one experiment
    d = expt()
    save_tsv_table('vanilla.tsv', {'---': d})


if __name__ == '__main__':
  tf.app.run()
