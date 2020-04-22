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
"""Neural Query Language (NQL) with support on multi GPUs.

This is the implementation of distributed execution of NQL expressions as
described in the Appendix D in the paper ``Scalable Neural Methods for
Reasoning With a Symbolic Knowledge Base''.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import nql
from nql import Error
from nql import RelationDeclaration
import numpy as np
import scipy.sparse
import tensorflow.compat.v1 as tf



class InsufficientGPUError(Error):
  """Not enough GPU available as requested by the user.

  Attributes:
    requested_gpu: The number of GPUs requested by the user.
    available_gpu: The number of GPUs available on the machine.
    message: An explanation of the error
  """

  def __init__(self, requested_gpu, available_gpu, message):
    Error.__init__(self)
    self.requested_gpu = requested_gpu
    self.available_gpu = available_gpu
    self.message = message

  def __str__(self):
    return '%s -- requested: %d available: %d' % (
        self.message, self.requested_gpu, self.available_gpu)


class DistributedNeuralQueryExpression(nql.NeuralQueryExpression):
  """An expression in NQL for use across multiple GPUs.

  Important: Config tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  so that it falls back to CPU if some computation cannot run on GPU. This is
  important since some part of sparse matrix has to be pinned on CPU.

  Attributes:
    context: a pointer back to the DistributedNeuralQueryContext in which this
      expression should be evaluated.
    tf_expr: a Tensorflow expression or list of values which can be converted
      into a Tensorflow expression via tf.constant which evaluates this
      DistributedNeuralQueryContext
    type_name: type of the (Distributed)NeuralQueryExpression
    provenance: a parse tree for the expression, which could be used to compile
      out an expression in some other interpreter, or for debugging the
      underlying structure.
  """

  def __init__(self,
               context,
               tf_expr,
               type_name,
               provenance = None):
    nql.NeuralQueryExpression.__init__(self, context, tf_expr, type_name,
                                       provenance)

  def _follow_named_rel(self, rel_name,
                        inverted):
    """Follow a specific relation in the KG.

    A relation-set following on a reified KB consists of two executions of
    _follow_named_rel(), one in the forward direction (inverse=1) and
    one in the backward direction (inverse=-1), described in Eq. 5 and 6
    respectively. Each execution of the function _follow_named_rel()
    requires to distribute the shards of the reified KB to different GPU
    devices and then reduce the results.

    Args:
      rel_name: a string naming a declared KG relation
      inverted: +1 to follow the relation normally, -1 to use inverses

    Returns:
      A NeuralQueryExpression
    """
    if inverted == +1:
      input_type_name = self.context.get_domain(rel_name)
      output_type_name = self.context.get_range(rel_name)
    else:
      if inverted != -1:
        raise ValueError('Inverted (%d) is neither +1 nor -1' % inverted)
      input_type_name = self.context.get_range(rel_name)
      output_type_name = self.context.get_domain(rel_name)

    m_is_sparse = not self.context.is_dense(rel_name)
    transpose_m = (inverted == -1)
    self._check_type_compatibility(input_type_name, self._type_name, 'follow')

    # we reserve GPU:0 to handle other computations, e.g. embedding lookup,
    # softmax, losses, etc.
    num_shards = self.context.gpus - 1
    sharded_outputs = [None] * num_shards
    # shard_offset is used for tracking where a shard starts, since each shard
    # may be of different sizes.
    shard_offset = 0
    for shard_id in range(num_shards):
      with tf.device('/gpu:%s' % (shard_id + 1)):
        # sharded reified KB are processed on different GPU devices, and
        # stored as a list of tensors.
        m = self.context.get_tf_tensor(rel_name, shard_id)
        x = self.tf if inverted < 0 \
            else self.tf[:, shard_offset: shard_offset + m.shape[1]]
        shard_offset += m.shape[1]
        output_expr = nql.matmul_any_tensor_dense_tensor(
            m, x, a_is_sparse=m_is_sparse, transpose_a=transpose_m)
        sharded_outputs[shard_id] = output_expr

    with tf.device('/gpu:0'):
      # sharded results are then combined when all shards are ready.
      output_expr = tf.concat(sharded_outputs, axis=1) if inverted < 0 \
          else tf.add_n(sharded_outputs)
    provenance = nql.NQExprProvenance(
        operation='follow', args=(rel_name, inverted), inner=self.provenance)
    return self.context.as_nql(output_expr, output_type_name, provenance)


class DistributedNeuralQueryContext(nql.NeuralQueryContext):
  """Context object needed to define DistributedNeuralQueryExpressions."""

  def __init__(self, gpus):
    nql.NeuralQueryContext.__init__(self)
    available_gpus = _get_available_gpus()
    tf.logging.info('Available GPUs for computation: %s', available_gpus)
    if gpus <= 1:
      raise InsufficientGPUError(
          gpus, len(available_gpus),
          'At least 2 GPUs required for DistributedNeuralQueryContext')
    if len(available_gpus) < gpus:
      tf.logging.warning('%d GPUs requested but %d available', gpus,
                         len(available_gpus))
      raise InsufficientGPUError(gpus, len(available_gpus), 'Not enough GPUs')

    self.gpus = gpus
    self.expression_factory_class = DistributedNeuralQueryExpression

  def declare_relation(self,
                       rel_name,
                       domain_type,
                       range_type,
                       trainable = False,
                       dense = False,
                       gpus = 0):
    """Declare the domain and range types for a relation.

    Args:
      rel_name: string naming a relation
      domain_type: string naming the type of subject entities for the relation
      range_type: string naming the type of object entities for the relation
      trainable: boolean, true if the weights for this relation will be trained
      dense: if true, store data as a dense tensor instead of a SparseTensor
      gpus: number of gpus available for computation

    Raises:
      RelationNameError: If a relation with this name already exists.
    """
    super(DistributedNeuralQueryContext,
          self).declare_relation(rel_name, domain_type, range_type, trainable,
                                 dense)
    if gpus <= 0:
      self._declaration[rel_name].underlying_parameters = [None] * gpus

  def get_underlying_parameters(self, rel_name):
    """The list of parameters that will be learned for a trainable relation.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a list of dense 1-D Tensor holding weights for relation triples. The
      return value is None if it doesn't run under multi-gpus. A tensor in the
      list is None if its relation is not trainable.
    None if the relation is not traininable.
    """
    return self._declaration[rel_name].underlying_parameters

  def construct_relation_group(
      self,
      group_name,
      domain_type,
      range_type,
      group_members = None):
    """Declare a group of relations.

    Codes to shard the reified KB is presented in the function
    construct_relation_group().

    Triples are constructed in the same way as construct_relation_group()
    in NeuralQueryContext except that they are stored in multiple sparse
    relation matrices. Relations are also declared normally.

    For example, "rel_g_obj" is the variable for M_obj in Appendix D of
    the paper. "rel_g_obj" is sharded into n sparse matrices
    "rel_g_obj_0", ..., "rel_g_obj_(n-1)", where n is the number of GPUs
    available for computation. The i'th shard "rel_g_obj_i" corresponds
    to M_obj,1 discussed in the paper.

    Args:
      group_name: a string of the to-be-constructed type, which will hold a set
        of relations
      domain_type: a string naming the subject entity type of all the relations
      range_type: a string naming the objects entity type of all the relations
      group_members: the relations to add to the group. By default members of
        group will be all declared relations with that type signature (ie, with
        the specified domain and range).

    Returns:
      a RelationGroup object which summarizes all of these names.

    Raises:
      ValueError: If dense relations are specified as they cannot be handled.
    """
    if not group_members:
      group_members = sorted([
          rel for rel in self.get_relation_names()
          if self.get_domain(rel) == domain_type and
          self.get_range(rel) == range_type
      ])
    if self.is_type(group_name):
      # if this type is already defined the members need to be in the
      # order associated with that type id
      group_members = sorted(
          group_members, key=lambda r: self.get_id(r, group_name))
    else:
      self.extend_type(group_name, group_members)

    for r in group_members:
      if self.is_dense(r):
        raise ValueError('Dense relation %r is unsupported.' % r)

    group = nql.RelationGroup(group_name, group_members)
    self._group[group_name] = group
    # declare the schema for the necessary extension to the KG
    self.declare_relation(
        group.relation_rel, group.triple_type, group_name, gpus=self.gpus)
    self.declare_relation(
        group.subject_rel, group.triple_type, domain_type, gpus=self.gpus)
    self.declare_relation(
        group.object_rel, group.triple_type, range_type, gpus=self.gpus)
    # relation i in this group has num_rows[i] rows
    num_rows = [self._np_initval[r].data.shape[0] for r in group.members]
    total_num_rows = sum(num_rows)
    # names of all those triples
    self.extend_type(
        group.triple_type,
        [group.triple_prefix + str(i) for i in range(total_num_rows)])
    # now populate the sparse matrixes
    triple_indices = np.arange(total_num_rows, dtype='int32')
    rel_indices = np.hstack([
        np.ones(num_rows[i], dtype='int32') * i
        for i in range(len(group.members))
    ])

    subj_indices = np.hstack([self._np_initval[r].col for r in group.members])
    obj_indices = np.hstack([self._np_initval[r].row for r in group.members])
    weight_data = np.hstack([self._np_initval[r].data for r in group.members])
    ones_data = np.ones_like(weight_data)

    # GPU:0 is reserved for all other computation except for KB inference,
    # e.g. embedding lookup, losses, etc. So the total number of shards equals
    # to num_gpus - 1
    num_shards = self.gpus - 1
    shard_size = int(total_num_rows / num_shards) + 1
    for shard_id in range(num_shards):
      start = shard_id * shard_size
      end = min((shard_id + 1) * shard_size, total_num_rows)
      triple_indices = np.arange(end - start, dtype='int32')
      self._np_initval[self._sharded_rel_name(
          group.relation_rel, shard_id)] = scipy.sparse.coo_matrix(
              (weight_data[start:end],
               (rel_indices[start:end], triple_indices)),
              shape=(len(group.members), end - start),
              dtype='float32')
      self._np_initval[self._sharded_rel_name(
          group.subject_rel, shard_id)] = scipy.sparse.coo_matrix(
              (ones_data[start:end], (subj_indices[start:end], triple_indices)),
              shape=(self.get_max_id(domain_type), end - start),
              dtype='float32')
      self._np_initval[self._sharded_rel_name(
          group.object_rel, shard_id)] = scipy.sparse.coo_matrix(
              (ones_data[start:end], (obj_indices[start:end], triple_indices)),
              shape=(self.get_max_id(range_type), end - start),
              dtype='float32')
    return group

  def get_tf_tensor(self, rel_name,
                    shard_id = -1):
    """Get the Tensor that represents a relation.

    Args:
      rel_name: string naming a declared relation
      shard_id: the i'th shard of the matrix. -1 if the matrix is not sharded.

    Returns:
      tf.SparseTensor

    Raises:
      RuntimeError: If the expression has no initial value.
    """
    if shard_id < 0:
      return super(DistributedNeuralQueryContext, self).get_tf_tensor(rel_name)

    sharded_rel_name = self._sharded_rel_name(rel_name, shard_id)
    # construct tensor if it's not been created before, and cache it.
    if sharded_rel_name not in self._cached_tensor:
      if sharded_rel_name not in self._np_initval:
        raise RuntimeError('KG relation named %r has no initial value.' %
                           sharded_rel_name)
      if self.is_dense(rel_name):
        raise TypeError(
            'DistributedNeuralQueryContext does not support dense relation %d' %
            rel_name)

      m = self._np_initval[sharded_rel_name]
      n_rows, n_cols = m.shape
      data_m = np.transpose(np.vstack([m.row, m.col]))
      if self.is_trainable(rel_name):  # construct tf variable if trainable
        data_var_name = 'nql/%s_values' % rel_name
        data_var = tf.Variable(m.data, trainable=True, name=data_var_name)
        sparse_tensor = tf.SparseTensor(data_m, data_var, [n_rows, n_cols])
        assert self._declaration[rel_name].underlying_parameter is None
        self._declaration[rel_name].underlying_parameters[shard_id] = data_var
      else:  # if not trainable, construct a constant tensor
        sparse_tensor = tf.SparseTensor(data_m, m.data, [n_rows, n_cols])

      self._cached_tensor[sharded_rel_name] = sparse_tensor
    return self._cached_tensor[sharded_rel_name]

  def get_initializers(self):
    """Tensorflow initializers for all relation parameters.

    Returns:
      a list of Tensorflow ops.
    """
    result = super(DistributedNeuralQueryContext, self).get_initializers()
    for rel_name in self.get_relation_names():
      ps = self.get_underlying_parameters(rel_name)
      result.extend([p.initializer for p in ps if p is not None])
    return result

  def _sharded_rel_name(self, rel_name, shard_id = -1):
    """Helper function to append shard_id to the end of rel_name.

    Args:
      rel_name: string naming a declared relation
      shard_id: the i'th shard of the matrix.

    Returns:
      A string of rel_name appended by shard_id
    """
    return rel_name + '_' + str(shard_id)


class DistributedRelationDeclaration(RelationDeclaration):
  """Holds information about a relation.

  See docs for declare_relation for explanation.
  """

  def __init__(self, rel_name, domain_type, range_type, trainable, dense):
    super(DistributedRelationDeclaration,
          self).__init__(rel_name, domain_type, range_type, trainable, dense)
    self.underlying_parameters = None


def _get_available_gpus():
  local_devices = tf.config.experimental.list_physical_devices('GPU')
  return [
      device.name for device in local_devices if device.device_type == 'GPU'
  ]
