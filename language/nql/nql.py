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
"""Main implementation of Neural Query Language.

"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from language.nql import nql_io
from language.nql import nql_symbol
import numpy as np
import scipy.sparse
import tensorflow as tf




class Error(Exception):
  """Container class for exceptions in this module."""
  pass


class TypeNameError(Error):
  """The type is undefined.

  Attributes:
    type_name: The type that is not defined.
    message: An explanation of the error
  """

  def __init__(self, type_name, message):
    Error.__init__(self)
    self.type_name = type_name
    self.message = message

  def __str__(self):
    return '%s type_name:%s' % (self.message, self.type_name)


class EntityNameError(Error):
  """The entity is undefined and cannot be mapped onto the unknown id.

  Attributes:
    entity_name: The entity that is not defined.
    type_name: The type that is not defined.
    message: An explanation of the error
  """

  def __init__(self, entity_name, type_name, message):
    Error.__init__(self)
    self.entity_name = entity_name
    self.type_name = type_name
    self.message = message

  def __str__(self):
    return '%s entity_name:%s of type_name:%s' % (
        self.message, self.entity_name, self.type_name)


class RelationNameError(Error):
  """The relation is undefined or invalid.

  Attributes:
    relation_name: The relation that is not defined.
    message: An explanation of the error
  """

  def __init__(self, relation_name, message):
    Error.__init__(self)
    self.relation_name = relation_name
    self.message = message

  def __str__(self):
    return '%s relation_name:%s' % (self.message, self.relation_name)


class TypeCompatibilityError(Error):
  """Two types are not compatible for use in an operation.

  Attributes:
    type_name1: The name of type 1.
    type_name2: The name of type 2.
    operation: The operation.
  """

  def __init__(self, type_name1, type_name2, operation):
    Error.__init__(self)
    self.type_name1 = type_name1
    self.type_name2 = type_name2
    self.operation = operation

  def __str__(self):
    return 'type1:%s and type2:%s are not compatible for operation:%s' % (
        self.type_name1, self.type_name2, self.operation)


class NeuralQueryExpression(object):
  """An expression in NQL.

  Attributes:
    context: a pointer back to the NeuralQueryContext in which this expression
      should be evaluated.
    tf_expr: a Tensorflow expression or list of values which can be converted
      into a Tensorflow expression via tf.constant which evaluates this
      NeuralQueryExpression
    type_name: shortcut for self.tf.name
    provenance: a parse tree for the expression, which could be used to compile
      out an expression in some other interpreter, or for debugging the
      underlying structure.
  """

  def __init__(self,
               context,
               tf_expr,
               type_name,
               provenance = None):
    self.context = context
    self._tf = tf_expr
    self._type_name = type_name
    self.provenance = provenance

  @property
  def tf(self):
    """A Tensorflow expression which evaluates this NeuralQueryExpression.

    Returns:
      A Tensorflow expression that computes this NeuralQueryExpression's value.
    """
    if isinstance(self._tf, tf.Tensor) or isinstance(self._tf, tf.Variable):
      return self._tf
    else:
      return tf.constant(self._tf)

  @property
  def name(self):
    """Name of the underlying TensorFlow expression.

    Returns:
      The Tensorflow expression name computing this.
    """
    return self._tf.name

  @property
  def type_name(self):
    """Declared object types in the set defined by this.

    Returns:
      A string name of the type.
    """
    return self._type_name

  def _check_type_compatibility(self, type_name1, type_name2,
                                operation):
    """Check if two types can be used with an operation.

    Args:
      type_name1: A string name for a type
      type_name2: A string name for a type
      operation: A string decription of the operation.

    Raises:
      TypeCompatibilityError: If incomptaible.
    """
    if type_name1 != type_name2:
      raise TypeCompatibilityError(type_name1, type_name2, operation)

  def follow(self,
             rel_specification,
             inverted = +1):
    """Follow a relation, or relation group, in the knowledge graph.

    Specifically, if x evaluates to a set of knowledge graph entities, then
    x.follow('foo') evaluates to the set of all entities which are related to x
    via the relation 'foo'.

    When rel_specification is a string naming a relation - ie it is a relation
    named in the expression's context - then follow only that relation in the
    knowledge graph. If inverted == -1, then follow the inverse of the relation.

    When rel_specification is an expression that evaluates to a a type
    associated with a relation group, then follow all the relations in that
    group, and return an appropriately weighted mixture of the results. If
    inverted == -1, then do the same for the inverses of the relation.

    Args:
      rel_specification: a string or NeuralQueryExpression for what to follow.
      inverted: +1 to follow the relations normally, -1 to use inverses

    Returns:
      A NeuralQueryExpression from following the relation.

    Raises:
      RelationNameError: If the relation cannot be found.
    """
    if (isinstance(rel_specification, str)
        & self.context.is_relation(rel_specification)
        & (inverted == +1 or inverted == -1)):
      what_to_follow = rel_specification + ('_inverse' if inverted < 0 else '')
      with tf.name_scope('follow_%s' % what_to_follow):
        return self._follow_named_rel(rel_specification, inverted)
    elif isinstance(rel_specification, NeuralQueryExpression):
      return self._follow_relation_set(rel_specification, inverted)
    else:
      raise RelationNameError(
          str(rel_specification), 'Illegal rel_specification.')

  def _follow_relation_set(self, rel_expr,
                           inverted):
    """Follow a relation group.

    Args:
      rel_expr: A NeuralQueryExpression for what to follow.
      inverted: +1 to follow the relations normally, -1 to use inverses  This is
        a macro, which expands to an expression that computes a weighted set of
        objects obtained by following the relations associated with rel_expr, or
        their inverses if inverted == -1.

    Returns:
      A NeuralQueryExpression

    Raises:
      RelationNameError: If the type of the expression is not a relation group.
    """
    if not self.context.is_group(rel_expr.type_name):
      raise RelationNameError(rel_expr.type_name,
                              'Expression type is not a relation group.')
    g = self.context.get_group(rel_expr.type_name)
    if inverted == +1:
      with tf.name_scope('follow_group_%s' % rel_expr.type_name):
        return (self.follow(g.subject_rel, -1).follow(g.weight_rel) * \
                rel_expr.follow(g.relation_rel, -1)).follow(g.object_rel)
    else:
      with tf.name_scope('follow_group_%s_inverse' % rel_expr.type_name):
        return (self.follow(g.object_rel, -1).follow(g.weight_rel) * \
                rel_expr.follow(g.relation_rel, -1)).follow(g.subject_rel)

  def _follow_named_rel(self, rel_name,
                        inverted):
    """Follow a specific relation in the KG.

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
    m = self.context.get_tf_tensor(rel_name)
    m_is_sparse = not self.context.is_dense(rel_name)
    transpose_m = (inverted == -1)

    self._check_type_compatibility(input_type_name, self._type_name, 'follow')
    output_expr = matmul_any_tensor_dense_tensor(
        m, self.tf, a_is_sparse=m_is_sparse, transpose_a=transpose_m)
    provenance = NQExprProvenance(
        operation='follow', args=(rel_name, inverted), inner=self.provenance)
    return self.context.as_nql(output_expr, output_type_name, provenance)

  def __getattr__(self,
                  rel_name):
    """Allow x.rel_name() to be used instead of x.follow('rel_name').

    Args:
      rel_name: a string naming a declared KG relation

    Returns:
      Function that creates the appropriate 'follow' construct.
    """
    return lambda inverted=+1: self.follow(rel_name, inverted=inverted)

  def weighted_by(self, weight_rel_name,
                  category_name):
    """Weight elements in set by confidence in some relationship holding.

    Specifically, weight all elements x in a set by the confidence
    given to the trip weight_rel_name(x,category_name).
    is equivalent to

    x * x.jump_to(category_name,category_type).follow(weight_rel_name,-1)

    where category_type is the appropriate type for, is determined by the
    declaration for weight_rel_name.

    Args:
      weight_rel_name: a string naming a declared KG relation
      category_name: an entity name in the KG

    Returns:
      A NeuralQueryExpression
    """
    self._check_type_compatibility(
        self.context.get_domain(weight_rel_name), self._type_name,
        'weighted_by')
    category_type = self.context.get_range(weight_rel_name)
    with tf.name_scope('weighted_by_%s_%s' % (weight_rel_name, category_name)):
      weight_vector = self.context.one(category_name, category_type).follow(
          weight_rel_name, -1)
      return self.__mul__(weight_vector)

  def filtered_by(self, rel_name,
                  related_entity):
    """Alias for weighted_by which makes more sense for 'hard' relationships.

    Args:
      rel_name: a string naming a declared KG relation
      related_entity: an entity name in the KG

    Returns:
      A NeuralQueryExpression.
    """
    return self.weighted_by(rel_name, related_entity)

  def weighted_by_sum(self, other
                     ):
    """Weight elements in some set by the sum of the scores in some other set.

    Args:
      other: A NeuralQueryExpression

    Returns:
      The NeuralQueryExpression that evaluates to the reweighted version of
    the set obtained by evaluating 'self'.
    """
    provenance = NQExprProvenance(
        operation='weighted_by_sum',
        inner=self.provenance,
        other=other.provenance)
    with tf.name_scope('weighted_by_sum'):
      return self.context.as_nql(
          self.tf * tf.reduce_sum(input_tensor=other.tf, axis=1, keepdims=True),
          self._type_name, provenance)

  def if_any(self, other):
    """Alias for weighted_by_sum which makes more sense for 'hard' sets.

    Args:
      other: A NeuralQueryExpression

    Returns:
      A NeuralQueryExpression.
    """
    return self.weighted_by_sum(other)

  def __add__(self, other):
    """Add the underlying expressions.

    Args:
      other: A NeuralQueryExpression

    Returns:
      A NeuralQueryExpression.
    """
    if isinstance(other, NeuralQueryExpression):
      self._check_type_compatibility(self.type_name, other.type_name, 'add')
      provenance = NQExprProvenance(
          operation='add', inner=self.provenance, other=other.provenance)
      return self.context.as_nql(self.tf + other.tf, self.type_name, provenance)
    else:
      # hopefully a constant
      provenance = NQExprProvenance(
          operation='add',
          inner=self.provenance,
          args=(None, other),
          other=NQExprProvenance(operation='constant'))
      return self.context.as_nql(self.tf + other, self.type_name, provenance)

  def __or__(self, other):
    """Alias for __add__ but more appropriate for 'hard' sets.

    Args:
      other: A NeuralQueryExpression

    Returns:
      A NeuralQueryExpression.
    """
    return self.__add__(other)

  def __mul__(self, other):
    """Component-wise multiply the underlying expressions.


    Args:
      other: A NeuralQueryExpression

    Returns:
      A NeuralQueryExpression.
    """
    if isinstance(other, NeuralQueryExpression):
      self._check_type_compatibility(self.type_name, other.type_name, 'mul')
      provenance = NQExprProvenance(
          operation='add', inner=self.provenance, other=other.provenance)
      return self.context.as_nql(
          tf.multiply(self.tf, other.tf), self.type_name, provenance)
    else:
      provenance = NQExprProvenance(
          operation='mul',
          inner=self.provenance,
          other=NQExprProvenance(operation='constant', args=(None, other)))
      return self.context.as_nql(
          tf.multiply(self.tf, other), self.type_name, provenance)

  def __and__(self, other):
    """Alias for __mul__ which is more appropriate for hard sets.

    Args:
      other: A NeuralQueryExpression

    Returns:
      A NeuralQueryExpression.
    """
    return self.__mul__(other)

  def jump_to(self, entity_name,
              type_name):
    """A singleton set.

    Args:
      entity_name: a string naming an entity
      type_name: the string type name of the named entity

    Returns:
      A NeuralQueryExpression.
    """
    return self.context.one(entity_name, type_name)

  def jump_to_all(self, type_name):
    """A universal set containing all entities of some type.

    Args:
      type_name: the string type name

    Returns:
      A NeuralQueryExpression for an all-ones vector for the types.
    """
    return self.context.all(type_name)

  def jump_to_none(self, type_name):
    """An empty set for some type.


    Args:
      type_name: the string type name

    Returns:
      A NeuralQueryExpression for an all-zeros vector for the types.
    """
    return self.context.none(type_name)

  def tf_op(self, py_fun
           ):
    """Apply any shape-preserving function to the underlying expression."""
    with tf.name_scope('tf_op'):
      return self.context.as_nql(py_fun(self.tf), self._type_name)

  def eval(self,
           session = None,
           as_dicts = True,
           as_top = 0,
           simplify_unitsize_minibatch=True,
           feed_dict = None):
    """Evaluate the Tensorflow expression associated with this NeuralQueryExpression.

    Args:
      session: tf.Session used to evaluate
      as_dicts: if true, convert each row of the minibatch to to a dictionary
        where the keys are entity names, and the values are weights for those
        entities.  Each 'row dictionary' is returned in an array. If as_dicts is
        false, then just return the result, which is typically a numpy array.
      as_top: if positive, return a list of the top k-scoring k-items from the
        as_dicts output.
      simplify_unitsize_minibatch: if true and as_dicts is also true, and the
        minibatch size is 1, then just return the single row dictionary.
      feed_dict: dictionary mapping placeholder names to initial values

    Returns:
      Result of evaluating the underlying NeuralQueryExpression, in a format
      determined by as_dicts, simplify_unitsize_minibatch, and as_top.
    """
    result = self.tf.eval(feed_dict=feed_dict, session=session)
    if as_top > 0:
      return self.context.as_top_k(
          as_top,
          result,
          self.type_name,
          simplify_unitsize_minibatch=simplify_unitsize_minibatch)
    elif as_dicts:
      return self.context.as_dicts(
          result,
          self.type_name,
          simplify_unitsize_minibatch=simplify_unitsize_minibatch)
    else:
      return result


class NeuralQueryContext(object):
  """Context object needed to define NeuralQueryExpressions."""

  # used for serialization of sparse tensors
  VALUE_PART = 0
  INDEX_PART = 1

  def __init__(self):
    # symbol table for every type
    self._symtab = dict()
    # and a special symbol table for relation names
    self._rel_name_symtab = nql_symbol.SymbolTable()
    # map relation name to information like domain, range, ...
    self._declaration = dict()
    # sparse scipy matrix with initial values for each relation
    self._np_initval = dict()
    # cached out tensors for each relation
    self._cached_tensor = dict()
    # factory class used when you convert to an expression
    # using as_nql - you can set this to a subclass
    # of NeuralQueryExpression if you, eg, want to
    # define your own expression methods
    self.expression_factory_class = NeuralQueryExpression
    # map relation group name to information about that group
    self._group = dict()

  # Basic API

  def as_nql(self,
             expr,
             type_name,
             provenance = None
            ):
    """Convert a Tensorflow expression to a NeuralQueryExpression.

    Args:
      expr: a Tensorflow expression which evaluates to a dense matrix of size
        N*M, where M is the minibatch size, and N is the max id for the named
        type.  Alternatively, expr can be a NeuralQueryExpression, which will be
        typechecked and returned unchanged
      type_name: string naming a type
      provenance: if provided, an NQExprProvenance object

    Returns:
      A NeuralQueryExpression

    Raises:
      TypeCompatibilityError: If expression is not of type_name type.
      TypeError: If the factory method returned an unexpected type.
    """
    if isinstance(expr, NeuralQueryExpression):
      if expr.type_name != type_name:
        raise TypeCompatibilityError(expr.type_name, type_name, 'as_nql')
      return expr
    else:
      tf_expr = expr
      result = self.expression_factory_class(
          self, tf_expr, type_name, provenance=provenance)
      if not isinstance(result, NeuralQueryExpression):
        raise TypeError(
            'Factory returned %r instead of a NeuralQueryExpression:' %
            type(result))
      return result

  def as_tf(self,
            expr):
    """Convert a NeuralQueryExpression expression to a Tensorflow expression.

    If called on a Tensorflow expression, it will return expr unchanged.

    Args:
      expr: either a NeuralQueryExpression or a Tensorflow expression.

    Returns:
      a Tensorflow expression
    """
    if isinstance(expr, NeuralQueryExpression):
      return expr.tf
    else:
      return expr

  def as_weighted_triples(self, rel_name,
                          m):
    """Make the value of a SparseTensor for a relation human-readable.

    The human-readable format is a dictionary mapping tuples of strings
    (rel_name,subject_entity,object_entity) to weights.

    Args:
      rel_name: a string naming a relation
      m: a tf.SparseTensorValue object

    Returns:
      a dictionary
    """
    nnz = m.indices.shape[0]

    def as_domain_sym(entity_id):
      return self.get_entity_name(entity_id, self.get_domain(rel_name))

    def as_range_sym(entity_id):
      return self.get_entity_name(entity_id, self.get_range(rel_name))

    def triple(k):
      return (rel_name, as_domain_sym(m.indices[1, k]),
              as_range_sym(m.indices[0, k]))

    keys = [triple(i) for i in range(nnz)]
    vals = [m.values[i] for i in range(nnz)]
    return dict(zip(keys, vals))

  def as_top_k(self,
               k,
               matrix,
               type_name,
               simplify_unitsize_minibatch = True
              ):
    """Make an evaluated NeuralQueryExpression human-readable.

    Args:
      k: integer
      matrix: numpy matrix
      type_name: string name of type associated with matrix
      simplify_unitsize_minibatch: if true and as_dicts is also true, and the
        minibatch size is 1, then just return the single row dictionary.

    Returns:
      similar to as_dicts but returns lists of top-scoring k items
    """

    def top_k(k, d):
      return sorted(d.items(), key=lambda t: -t[1])[:k]

    dict_result = self.as_dicts(
        matrix,
        type_name,
        simplify_unitsize_minibatch=simplify_unitsize_minibatch)
    if isinstance(dict_result, dict):
      return top_k(k, dict_result)
    else:
      return [top_k(k, d) for d in dict_result]

  def as_dicts(self,
               matrix,
               type_name,
               simplify_unitsize_minibatch = True
              ):
    """Make an evaluated NeuralQueryExpression human-readable.

    Args:
      matrix: numpy matrix
      type_name: string name of type associated with matrix
      simplify_unitsize_minibatch: if true and as_dicts is also true, and the
        minibatch size is 1, then just return the single row dictionary.

    Returns:
      dictionary structure or list of dictionaries.
    """
    num_entity_sets = matrix.shape[0]
    row_indices, col_indices = np.nonzero(matrix)
    result = [{} for _ in range(num_entity_sets)]
    for k in range(row_indices.shape[0]):
      set_index = row_indices[k]
      entity_index = col_indices[k]
      weight = matrix[set_index, entity_index]
      entity_name = self.get_entity_name(entity_index, type_name)
      result[set_index][entity_name] = weight
    if simplify_unitsize_minibatch and num_entity_sets == 1:
      return result[0]
    else:
      return result

  def constant(self, value,
               type_name):
    """An NQL encoding of an appropriate Tensorflow constant.

    The constant should be a Tensorflow expression corresponding to a minibatch
    of weighted sets of this type.

    Args:
      value: a Tensorflow expression
      type_name: a string naming a type

    Returns:
      A NeuralQueryExpression.
    """
    provenance = NQExprProvenance(operation='constant', args=(type_name, value))
    return self.as_nql(value, type_name, provenance)

  def placeholder(self, name, type_name):
    """An NQL encoding of a Tensorflow placeholder.

    The placehold will be configured to hold weighted sets of entities of the
    appropriate type.

    Args:
      name: the name to give to the placeholder
      type_name: a string naming a type

    Returns:
      A NeuralQueryExpression.
    """
    provenance = NQExprProvenance(
        operation='placeholder', args=(type_name, name))
    value = tf.placeholder(
        tf.float32, shape=[None, self.get_max_id(type_name)], name=name)
    return self.as_nql(value, type_name, provenance)

  def one(self, entity_name, type_name):
    """An NQL encoding of a one-hot set containing just this entity.

    Args:
      entity_name: a string naming an entity
      type_name: the string type name of the named entity

    Returns:
      A NeuralQueryExpression.
    """
    provenance = NQExprProvenance(
        operation='one', args=(type_name, entity_name))
    return self.as_nql(
        self.one_hot_numpy_array(entity_name, type_name), type_name, provenance)

  def all(self, type_name):
    """A universal set containing all entities of some type.

    Args:
      type_name: the string type name

    Returns:
      A NeuralQueryExpression for an all-ones vector for the types.
    """
    provenance = NQExprProvenance(operation='all', args=(type_name, None))
    np_vec = np.ones((1, self.get_max_id(type_name)), dtype='float32')
    return self.as_nql(np_vec, type_name, provenance)

  def none(self, type_name):
    """An empty set for some type.

    Args:
      type_name: the string type name

    Returns:
      A NeuralQueryExpression for an all-zeros vector for the types.
    """
    provenance = NQExprProvenance(operation='none', args=(type_name, None))
    np_vec = self.zeros_numpy_array(type_name, as_matrix=True)
    return self.as_nql(np_vec, type_name, provenance)

  def zeros_numpy_array(self, type_name,
                        as_matrix = True):
    """A zero valued row vector encoding this entity.

    Args:
      type_name: the string type name of the named entity
      as_matrix: if true, return a 1-by-N matrix instead of a vector.

    Returns:
      A numpy array.  If as_matrix is True, the array has shape (1,n)
      where n is the number of columns, else the shape is (n,).
    """
    if as_matrix:
      return np.zeros((1, self.get_max_id(type_name)), dtype='float32')
    else:
      return np.zeros((self.get_max_id(type_name),), dtype='float32')

  def one_hot_numpy_array(self,
                          entity_name,
                          type_name,
                          as_matrix = True):
    """A one-hot row vector encoding this entity.

    Args:
      entity_name: a string naming an entity.
      type_name: the string type name of the named entity
      as_matrix: if true, return a 1-by-N matrix instead of a vector.

    Returns:
      A numpy array.  If as_matrix is True, the array has shape (1,n)
      where n is the number of columns, else the shape is (n,).

    Raises:
      EntityNameError: if entity_name does not map to a legal id.
    """
    index = self.get_id(entity_name, type_name)
    if index is None:
      raise EntityNameError(
          entity_name=entity_name,
          type_name=type_name,
          message='Cannot make a one-hot vector')
    result = self.zeros_numpy_array(type_name, as_matrix)
    if as_matrix:
      result[0, index] = 1.0
    else:
      result[index] = 1.0
    return result

  # schema operations

  def declare_entity_type(self,
                          type_name,
                          fixed_vocab = None,
                          unknown_marker = '<UNK>',
                          fixed_vocab_size = 0):
    """Declare an entity type.

    Declared entity types can be used as domain and ranges for relations.

    There are three flavors of entity types.  By default, entity types will have
    an open-ended vocabulary - any string can be used as an entity name, and ids
    will be allocated for new strings as they are needed.  One can also fix
    either the vocabulary size or the vocabulary.  Entity names used as domains
    or ranges declare_relation are of this variety.

    If the vocabulary size is fixed to N, there will always be exactly N id's
    allocated for this entity type, which means matrices defining relations
    involving this type will have predictable shapes. (Eg a relation with this
    as the domain_type will have N rows.)  You can add strings to a type with
    fixed vocabulary size, but an error will be raised if you add too many.
    Entity types with fixed vocabulary size are useful for sets of 'object ids',
    like question_ids, which are not used in modeling, but are related to 'real
    objects', eg features of questions: the intended use case is to set up a
    type, load in the objects for a minibatch, and then reset the entity type
    before loading in the next set of objects.

    If the vocabulary is fixed, then the set of entity names is also fixed.
    Entities with names not in the vocabulary will be mapped to a special
    'unknown_entity' id, which has the name given by the unknown_marker argument
    given above.  If the unknown_marker is None then an error is raised when
    entity names outside the fixed vocabulary are inserted.

    Args:
      type_name: string name of this type
      fixed_vocab: an list of strings
      unknown_marker: a string or None
      fixed_vocab_size: a positive integer

    Raises:
      TypeNameError: If the type has already been declared.
      ValueError: If both fixed vocab and fixed vocab size are specified.
    """
    if type_name in self._symtab:
      raise TypeNameError(type_name, 'Type is already declared.')
    symtab = self._symtab[type_name] = nql_symbol.SymbolTable()
    if fixed_vocab:
      if fixed_vocab_size:
        raise ValueError(
            'Fixed vocab and fixed vocab size are mutually exclusive')
      for entity_name in fixed_vocab:
        symtab.insert(entity_name)
      symtab.freeze(unknown_marker=unknown_marker)
    if fixed_vocab_size:
      symtab.pad_to_vocab_size(fixed_vocab_size)
      symtab.reset()

  def clear_entity_type_vocabulary(self, type_name):
    """Clear the vocabulary for an entity type.

    Args:
      type_name: string name of type
    """

    symtab = self._symtab[type_name]
    symtab.reset()

  def declare_relation(self,
                       rel_name,
                       domain_type,
                       range_type,
                       trainable = False,
                       dense = False):
    """Declare the domain and range types for a relation.

    Args:
      rel_name: string naming a relation
      domain_type: string naming the type of subject entities for the relation
      range_type: string naming the type of object entities for the relation
      trainable: boolean, true if the weights for this relation will be trained
      dense: if true, store data as a dense tensor instead of a SparseTensor

    Raises:
      RelationNameError: If a relation with this name already exists.
    """
    if rel_name in self._declaration:
      raise RelationNameError(rel_name, 'Multiple declarations for relation.')
    reserved = dir(NeuralQueryExpression)
    if rel_name in reserved:
      tf.logging.warn(
          'rel_name prohibits expr.%s() as it matches a reserved word in: %r',
          rel_name, reserved)
    self._declaration[rel_name] = RelationDeclaration(rel_name, domain_type,
                                                      range_type, trainable,
                                                      dense)
    for type_name in [domain_type, range_type]:
      if type_name not in self._symtab:
        self._symtab[type_name] = nql_symbol.SymbolTable()
    self._rel_name_symtab.insert(rel_name)

  def construct_relation_group(self,
                               group_name,
                               domain_type,
                               range_type,
                               group_members = None
                              ):
    """Declare a group of relations.

    This must be called after all the triples for all the relations in the group
    have already been defined.

    A group of relations is a special kind of type, which can be aggregated over
    with the follows(group) command.  Examples of group declarations:

      context.construct_relation_group(
        'dir_g', 'place_t', 'place_t')
      context.construct_relation_group(
         'vdir_g', 'place_t', 'place_t', ['n', 's'])

    After a group 'foo' is defined then you can use methods like
    x.follow(context.all('dir_g')) which is equivalent to (x.follow('r1') |
    ... | x.follow('rk')) where r1, ..., rk are the relations in the group 'foo'

    In general x.follow(g) is a weighted version of the above, ie, if g gives
    weights w1...wk to r1...rk, then

      x.follow(g) == w1*x.follow('r1') + ... + wk*x.follow('rk')

    How this is implemented: The group is basically just a type where the
    members are relation names, but creating the group also extends the schema,
    and the sparse matrices that define the knowledge graph.  If there is a
    group called 'foo_g', then the following relations and type will be created
    and populated:

    - foo_g: is the type for the relation names
    - foo_g_triple_t: a type for 'triples' for relation group foo.

     A triple is a conceptually a datastructure with three parts: (rel,
     subj,obj) represented with a triple id and these relations:

    - foo_g_sub, foo_g_obj, foo_g_rel: these relations map foo_g triple ids to
      their subjects, objects, and relations (the relation being of type
      'foo_rel_t', and the subj and obj being of types 'domain_type' and
      'range_type' respectively.

    - Finally, foo_g_weight is a relation mapping every triple id to itself,
      with weight equal to the weight of the original fact the triple was
      derived from.

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

    group = RelationGroup(group_name, group_members)
    self._group[group_name] = group
    # declare the schema for the necessary extension to the KG
    self.declare_relation(group.relation_rel, group.triple_type, group_name)
    self.declare_relation(group.subject_rel, group.triple_type, domain_type)
    self.declare_relation(group.object_rel, group.triple_type, range_type)
    self.declare_relation(group.weight_rel, group.triple_type,
                          group.triple_type)
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
    # weights are in a diagonal matrix
    self._np_initval[group.weight_rel] = scipy.sparse.coo_matrix(
        (weight_data, (triple_indices, triple_indices)),
        shape=(total_num_rows, total_num_rows),
        dtype='float32')
    self._np_initval[group.relation_rel] = scipy.sparse.coo_matrix(
        (weight_data, (rel_indices, triple_indices)),
        shape=(len(group.members), total_num_rows),
        dtype='float32')
    self._np_initval[group.subject_rel] = scipy.sparse.coo_matrix(
        (ones_data, (subj_indices, triple_indices)),
        shape=(self.get_max_id(domain_type), total_num_rows),
        dtype='float32')
    self._np_initval[group.object_rel] = scipy.sparse.coo_matrix(
        (ones_data, (obj_indices, triple_indices)),
        shape=(self.get_max_id(range_type), total_num_rows),
        dtype='float32')
    return group

  def get_relation_names(self):
    """All declared relations.

    Returns:
      an list of string relation names.
    """
    return self._declaration.keys()

  def get_type_names(self):
    """All declared types.

    Returns:
      an list of string type names.
    """
    return self._symtab.keys()

  def get_domain(self, rel_name):
    """The domain for this relation.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a string naming the type of the subject entities for the relation.
    """
    return self._declaration[rel_name].domain_type

  def get_range(self, rel_name):
    """The range for this relation.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a string naming the type of the object entities for the relation.
    """
    return self._declaration[rel_name].range_type

  def is_trainable(self, rel_name):
    """Test if the relation has been declared as 'trainable'.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a boolean
    """
    return self._declaration[rel_name].trainable

  def is_dense(self, rel_name):
    """Test if the relation has been declared as 'dense'.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a boolean
    """
    return self._declaration[rel_name].dense

  def get_underlying_parameter(self, rel_name):
    """The Tensorflow parameter that will be learned for a trainable relation.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a dense 1-D Tensor holding weights for relation triples, or
    None if the relation is not traininable.
    """
    return self._declaration[rel_name].underlying_parameter

  def get_initializers(self):
    """Tensorflow initializers for all relation parameters.

    Returns:
      a list of Tensorflow ops.
    """
    result = []
    for rel_name in self.get_relation_names():
      p = self.get_underlying_parameter(rel_name)
      if p is not None:
        result.append(p.initializer)
    return result

  def is_relation(self, rel_name):
    """Test if the relation has been previously declared.

    Args:
      rel_name: a string naming a declared relation.

    Returns:
      a boolean
    """
    return rel_name in self._declaration

  def is_type(self, type_name):
    """Test if the type has been previously declared.

    Args:
      type_name: a string naming a declared type.

    Returns:
      boolean
    """

    return type_name in self._symtab

  def is_group(self, group_name):
    """Test if the type is declared as a relation group.

    Args:
      group_name: a string possible naming a group

    Returns:
      boolean
    """

    return group_name in self._group

  def get_group(self, group_name):
    """Return information associated with a declared relation group.

    Args:
      group_name: a string naming a group

    Returns:
      The RelationGroup object.
    """

    return self._group[group_name]

  def extend_type(self, type_name, instances):
    """Provide a list of additional instances of a type.

    If the type doesn't previously exist it will be defined.

    Args:
      type_name: string name of a type.
      instances: list of entity names to add to this type.
    """
    if type_name not in self._symtab:
      self._symtab[type_name] = nql_symbol.SymbolTable()
    for entity_name in instances:
      self._symtab[type_name].insert(entity_name)

  def freeze(self, type_name):
    """Freeze the SymbolTable for a specific type.

    Args:
      type_name: string name of a type.
    """
    self._symtab[type_name].freeze()

  def get_shape(self, rel_name):
    """Return the shape of the matrix defining this relation.

    Args:
      rel_name: a string naming a KG relation.

    Returns:
      a tuple of two integers.
    """
    return (self.get_max_id(self.get_domain(rel_name)),
            self.get_max_id(self.get_range(rel_name)))

  def load_schema(self, schema_file):
    """Load relation type declarations from a file.

    Each line in the file tab-separated with fields

      relation_name domain_type range_type

    Args:
      schema_file: a file_like object

    Raises:
      ValueError: If a line is formatted wrong.
    """
    with open(schema_file) as fp:
      for line in nql_io.lines_in(fp):
        parts = line.strip().split('\t')
        if len(parts) != 3:
          raise ValueError('invalid type declaration %r' % line.strip())
        self.declare_relation(parts[0], parts[1], parts[2])

  # Knowledge graph loading and access

  def load_kg(self,
              files = None,
              lines = None,
              freeze = False,
              ignore_undef_rels = False):
    """Load knowledge graph data from a files or lines of strings.

    Each line in the file is tab-separated with fields

      binary_np_initval arg1 arg2 [confidence]

    Confidence defaults to 1.0.

    Calling this method more than once can grow unfrozen symbol tables and cause
    the constructed data matrices to become incompatible sizes.

    Args:
      files: If specified, a file-like or array of file-like objects.
      lines: If specified, a string or an array of strings.
      freeze: Boolean. If true, freeze dictionaries for all types after loading.
      ignore_undef_rels: Boolean. If true, emit a warning instead of terminating
        when an undefined relation is encountered.

    Raises:
      RelationNameError: If a relation found in a line is undefined and
        ignore_undef_rels is False.
    """

    def empty_buffer():
      return dict((rel_name, []) for rel_name in self.get_relation_names())

    domain_buf = empty_buffer()
    range_buf = empty_buffer()
    data_buf = empty_buffer()
    for line in nql_io.lines_in_all(files, lines):
      # format: relation entity1 entity2 [weight]
      parts = line.strip().split('\t')
      rel_name = parts[0]
      try:
        weight = float(parts[3])
      except IndexError:
        weight = 1.0
      except ValueError:
        tf.logging.warn('confidence %r is not numeric: line %r', parts[-1],
                        line.strip())
        continue
      if not 3 <= len(parts) <= 4:
        tf.logging.warn('ignored illegal kg line: %r', line.strip())
        continue
      if rel_name not in self._declaration:
        error = 'ignored relation %s in line: %r', (rel_name, line.strip())
        if ignore_undef_rels:
          tf.logging.warn(error)
          continue
        else:
          raise RelationNameError(rel_name,
                                  'Unknown relation in line: %r' % line.strip())
      rel_domain = self.get_domain(rel_name)
      rel_range = self.get_range(rel_name)
      i = self._get_insert_id(parts[1], rel_domain)
      j = self._get_insert_id(parts[2], rel_range)
      domain_buf[rel_name].append(i)
      range_buf[rel_name].append(j)
      data_buf[rel_name].append(weight)
    if freeze:
      for rel_name in data_buf:
        self.freeze(self.get_domain(rel_name))
        self.freeze(self.get_range(rel_name))
    # clear buffers and create a coo_matrix for each relation
    for rel_name in data_buf:
      num_domain = self.get_max_id(self.get_domain(rel_name))
      num_range = self.get_max_id(self.get_range(rel_name))
      # matrix is the transpose of what you'd expect. it maps range to
      # domain - because we follow relation R from vector x by
      # multiplication (y = R x) because tf only supports sparse-dense
      # matrix-matrix multiplication
      self._np_initval[rel_name] = scipy.sparse.coo_matrix(
          (data_buf[rel_name], (range_buf[rel_name], domain_buf[rel_name])),
          shape=(num_range, num_domain),
          dtype='float32')

  def serialize_trained(self,
                        output_file,
                        session = None):
    """Save the current value of all trainable relations in a file.

    Args:
      output_file: Filename string or FileLike object.
      session: a tf.Session used to find values of trained SparseTensors
    """
    trained_rels = [
        rel_name for rel_name in self.get_relation_names()
        if self.is_trainable(rel_name)
    ]
    sparse_tensors = [self.get_tf_tensor(rel_name) for rel_name in trained_rels]
    if session is None:
      session = tf.get_default_session()
    trained_dict = {
        name: tensor
        for name, tensor in zip(trained_rels, session.run(sparse_tensors))
    }
    nql_io.write_sparse_tensor_dict(output_file, trained_dict)

  def deserialize_trained(self, input_file):
    """Restore the current value of all trainable relations from a file.

    Args:
      input_file: Filename string or FileLike object.
    """
    relation_dict = nql_io.read_sparse_matrix_dict(input_file)
    for name, relation in relation_dict.items():
      self.set_initial_value(name, relation)

  def serialize_dictionaries(self,
                             output_file,
                             restrict_to = None):
    """Saves all symbol tables to a file preserving their ids.

    Args:
      output_file: Filename string or FileLike object.
      restrict_to: If defined, a list of types to restrict to.
    """
    nql_io.write_symbol_table_dict(output_file, self._symtab, restrict_to)

  def deserialize_dictionaries(self,
                               input_file,
                               restrict_to = None
                              ):
    """Restore all symbol tables from a file.

    Existing tables having the same type_name are overwritten.

    Args:
      input_file: Filename string or FileLike object.
      restrict_to: If defined, a list of types to restrict to.
    """
    for type_name, symbol_table in nql_io.read_symbol_table_dict(
        input_file, restrict_to).items():
      self._symtab[type_name] = symbol_table

  def get_initial_value(self, rel_name
                       ):
    """Return value that will be used to initialize a relation matrix.

    Args:
      rel_name: string name of relation

    Returns:
       A numpy matrix or scipy sparse matrix.
    """
    return self._np_initval[rel_name].transpose()

  def set_initial_value(self, rel_name, m):
    """Provide value that will be used to initialize a relation matrix.

    Args:
      rel_name: string name of relation
      m: a matrix that can used as argument to scipy.coo_matrix, for a sparse
        relation, or any matrix, for a dense relation

    Raises:
      RelationNameError: If the relation cannot be found.
      ValueError: If the relation and initial_value have different shapes.
    """
    if not self.is_relation(rel_name):
      raise RelationNameError(rel_name, 'Relation is not defined.')
    expected_shape = self.get_shape(rel_name)
    if m.shape[1] != expected_shape[1]:
      raise ValueError(
          'relation and initial_value have different columns: %d vs %d' %
          (expected_shape[1], m.shape[1]))
    if self.is_dense(rel_name):
      self._np_initval[rel_name] = m.transpose()
    else:
      self._np_initval[rel_name] = scipy.sparse.coo_matrix(m.transpose())

  def get_tf_tensor(self, rel_name):
    """Get the Tensor that represents a relation.

    Args:
      rel_name: string naming a declared relation

    Returns:
      tf.SparseTensor

    Raises:
      RuntimeError: If the expression has no initial value.
    """
    if rel_name not in self._cached_tensor:
      if rel_name not in self._np_initval:
        raise RuntimeError('KG relation named %r has no initial value.' %
                           rel_name)
      m = self._np_initval[rel_name]
      n_rows, n_cols = m.shape
      if self.is_dense(rel_name):
        if self.is_trainable(rel_name):
          self._cached_tensor[rel_name] = tf.Variable(
              m, trainable=True, name='nlq/' + rel_name)
        else:
          self._cached_tensor[rel_name] = tf.constant(m)
      else:
        data_m = np.transpose(np.vstack([m.row, m.col]))
        if not self.is_trainable(rel_name):
          sparse_tensor = tf.SparseTensor(data_m, m.data, [n_rows, n_cols])
        else:
          data_var_name = 'nql/%s_values' % rel_name
          data_var = tf.Variable(m.data, trainable=True, name=data_var_name)
          sparse_tensor = tf.SparseTensor(data_m, data_var, [n_rows, n_cols])
          self._declaration[rel_name].underlying_parameter = data_var
        self._cached_tensor[rel_name] = sparse_tensor
    return self._cached_tensor[rel_name]

  def _get_insert_id(self, entity_name, type_name):
    """Retrieve the index of entity with given name and type.

    Insert the entity in the appropriate symbol table if it has not yet been
    added.

    Args:
      entity_name: string name of the entity
      type_name: string name of declared type

    Returns:
      integer id for the entity
    """
    return self._symtab[type_name].get_insert_id(entity_name)

  def get_id(self, entity_name, type_name):
    """Retrieve the index of entity with given name and type.

    Args:
      entity_name: string name of the entity
      type_name: string name of declared type

    Returns:
      integer id for the entity

    Raises:
      TypeNameError: If the type_name cannot be found.
      EntityNameError: If the entity_name has no valid id.
    """
    if not self.is_type(type_name):
      raise TypeNameError(type_name, 'Undeclared type')
    try:
      return self._symtab[type_name].get_id(entity_name)
    except KeyError:
      raise EntityNameError(entity_name, type_name, 'No entity mapping')

  def get_entity_name(self, entity_id, type_name):
    """Retrieve the string name the entity with given index and type.

    Args:
      entity_id: integer id for entity
      type_name: string name of declared type

    Returns:
      string name for entity`
    """
    return self._symtab[type_name].get_symbol(entity_id)

  def get_max_id(self, type_name):
    """Retrieve the maximum index of any entity with given type.

    Args:
      type_name: string name of a declared type

    Returns:
      Maximum index of any entity with the given type.
    """
    return self._symtab[type_name].get_max_id()

  def get_unk_id(self, type_name):
    """Return the unknown symbol id with given name, or None if nonexistent."""
    if not self.is_type(type_name):
      raise TypeNameError(type_name, 'Type is not defined.')
    return self._symtab[type_name].get_unk_id()

  def query_kg(self, rel_name, entity_name,
               as_object = False):
    """Simple method to query the KG, mainly for debugging.

    Finds things related to the named entity_name by the named relation.  When
    as_object is True then entity_name should be a second argument/object for
    the triple, rather than the first argument, similarly to follow with
    inverted == -1.

    Args:
      rel_name: string name for KG relation
      entity_name: string name for KG entity
      as_object: if True use inverse of relation.

    Yields:
      Tuples of entity string names and their weights.
    """
    m = self._np_initval[rel_name].transpose()
    query_type_finder_fun = self.get_range if as_object else self.get_domain
    answer_type_finder_fun = self.get_domain if as_object else self.get_range
    array_to_match = m.col if as_object else m.row
    array_to_extract_from = m.row if as_object else m.col
    answer_type = answer_type_finder_fun(rel_name)
    i = self.get_id(entity_name, query_type_finder_fun(rel_name))
    for k in range(m.nnz):
      if array_to_match[k] == i:
        entity_name = self.get_entity_name(array_to_extract_from[k],
                                           answer_type)
        yield entity_name, m.data[k]


# Helper classes and utilities


def _check_type(name, value, expected_type):
  if not isinstance(value, expected_type):
    raise ValueError('%s should be a %r but is %r:%r' %
                     (name, expected_type, type(value), value))


def matmul_any_tensor_dense_tensor(a,
                                   b,
                                   a_is_sparse = True,
                                   transpose_a = False):
  """Like tf.matmul except that first argument might be a SparseTensor.

  Args:
    a: SparseTensor or Tensor
    b: Tensor
    a_is_sparse: true if a is a SparseTensor
    transpose_a: transpose a before performing matmal operation.

  Returns:
    TF expression for a.dot(b) or a.transpose().dot(b)

  Raises
    ValueError: If a or b are of the wrong type.
  """
  _check_type('b', b, tf.Tensor)
  if a_is_sparse:
    _check_type('a', a, tf.SparseTensor)
    a1 = tf.sparse.transpose(a) if transpose_a else a
    return tf.transpose(a=tf.sparse.sparse_dense_matmul(a1, tf.transpose(a=b)))
  else:
    _check_type('a', a, tf.Tensor)
    return tf.transpose(
        a=tf.matmul(a, tf.transpose(a=b), transpose_a=transpose_a))


def nonneg_softmax(expr,
                   replace_nonpositives = -10):
  """A softmax operator that is appropriate for NQL outputs.

  NeuralQueryExpressions often evaluate to sparse vectors of small, nonnegative
  values. Softmax for those is dominated by zeros, so this is a fix.  This also
  fixes the problem that minibatches for NQL are one example per column, not one
  example per row.

  Args:
    expr: a Tensorflow expression for some predicted values.
    replace_nonpositives: will replace zeros with this value before computing
      softmax.

  Returns:
    Tensorflow expression for softmax.
  """
  if replace_nonpositives != 0.0:
    ones = tf.ones(tf.shape(input=expr), tf.float32)
    expr = tf.where(expr > 0.0, expr, ones * replace_nonpositives)
  return tf.nn.softmax(expr)


def nonneg_crossentropy(expr, target):
  """A cross entropy operator that is appropriate for NQL outputs.

  Query expressions often evaluate to sparse vectors.  This evaluates cross
  entropy safely.

  Args:
    expr: a Tensorflow expression for some predicted values.
    target: a Tensorflow expression for target values.

  Returns:
    Tensorflow expression for cross entropy.
  """
  expr_replacing_0_with_1 = \
     tf.where(expr > 0, expr, tf.ones(tf.shape(input=expr), tf.float32))
  cross_entropies = tf.reduce_sum(
      input_tensor=-target * tf.math.log(expr_replacing_0_with_1), axis=1)
  return tf.reduce_mean(input_tensor=cross_entropies, axis=0)


class RelationDeclaration(object):
  """Holds information about a relation.

  See docs for declare_relation for explanation.
  """

  def __init__(self, rel_name, domain_type, range_type, trainable, dense):
    self.rel_name = rel_name
    self.domain_type = domain_type
    self.range_type = range_type
    self.trainable = trainable
    self.dense = dense
    self.underlying_parameter = None

  def __str__(self):
    return 'RelationDeclaration(%r)' % self.__dict__


class RelationGroup(object):
  """Holds information about a relation group.

  See docs for construct_relation_group for explanation.
  """

  def __init__(self, group_name, group_members):
    self.name = group_name
    self.triple_type = '%s_triple_t' % group_name
    self.subject_rel = '%s_subj' % group_name
    self.object_rel = '%s_obj' % group_name
    self.relation_rel = '%s_rel' % group_name
    self.weight_rel = '%s_weight' % group_name
    self.triple_prefix = '%s_trip_' % group_name
    self.members = group_members

  def __str__(self):
    return 'RelationGroup(%r)' % self.__dict__


class NQExprProvenance(object):
  """Information on how a NeuralQueryExpression was constructed.

  Args:
    operation: A text description of the operation.
    inner: The nested operation.
    other: A secondary operation.
    args: Other arguments provided.
  """

  def __init__(self,
               operation = None,
               inner = None,
               other = None,
               args = None):
    self.operation = operation
    self.inner = inner
    self.other = other
    self.args = args

  def __str__(self):
    return 'NQExprProvenance(%r)' % self.__dict__

  def pprintable(self):
    """Convert to a nested dictionary structure.

    Returns:
     a structure that the pprint module can render nicely.
    """

    def as_pprintable(v):
      return v.pprintable() if isinstance(v, NQExprProvenance) else v

    return dict([(k, as_pprintable(v)) for k, v in self.__dict__.items() if v])
