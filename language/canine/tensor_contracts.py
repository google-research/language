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
"""A decorator-based function-level contracting library for TensorFlow.

Expressing preconditions in this way solves an important issue in TensorFlow:
Many preconditions can't be checked at graph construction time (when our
Python code is running) -- it must be checked at graph execution time (later).
Because of this, tf.debugging asserts must be forcefully embedded in the graph
with `tf.control_dependencies`, which is both uninuitive (it can be forgotten)
and is a bit of an eyesore, requiring awkardly placed `with` blocks. Decorators
solve both of these problems.
"""

import contextlib
import inspect
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Sequence, Text, Tuple, Union

import tensorflow.compat.v1 as tf

# Constant for the return value. Can be used in Ensure, but not Require.
RESULT = "RESULT"


_dynamic_asserts_enabled = True
_predicates_enabled = True
_all_disabled = False
_summarize_num_elements = 256
_add_function_variable_scopes = False


def enable_dynamic_asserts(enable: bool = True):
  """Turn on or off using dynamic (graph execution time) checks.

  Enabled by default, but can be disabled to ensure we aren't doing any
  extraneous compute during graph evaluation time.

  Args:
    enable: Whether or not to enable dynamic asserts.
  """
  global _dynamic_asserts_enabled
  _dynamic_asserts_enabled = enable


def disable_dynamic_asserts():
  """Convenience method for `enable_dynamic_asserts(False)`."""
  enable_dynamic_asserts(False)


def enable_predicates(enable: bool = True):
  """Turn on or off whether predicates (`RequireTrue`/`EnsureTrue`) are used.

  Enabled by default, but can be disabled to ensure we aren't doing any
  extraneous compute during graph evaluation time. Lambdas can also sometimes
  cause problems when Autograph is trying to convert trace functions within
  a `tf.data.map`.

  Args:
    enable: Whether or not to enable predicates.
  """
  global _predicates_enabled
  _predicates_enabled = enable


def disable_predicates():
  """Convenience method for `enable_predicates(False)`."""
  enable_predicates(False)


def disable_all_checks():
  """Disables all checks (both static and dynamic) by TensorContracts."""
  global _all_disabled
  _all_disabled = True


def enable_add_function_variable_scopes(enable: bool = True):
  """Add tf.variable_scope with the function name in each `tc.contract`?

  This is disabled by default because it will *CHANGE THE NAMES OF YOUR
  VARIABLES* when used in modeling code (i.e. things in `model_fn` that allocate
  `tf.Variable`s. However, this can be extremely useful for getting helpful
  error messages with meaningful node names when working with non-modeling code
  such as pre-processing code in `input_fn`s.

  Args:
    enable: Whether to enable injection of function variable scopes.
  """
  global _add_function_variable_scopes
  _add_function_variable_scopes = enable


def set_summarize_num_elements(n: int):
  """Sets number of tensor elements to print in `Assert`s."""
  global _summarize_num_elements
  _summarize_num_elements = n


class _Context(object):
  """Contains information that we might want to print out in error messages."""

  def __init__(self, func_name: Text, caller_name: Text, caller_file: Text,
               caller_line: int):
    self.func_name = func_name
    self.caller_name = caller_name
    self.caller_file = caller_file
    self.caller_line = caller_line


class Condition(object):
  """Marker class for 'the set of things passable to `tf.contract` decorator."""

  pass


class CheckableCondition(Condition):
  """A condition that implements `check`."""

  def check(self, context: _Context, args_dict: Dict[Text, Any],
            named_dims: Dict[Text, Union[int, tf.Tensor]],
            dynamic_asserts: bool) -> Iterable[tf.Operation]:
    # Unused:
    del context
    del args_dict
    del named_dims
    del dynamic_asserts
    raise NotImplementedError()


# TODO(jhclark): Allow implicit named dimensions based on `shape` and
# `bounding_shape`.
class NamedDim(Condition):
  """Specifies a name of a dimension that can be used in `shape` checks."""

  def __init__(self,
               dim_name: Text,
               tensor: Optional[Text] = None,
               dim: Optional[int] = None,
               tuple_index: Optional[int] = None,
               value_of: Optional[Text] = None,
               var: Optional[tf.Tensor] = None,
               var_name: Optional[Text] = None):
    """Creates a `NamedDim`.

    Caller must specify either (`tensor` and `dim`) OR `value_of`.

    Args:
      dim_name: The name of the dimension, which will be usable elsewhere.
      tensor: The name of the function argument (a tensor) that defines the
        dimension.
      dim: The index of the desired tensor dimension.
      tuple_index: Index of the desired tensor inside `tensor` iff `tensor` is
        actually a tuple return type.
      value_of: The name of a function argument whose type is an `int` or
        scalar `tf.Tensor`; this value will define this dimension.
      var: For use in `tc.Dynamic` conditions: The actual variable instance to
        use as a `tensor`. (In normal conditions, `tensor` is a string that
        names the function argument to use).
      var_name: For use when `var` is specified: The name of the variable being
        passed to `var`, which will be used to provide meaningful error
        messages.
    """
    self.dim_name = dim_name
    self.tensor = tensor
    self.dim = dim
    self.tuple_index = tuple_index
    self.var = var
    self.var_name = var_name
    self.value_of = value_of

    # PyType doesn't always catch this, which can lead to very confusing errors
    # given that this is the first argument.
    if not isinstance(dim_name, str):
      raise ValueError("Expected `dim_name` to be a string but got: {}".format(
          type(dim_name)))

    if var is not None:
      if not var_name:
        raise ValueError(
            "Expected `var_name` to be specified when `var` is specified.")
      if tensor is not None:
        raise ValueError(
            "Expected `tensor` to be `None` because `var` is specified.")
      if dim is None:
        raise ValueError(
            "Expected `dim` to be specified because `var` is specified.")
      return

    valid_tensor_and_dim = (tensor and dim is not None)
    valid_value_of = value_of is not None
    specified_both = (tensor or dim is not None) and value_of is not None
    if (not valid_tensor_and_dim and not valid_value_of) or specified_both:
      raise ValueError(
          "Either (`tensor` and `dim`) or (`value_of`) must be "
          "specified when defining `dim_name`='{}' in `NamedDim`.".format(
              dim_name))


class Unchecked(object):
  """Represents a dimension within a `shape` that is not checked.

  e.g. `shape=["batch_size", tc.Unchecked('seq_length')]`
  """

  def __init__(self, name: Optional[Text] = None):
    pass


class Require(CheckableCondition):
  """Specifies preconditions within a `tc.contract` decorator."""

  def __init__(
      self,
      tensor: Optional[Text] = None,
      tuple_index: Optional[int] = None,
      rank: Optional[int] = None,
      shape: Optional[Sequence[Union[int, Text, Unchecked]]] = None,
      static_dims: Optional[Sequence[int]] = None,
      bounding_shape: Optional[Sequence[Union[int, Text, Unchecked]]] = None,
      shape_of: Optional[Text] = None,
      dtype: Optional[Union[tf.dtypes.DType, Sequence[tf.dtypes.DType]]] = None,
      dtype_of: Optional[Text] = None,
      is_tensor: bool = True,
      optional: bool = False,
      vanilla_tensor: bool = False,
      ragged: bool = False,
      row_splits_dtype: Optional[tf.dtypes.DType] = None,
      var: Optional[tf.Tensor] = None,
      var_name: Optional[Text] = None):
    """Creates `Require` contract spec for use within a `tc.contract` decorator.

    These same options apply to `Ensure`, but `tensor` must be specified as
    `tc.RESULT` for `Ensure`.

    Args:
      tensor: Name of the function argument (in the decorated function) to be
        checked. Required unless using `var`.
      tuple_index: Index within the tuple (or list-like structure). This is
        typically used in `Ensure` conditions when multiple tensors are being
        returned.
      rank: Require that the rank of the tensor is equal to this value. Ranks
        are currently expected to be statically determinable (unlike shapes).
      shape: Require that the shape of this tensor is equal to this value.
      static_dims: Require that the dimensions indices in this list are static
        (can be determined at graph-building time), not dynamic.
      bounding_shape: For `tf.RaggedTensor`s: require that their bounding shape
        is equal to this value.
      shape_of: Checks that the shape of the tensor is exactly the same as the
        specified argument. This is especially useful for comparing
        `RaggedTensor`s such that we can compare the entire shape (including
        row lengths).
      dtype: Require that the dtype of the tensor is this type.
      dtype_of: Require that the dtype of the tensor is equal to the value of
        the function argument with this name. The function argument itself must
        be either a dtype or tensor-like.
      is_tensor: Require that `tensor` is truly tensor-like (convertible to
        tensor). This excludes structures such as tuples.
      optional: Is `None` a valid value for this tensor?
      vanilla_tensor: Require that `tensor` is a plain `tf.Tensor`. Not a list,
        numpy array, `tf.RaggedTensor`, nor `tf.SparseTensor`.
      ragged: Allow (but not require) that `tensor` can be a `tf.RaggedTensor`.
      row_splits_dtype: Require that `tensor` (which should be a
        `tf.RaggedTensor`) has the specified dtype for its row splits dtype.
      var: For use in `Dynamic` conditions only: The actual argument instance
        to place conditions on.
      var_name: The name of the variable `var`, used to produce meaningful
        error messages.
    """
    self.tensor = tensor
    self.var = var
    self.var_name = var_name
    self.tuple_index = tuple_index
    self.rank = rank
    self.shape = shape
    self.static_dims = static_dims or []
    self.bounding_shape = bounding_shape
    self.shape_of = shape_of
    self.dtype = dtype
    self.dtype_of = dtype_of
    self.is_tensor = is_tensor
    self.optional = optional
    self.vanilla_tensor = vanilla_tensor
    self.ragged = ragged
    self.row_splits_dtype = row_splits_dtype

    if var is None:
      if tensor is None:
        raise ValueError(
            "Expected `tensor` to be specified (unless `var` is specified "
            "inside a `Dynamic` contract).")
    else:
      # `var` is specified.
      if var_name is None:
        raise ValueError(
            "Expected `var_name` to be specified when `var` is specified.")
      if tensor is not None:
        raise ValueError("Exactly one of `tensor` and `var` may be specified.")
      if tuple_index is not None:
        raise ValueError("`tuple_index` may not be specified with `var`.")

    if rank is None:
      # TODO(jhclark): Add implicit rank check back once we support dynamic
      # ranks (right now we assume rank is statically determinable).
      if shape is not None:
        # self.rank = len(shape)
        pass
      elif bounding_shape is not None:
        # self.rank = len(bounding_shape)
        pass
    # TODO(jhclark): Validate that shape is consistent with rank right here
    # (i.e. fail fast).

  def check(self, context: _Context, args_dict: Dict[Text, Any],
            named_dims: Dict[Text, Union[int, tf.Tensor]],
            dynamic_asserts: bool) -> Iterable[tf.Operation]:

    assert_ops = []

    if self.var is not None:
      tensor = self.var
      pretty_tensor_name = "Dynamic:var:{}".format(self.var_name)
    else:
      tensor = _get_tensor_arg(
          args_dict, self.tensor, tuple_index=self.tuple_index, context=context)
      if self.optional:
        if tensor is None:
          return []
      if self.tuple_index is None:
        pretty_tensor_name = self.tensor
      else:
        pretty_tensor_name = "{}[{}]".format(self.tensor, self.tuple_index)

    if self.vanilla_tensor:
      if (not isinstance(tensor, tf.Tensor)) or (
          isinstance(tensor, (tf.RaggedTensor, tf.SparseTensor))):
        raise ValueError(
            ("`{}`: Expected tensor arg `{}` to be strictly a `vanilla_tensor` "
             "(`tf.Tensor`), but got type `{}` with value: {}".format(
                 context.func_name, pretty_tensor_name, type(tensor), tensor)))
    elif self.ragged:
      if not _is_ragged_tensor(tensor):
        raise ValueError(
            ("`{}`: Expected tensor arg `{}` to be (ragged) tensor-like, but "
             "got type `{}` with value: {}".format(context.func_name,
                                                   pretty_tensor_name,
                                                   type(tensor), tensor)))
    elif self.is_tensor:
      if not _is_tensor(tensor):
        raise ValueError(
            ("`{}`: Expected tensor arg `{}` to be tensor-like, but "
             "got type `{}` with value: {}".format(context.func_name,
                                                   pretty_tensor_name,
                                                   type(tensor), tensor)))

    tensor = _to_tensor(tensor)
    if self.dtype is not None:
      if tensor.dtype != self.dtype:
        raise ValueError(
            "`{}`: Expected tensor arg `{}` to have dtype `{}`, but got `{}`"
            .format(context.func_name, pretty_tensor_name, self.dtype,
                    tensor.dtype))
    if self.dtype_of is not None:
      target = _get_arg(args_dict, self.dtype_of, context=context)
      if isinstance(target, tf.dtypes.DType):
        expected_dtype = target
      elif isinstance(target, tf.Tensor):
        expected_dtype = target.dtype
      elif isinstance(target, tf.RaggedTensor):
        expected_dtype = target.dtype
      else:
        raise ValueError(
            "`{}`: For tensor `{}`, expected target of `dtype_of` (`{}`) "
            "to be `tf.Tensor` or `tf.dtypes.DType`, but got `{}`".format(
                context.func_name, pretty_tensor_name, self.dtype_of,
                type(target)))
      if tensor.dtype != expected_dtype:
        raise ValueError(
            "`{}`: Expected tensor arg `{}` to have dtype of `` (`{}`), but "
            "got `{}`".format(context.func_name, pretty_tensor_name,
                              self.dtype_of, target.dtype))
    if self.row_splits_dtype is not None:
      if not self.ragged:
        raise ValueError(
            "`row_splits_dtype` is specified so `ragged` must be set to `True`."
        )
      if tensor.row_splits.dtype != self.row_splits_dtype:
        raise ValueError(
            "`{}`: Expected tensor arg `{}` to have row_splits.dtype `{}`,"
            "but got `{}`".format(context.func_name, pretty_tensor_name,
                                  self.row_splits_dtype,
                                  tensor.row_splits.dtype))
    if self.rank is not None:
      # We assume we're dealing with a statically known rank.
      # TODO(jhclark): Support multiple possible expected ranks.
      actual_rank = tensor.shape.ndims
      expected_rank = self.rank
      if actual_rank is None:
        raise ValueError(
            "Checking contract for `{}`: Unable to get rank for tensor "
            "arg `{}`. Got shape: {} ({})".format(
                context.func_name, pretty_tensor_name, tensor.shape, tensor))
      if actual_rank != expected_rank:
        raise ValueError(
            "Checking contract for `{}`: Expected tensor arg `{}` to have "
            "rank {} but found {}".format(context.func_name, pretty_tensor_name,
                                          expected_rank, actual_rank))
    if self.bounding_shape is not None:
      actual_bounding_shape = _bounding_shape(tensor)
      # For RaggedTensors, we compare their row lengths to get an accurate
      # shape comparison.
      expected_bounding_shape = _resolve_named_dims(
          shape=self.bounding_shape,
          named_dims=named_dims,
          tensor_name=self.tensor,
          func_name=context.func_name)
      has_unchecked_dim = any(
          isinstance(dim, Unchecked) for dim in expected_bounding_shape)
      if has_unchecked_dim:
        if dynamic_asserts:
          # We need to go dimension by dimension.
          for dim_index, expected_dim in enumerate(expected_bounding_shape):
            if isinstance(expected_dim, Unchecked):
              continue
            actual_dim = actual_bounding_shape[dim_index]
            assert_ops.append(
                tf.debugging.Assert(
                    condition=tf.equal(
                        tf.cast(expected_dim, tf.int32),
                        tf.cast(actual_dim, tf.int32)),
                    data=[
                        "Checking contract for `{}` (Called at {}:{}:{}): "
                        .format(context.func_name, context.caller_name,
                                context.caller_file, context.caller_line),
                        "Expected tensor arg `{}[{}]`".format(
                            pretty_tensor_name, dim_index),
                        "to have `bounding_shape[{}]`".format(dim_index),
                        expected_dim, "but found", actual_dim
                    ],
                    summarize=_summarize_num_elements))
      else:
        # We can check all dimensions at once.
        if dynamic_asserts:
          assert_ops.append(
              tf.debugging.Assert(
                  condition=tf.reduce_all(
                      tf.equal(
                          tf.cast(expected_bounding_shape, tf.int64),
                          tf.cast(actual_bounding_shape, tf.int64))),
                  data=[
                      "Checking contract for `{}` (Called at {}:{}:{}): "
                      .format(context.func_name, context.caller_name,
                              context.caller_file, context.caller_line),
                      "Expected tensor arg `{}`".format(pretty_tensor_name),
                      "to have `bounding_shape`", expected_bounding_shape,
                      "but found", actual_bounding_shape
                  ],
                  summarize=_summarize_num_elements))

    if self.shape is not None:
      if not self.is_tensor or isinstance(tensor, tf.RaggedTensor):
        raise ValueError(
            "`shape` specified in preconditions for function `{}` on "
            "tensor `{}` when is_tensor={} ragged={}. "
            "Shape checking is only supported for non-ragged tensors. "
            "For ragged tensors, consider using `bounding_shape`.".format(
                context.func_name, self.tensor, self.is_tensor, self.ragged))
      expected_dims = _resolve_named_dims(
          shape=self.shape,
          named_dims=named_dims,
          tensor_name=pretty_tensor_name,
          func_name=context.func_name)
      actual_dims = _get_tensor_shape(tensor, context=context)
      _check_static_shape(
          expected_dims=expected_dims,
          actual_dims=actual_dims,
          tensor_name=pretty_tensor_name,
          func_name=context.func_name)
      for i, (expected_dim, actual_dim) in (
          enumerate(zip(expected_dims, actual_dims))):
        # TODO(jhclark): Remove support for `None` in favor of `Unchecked` only.
        if expected_dim is None or isinstance(expected_dim, Unchecked):
          pass
        elif _is_dynamic(expected_dim) or _is_dynamic(actual_dim):
          if dynamic_asserts:
            assert_ops.append(
                tf.debugging.Assert(
                    condition=tf.equal(
                        tf.cast(expected_dim, tf.int32),
                        tf.cast(actual_dim, tf.int32)),
                    data=[
                        "Checking contract for `{}` (Called by {}:{}:{}):"
                        .format(context.func_name, context.caller_file,
                                context.caller_line, context.caller_name),
                        "Expected tensor arg `{}`".format(pretty_tensor_name),
                        "to have dimension {} of shape as".format(i),
                        expected_dim, "but found", actual_dim,
                        "Expected shape: {}".format(expected_dims),
                        "Actual shape: ", tf.shape(tensor)
                    ],
                    summarize=_summarize_num_elements))
        else:
          if expected_dim != actual_dim:
            raise ValueError("Expected shape {} does not match actual shape {} "
                             "when checking contract for tensor `{}` in "
                             "function `{}`".format(
                                 expected_dims, actual_dims,
                                 pretty_tensor_name, context.func_name))
    if self.static_dims:
      actual_dims = _get_tensor_shape(tensor, context=context)
      for dim_index in self.static_dims:
        if dim_index >= len(actual_dims):
          raise ValueError(
              f"Checking contract for `{context.func_name}`: "
              "Expected `static_dim` index {dim_index} for tensor "
              f"`{pretty_tensor_name}` but only found {len(actual_dims)} "
              "dimensions.")
        actual_dim = actual_dims[dim_index]
        if not isinstance(actual_dim, int):
          raise ValueError(
              f"Checking contract for `{context.func_name}`: "
              "Expected `static_dim` to hold in tensor "
              f"`{pretty_tensor_name}` for dimension "
              f"`{dim_index}`, but found type `{type(actual_dim)}`")

    if self.shape_of is not None:
      if isinstance(tensor, tf.RaggedTensor):
        # Compare row lengths, etc.
        other = _get_tensor_arg(
            args_dict,
            self.shape_of,
            tuple_index=None,
            context=context)
        if not isinstance(other, tf.RaggedTensor):
          raise ValueError("Constructing contract for `{}`: "
                           "Expected `shape_of` tensor (`{}`) to be Ragged "
                           "when `tensor` (`{}`) is Ragged.".format(
                               context.func_name, self.shape_of,
                               pretty_tensor_name))
        tensor_rows = tensor.row_lengths()
        other_rows = other.row_lengths()
        all_equal = tf.reduce_all(tf.equal(tensor_rows, other_rows))
        assert_ops.append(
            tf.debugging.Assert(
                condition=all_equal,
                data=[
                    "Checking contract for `{}` (Called by {}:{}:{}):".format(
                        context.func_name, context.caller_file,
                        context.caller_line, context.caller_name),
                    "Expected ragged tensor arg `{}`".format(
                        pretty_tensor_name), "to have ragged row lengths ",
                    other_rows, "but found", tensor_rows
                ],
                summarize=_summarize_num_elements))
      else:
        raise ValueError("TODO(jhclark): Implement normal tensors.")

    return assert_ops


class Ensure(Require):
  """Specifies a postcondition of a tensor.

  The `tensor` should generally be `tc.RESULT`. (Yes, you still need to pass it
  to keep the code relatively readable by those new to the library and because
  it makes our implementation much simpler.)

  When returning multiple tensors, `tuple_index` is your friend.

  See `Require` for definition of constructor.
  """
  pass


class Dynamic(Condition):
  """Specifies a function that generates `Condition`s in a `tc.contract`."""

  def __init__(self, condition_generator: Callable[..., Sequence[Condition]],
               *args):
    """Constructs a `tc.Dynamic` condition.

    These are for advanced usage only and typically you should use simpler
    constructs such as `Require` or `RequireTrue` if at all possible. `Dynamic`
    condition generators may be necessary when dealing with complex function
    inputs such as Sequences that have very specific relationships that must
    be satisfied by the function's contract.

    Args:
      condition_generator: A function that generates a list of `Condition`s.
        This function will be passed the arguments extracted from the function
        as specified by `args`. This must *not* be a generator function since
        AutoGraph (used by `tf.data` pipelines) does not support them.
      *args: The string names of arguments that should be extracted from the
        function that this `tc.contract` is decorating. These will be passed to
        the `condition_generator` function. `len(args)` must equal the number
        of arguments defined by the `condition_generator` function.
    """
    self.condition_generator = condition_generator
    self.args = args
    # TODO(jhclark): Check return type of `condition_generator` to make sure
    # it's not a generator.

  def generate(self, context: _Context,
               args_dict: Dict[Text, Any]) -> Iterable[Condition]:
    selected_args = []
    for requested_arg in self.args:
      arg_value = _get_arg(
          args_dict=args_dict,
          tensor_name=requested_arg,
          context=context)
      selected_args.append(arg_value)
    return self.condition_generator(*selected_args)


def _printable_tensor(t):
  """Formats tensor (potentially ragged) so that it can be printed in Assert."""
  if isinstance(t, tf.RaggedTensor):
    # Convert to a string so that it's obvious that padding positions aren't
    # just regular values.
    return tf.strings.as_string(t).to_tensor()
  else:
    return t


class RequireTrue(CheckableCondition):
  """A precondition check that performs custom tf ops on requested tensors."""

  def __init__(
      self,
      check_func: Callable[..., Union[tf.Tensor, bool]],
      tensors: Sequence[Union[Text, Tuple[Text, tf.Tensor]]],
      error: Text,
      error_tensors: Optional[Sequence[
          Union[Text, Tuple[Text, tf.Tensor]]]] = None,
      tensor_format: Optional[Callable[..., Sequence[tf.Tensor]]] = None,
      error_tensor_format: Optional[Callable[..., Sequence[tf.Tensor]]] = None):
    """Creates a `RequireTrue` condition.

    This is used for checking arbitrary conditions on tensors or groups of
    tensors not supported by `Require`, including invariants that inspect the
    contents of tensors at runtime. `RequireTrue` conditions returning a
    `tf.Tensor` will always be executed at runtime; conditions returning a
    python `bool` will be checked at graph building time.

    Args:
      check_func: A function that takes in the arguments specified by `tensors`
        and returns either (a) a boolean tensor, to be checked at graph
        execution time or (b) a python `bool`, to be checked immediately, at
        graph build time. For tensors, `tf.reduce_all` will be called on the
        result of `check_func`, so it's fine if the result of `check_func` is
        not rank 0. Implementors of `check_func` may wish to do additional
        type checking (e.g. asserts) to check the type of arguments since
        pytype will generally not be able to trace through the typing at this
        level. IMPORTANT: This should *NOT* be a lambda function as these are
        not supported by AutoGraph (used to construct `tf.data` pipelines).
        One exception is that lambdas seem to work fine if used inside a
        `local_invariant`.
      tensors: Typically, a sequence of strings, which names arguments to
        extract from the function that this `tc.contract` is decorating. When
        used in a `tc.Dynamic` condition, `tensors` may contain 2-tuples with
        (1) the name of the variable being passed, so that we can provide
        meaningful error messages and (2) the tensor instance to be inspected.
      error: An error message to show if this condition fails at runtime.
      error_tensors: Names of additional tensors to display if this check fails,
        to aid in debugging the failure.
      tensor_format: Function taking same number of tensors as in `tensors` and
        returning the same number of tensors, potentially formatted in some way
        to make interpreting the error message easier (e.g. `row_lengths` for
        ragged tensors).
      error_tensor_format: Like `tensor_format`, but parallel with
        `error_tensors`.
    """
    self.check_func = check_func
    self.error = error
    self.tensors = tensors
    self.error_tensors = error_tensors
    self.tensor_format = tensor_format
    self.error_tensor_format = error_tensor_format

    if not callable(check_func):
      raise ValueError("`check_func` must be a callable function.")

  def _get_my_args(self, tensors: Sequence[Union[Text, Tuple[Text, tf.Tensor]]],
                   args_dict: Dict[Text, Any],
                   context: _Context) -> Tuple[Sequence[Text], Sequence[Any]]:
    """Helper function for check`, which extracts function args."""
    requested_tensor_names = []
    tensor_args = []
    for tensor_item in tensors:
      if isinstance(tensor_item, tuple):
        # Use the name and instance provided by the user from a `Dynamic`
        # contract.
        given_name, tensor_instance = tensor_item
        if not isinstance(given_name, str):
          raise ValueError(
              "Constructing contract for {}: Expected first item of each tuple "
              "in `tensors` to be a string, but got: {}".format(
                  context.func_name, type(given_name)))
        requested_tensor_names.append(given_name)
        tensor_args.append(tensor_instance)
      else:
        # Look up the requested tensor in the function args.
        requested_tensor_name = tensor_item
        maybe_tensor = _get_arg(
            args_dict=args_dict,
            tensor_name=requested_tensor_name,
            context=context)
        requested_tensor_names.append(requested_tensor_name)
        tensor_args.append(maybe_tensor)
    return requested_tensor_names, tensor_args

  def check(self, context: _Context, args_dict: Dict[Text, Any],
            named_dims: Dict[Text, Union[int, tf.Tensor]],
            dynamic_asserts: bool) -> Iterable[tf.Operation]:

    # TODO(jhclark): Check type of `check_func` to make sure it's not a lambda,
    # but *only if* this is a decorator; lambdas seem fine in the case of
    # `local_invariants`.

    if not _predicates_enabled:
      return []

    # We require the user to enumerate the tensors they'll actually use to
    # (a) make the function syntax reasonably simple and (b) provide a helpful
    # error message about what's actually being checked.
    tensor_names, tensor_args = self._get_my_args(self.tensors, args_dict,
                                                  context)

    # Error data that will be printed for both static and dynamic checks:
    error_data = [
        "Checking contract for `{}` (Called at {}:{}:{}): ".format(
            context.func_name, context.caller_name, context.caller_file,
            context.caller_line),
        "Checking tensor args `({})`: {}".format(
            ",".join(tensor_names), self.error),
    ]

    # In case the check itself fails (as opposed to returning a boolean tensor
    # result as expected), enclose it in variable scope to make the failure
    # easier to find.
    with tf.variable_scope("RequireTrue_" + context.func_name):
      # `raw_check_result`: is either (a) a bool tensor of any rank or (b)
      # a python bool.
      raw_check_result = self.check_func(*tensor_args)
      if isinstance(raw_check_result, bool):
        # We can statically check this now (at graph building time).
        # *** Notice early return here for static checks. ***
        if raw_check_result:
          # Condition was satisfied.
          return []
        else:
          error_message = " ".join(error_data)
          raise ValueError("Static check failed: " + error_message)

      if not tf.is_tensor(raw_check_result):
        raise ValueError(
            "Error building contract for `{}`: Expected `check_func` to "
            "return a `tf.Tensor`, but found type `{}`".format(
                context.func_name, type(raw_check_result)))
      if raw_check_result.dtype != tf.bool:
        raise ValueError(
            "Error building contract for `{}`: Expected `check_func` to "
            "return a tensor of dtype `tf.bool`, but found `{}`".format(
                context.func_name, raw_check_result.dtype))

      # Force it into a tensor of rank 0 in case it isn't. This saves lots
      # of boilerplate `reduce_all` in calling code and doesn't hurt anything
      # if we already have a rank 0 boolean.
      check_result = tf.reduce_all(raw_check_result)

    if not dynamic_asserts:
      return []

    # Append the tensors being analyzed by the function so that the error
    # message is reasonably helpful.
    if self.tensor_format:
      formatted_tensors = self.tensor_format(*tensor_args)
    else:
      formatted_tensors = tensor_args
    for tensor_name, tensor in zip(tensor_names, formatted_tensors):
      error_data.append(tensor_name)
      error_data.append("value:")
      error_data.append(_printable_tensor(tensor))

    if self.error_tensors:
      # User specified certain tensors (transformations of the input tensors)
      # that they'd like to see when we error out.
      error_tensor_names, error_tensor_args = self._get_my_args(
          self.error_tensors, args_dict, context)
      if self.error_tensor_format:
        formatted_error_tensors = self.error_tensor_format(*error_tensor_args)
      else:
        formatted_error_tensors = error_tensor_args
      for tensor_name, instance in zip(error_tensor_names,
                                       formatted_error_tensors):
        error_data.append(tensor_name + "(formatted)")
        error_data.append(_printable_tensor(instance))

    error_data.append("Raw result of check_func:")
    error_data.append(_printable_tensor(raw_check_result))

    assert_op = tf.debugging.Assert(
        condition=check_result,
        data=error_data,
        summarize=_summarize_num_elements)
    return [assert_op]


class EnsureTrue(RequireTrue):
  """Checks postconditions for an arbitrary function.

  See `RequireTrue` for constructor arguments.

  While `EnsureTrue` does not yet support complex return types such as tuples,
  you can still consider using a `tc.local_invariant` just before function
  return to enforce such postconditions.
  """
  pass


def _check_static_shape(expected_dims: Sequence[Union[int, tf.Tensor]],
                        actual_dims: Sequence[Union[int, tf.Tensor]],
                        tensor_name: Text, func_name: Text):
  """Checks the shape of a tensor argument, if it's possible statically."""
  for expected_dim, actual_dim in zip(expected_dims, actual_dims):
    if _is_dynamic(expected_dim) or _is_dynamic(actual_dim):
      continue
    if expected_dim != actual_dim:
      raise ValueError("Expected shape {} does not match actual shape {} "
                       "when checking contract for tensor `{}` in "
                       "function `{}`".format(
                           expected_dims, actual_dims, tensor_name,
                           func_name))


def _is_dynamic(dim: Union[int, tf.Tensor]) -> bool:
  """Returns true for dynamic dimensions (a tensor value, not an int)."""
  return not isinstance(dim, int)


def _resolve_named_dims(shape: Sequence[Union[int, Text]],
                        named_dims: Dict[Text, Union[tf.Tensor, int]],
                        tensor_name: Text,
                        func_name: Text) -> Sequence[Union[int, tf.Tensor]]:
  """Resolves the name of any named dimensions and returns the scalar value."""
  numeric_shape: List[Union[int, tf.Tensor]] = list()
  for dim in shape:
    if isinstance(dim, str):
      numeric_dim = named_dims.get(dim, None)
      if numeric_dim is None:
        raise ValueError(
            "Named dimension `{}` not defined as `NamedDim` in "
            "contract when expanding checks for tensor `{}` "
            "in function `{}`. Have NamedDims: {}"
            .format(dim, tensor_name, func_name, " ".join(named_dims.keys())))

      numeric_shape.append(numeric_dim)
    else:
      numeric_shape.append(dim)
  return numeric_shape


def _get_arg(args_dict: Dict[Text, Any], tensor_name: Text,
             context: _Context) -> Any:
  """Gets the value of the specified function argument."""
  if "." not in tensor_name:
    if tensor_name not in args_dict:
      raise ValueError(
          "Require: Arg not found in function `{}`: {} (Have args: {})"
          .format(context.func_name, tensor_name, " ".join(args_dict)))
    return args_dict[tensor_name]

  # Allow tensor names to be attributes of args (e.g. items in named tuples).
  # The simple case of no dots in the name is handled above.
  # TODO(jhclark): Document this functionality. (including "RESULT.starts")
  name_chain = tensor_name.split(".")
  result = args_dict.get(name_chain[0], None)
  if result is None:
    raise ValueError("Require: Arg `{}` not found in function `{}`: {} "
                     "(Have args: {})".format(name_chain[0], context.func_name,
                                              tensor_name, " ".join(args_dict)))
  for name_part in name_chain[1:]:
    try:
      result = getattr(result, name_part)
    except AttributeError as e:
      raise AttributeError(
          "Require: Attribute `{}` not found on object `{}` in function `{}` "
          "(resolving `{}`): {}".format(name_part, type(result),
                                        context.func_name, tensor_name, str(e)))
  return result


def _get_tensor_arg(args_dict: Dict[Text, Any], tensor_name: Text,
                    tuple_index: Optional[int], context: _Context) -> Any:
  """Like `_get_arg`, but also resolves `tuple_index`."""
  maybe_tensor = _get_arg(
      args_dict=args_dict, tensor_name=tensor_name, context=context)
  if tuple_index is not None:
    # TODO(jhclark): More informative error message when `tuple_index` is out of
    # bounds.
    maybe_tensor = maybe_tensor[tuple_index]

  # TODO(jhclark): Move checks on tensor-ness here?
  return maybe_tensor


def _bounding_shape(tensor: Union[tf.Tensor, tf.RaggedTensor]) -> tf.Tensor:
  """Returns the bounding shape of a Tensor, which is possibly Ragged."""
  if isinstance(tensor, tf.RaggedTensor):
    # TODO(jhclark): How do we handle dynamic vs static dimensions here?
    return tensor.bounding_shape(out_type=tf.int32)
  else:
    return tf.shape(tensor)


def _getfullargspec(func):
  return inspect.getfullargspec(func)


# TODO(jhclark): Write tests with function with default args, etc.
def _get_func_args(func, func_args, func_kwargs) -> Dict[Text, Any]:
  """Gets the names and values of the arguments of `func`."""
  argspec = _getfullargspec(func)
  defaults = argspec.defaults or []
  default_args = argspec.args[-len(defaults):]

  # First, populate with defaults, then override with caller args.
  result: Dict[Text, Any] = dict()
  result.update(zip(default_args, defaults))
  result.update(zip(argspec.args, func_args))
  result.update(func_kwargs)
  return result


def _to_tensor(t):
  """Converts value to a tensor if possible; otherwise, returns the original."""
  try:
    return tf.convert_to_tensor(t)
  except TypeError:
    return t


def _is_tensor(t) -> bool:
  """Returns true if the value is a tensor or can be converted to a tensor."""
  if tf.is_tensor(t):
    return True
  try:
    tf.convert_to_tensor(t)
    return True
  except:  # pylint: disable=bare-except
    # Usually a `TypeError`, but we can occasionally get other strange
    # conversion errors such as non-descript `ValueError`s. Any failure here
    # is definitely due to not being converible to a tensor.
    return False


def _is_ragged_tensor(t) -> bool:
  """Returns true if the value is a RaggedTensor *or* a normal tensor."""
  if isinstance(t, tf.RaggedTensor):
    return True
  if _is_tensor(t):
    return True
  return False


def _get_tensor_shape(tensor: tf.Tensor,
                      context: _Context) -> Sequence[Union[int, tf.Tensor]]:
  """Gets actual shape of a Tensor: `int`s for static; `Tensor` for dynamic."""

  # This code is derived from BERT's `common_utils.get_shape_list`, but does
  # not necessarily assert.
  try:
    shape = tensor.shape.as_list()
  except ValueError:
    # `as_list()` is undefined for fully unknown TensorShapes.
    # TODO(jhclark): Support fully unknown shapes. (~15 minutes of work)
    raise ValueError(
        "Analyzing function `{}`: tensor_contracts does not yet support fully "
        "unknown shapes. Tensor: {}".format(context.func_name, tensor.name))

  non_static_indexes = []
  for index, dim in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def _is_scalar(tensor: tf.Tensor) -> bool:
  """Returns true for tensors that are a scalar."""
  return tensor.shape.ndims == 0


def _get_tensor_dim(tensor: tf.Tensor, dim: int,
                    context: _Context) -> Union[int, tf.Tensor]:
  """Gets  specified tensor dimension, backing off to a bound if ragged."""

  # TODO(jhclark): Add tutorial section on how this backoff works.
  if isinstance(tensor, tf.RaggedTensor):
    bounding_shape = tensor.bounding_shape()
    # We back off to the bounding box for finding named dimensions in
    # `RaggedTensor`s.
    return bounding_shape[dim]

  if tensor.shape.num_elements() is None:
    # Shape is fully unknown.
    dyn_shape = tf.shape(tensor)
    return dyn_shape[dim]

  shape = tensor.shape.as_list()
  if dim >= len(shape):
    raise ValueError(
        "Building contract for {}: Dimension index {} is out of range for "
        "shape ({}) in tensor {}".format(context.func_name, dim, shape,
                                         tensor.name))
  if shape[dim] is not None:
    return shape[dim]
  else:
    dyn_shape = tf.shape(tensor)
    return dyn_shape[dim]


def _get_caller_locals(stack) -> Dict[Text, Any]:
  """Gets the local variables defined by the calling function."""
  return stack[1][0].f_locals


def _get_function_and_line(stack):
  """Gets the function and line number of the caller according to `stack`."""
  caller_name = stack[1].function
  caller_file = stack[1].filename
  caller_line = stack[1].lineno
  return caller_name, caller_file, caller_line


def _get_named_dims(
    conditions: Sequence[Condition], args_dict: Dict[Text, Any],
    context: _Context) -> Dict[Text, Union[int, tf.Tensor]]:
  """Resolves `NamedDim`s to actual dimension (static or dynamic)."""

  named_dims: Dict[Text, Union[int, tf.Tensor]] = dict()
  for condition in conditions:
    if not isinstance(condition, NamedDim):
      continue

    # TODO(jhclark): Check bounds for `dim` and give good error message.
    if condition.tensor == RESULT:
      raise ValueError(
          "Analyzing function `{}`: `NamedDim` does not allow `{}` "
          "to be a parameter name. Please acquire the dimension from an "
          "input instead.".format(context.func_name, RESULT))
    if condition.dim is not None:
      if condition.var is not None:
        var_name = condition.var_name
        tensor = condition.var  # Get `var` from `Dynamic` contract.
      else:
        var_name = condition.tensor
        tensor = _get_tensor_arg(
            args_dict,
            condition.tensor,
            tuple_index=condition.tuple_index,
            context=context)
      tensor = _to_tensor(tensor)
      if not _is_tensor(tensor):
        raise ValueError(
            "`{}`: Error while attempting to get `NamedDim` for `{}`: {}"
            "Not a tensor: ".format(context.func_name, var_name, type(tensor)))
      dim: Union[int, tf.Tensor] = _get_tensor_dim(
          tensor, dim=condition.dim, context=context)
    else:
      assert condition.value_of is not None
      dim = _get_arg(args_dict, tensor_name=condition.value_of, context=context)
      if not isinstance(dim, tf.Tensor) and not isinstance(
          dim, int) and not _is_scalar(dim):
        raise ValueError("Expected target of `NamedDim`'s `value_of`='{}'"
                         "to be a scalar, but got {}".format(
                             condition.value_of, dim.get_shape()))
    named_dims[condition.dim_name] = dim
  return named_dims


def _check_preconditions(preconditions: Sequence[Union[Require, RequireTrue]],
                         args_dict: Dict[Text, Any],
                         named_dims: Dict[Text, Union[int, tf.Tensor]],
                         context: _Context) -> Sequence[tf.Operation]:
  """Check preconditions, returning any ops needed for dynamic checks."""
  precondition_ops = []
  for precondition in preconditions:
    if isinstance(precondition, Require):
      if precondition.tensor == RESULT:
        raise ValueError(
            "`Require` cannot use `RESULT`. Use `Ensure` instead.")
    ops = precondition.check(
        context,
        args_dict,
        named_dims,
        dynamic_asserts=_dynamic_asserts_enabled)
    if _dynamic_asserts_enabled:
      precondition_ops.extend(ops)
  return precondition_ops


def _check_postconditions(postconditions: Sequence[Union[Ensure, EnsureTrue]],
                          args_dict: Dict[Text, Any],
                          named_dims: Dict[Text, Union[int, tf.Tensor]],
                          context: _Context,
                          result: Any) -> Any:
  """Check postconditions, mutating `result` to include dynamic checks."""
  for postcondition in postconditions:
    if isinstance(postcondition, Ensure):
      if postcondition.tensor.startswith(RESULT + "."):
        # We need to do the work of writing to structured objects with
        # properties.
        raise ValueError(
            "Postconditions with RESULT.attributes are not yet supported.")
      if postcondition.tensor != RESULT:
        raise ValueError(
            "`Ensure` must use `RESULT` (or some property of result). "
            "Use `Require` instead.")
    # *Only* when checking postconditions, add `RESULT` to `args_dict`.
    args_dict[RESULT] = result
    postcondition_ops = postcondition.check(
        context,
        args_dict,
        named_dims,
        dynamic_asserts=_dynamic_asserts_enabled)

    tuple_index = None
    if isinstance(postcondition, Ensure):
      tuple_index = postcondition.tuple_index

    # If there's a tuple index for this postcondition, we must attach this
    # postcondition to exactly that tensor.
    target_tensor = result
    if tuple_index is not None:
      target_tensor = _get_tensor_arg(
          args_dict,
          postcondition.tensor,
          tuple_index=postcondition.tuple_index,
          context=context)
    # For TF1 / non-eager mode, we have to generate a dummy op to attach
    # the op to.
    if _dynamic_asserts_enabled:
      with tf.control_dependencies(postcondition_ops):
        target_tensor = tf.identity(target_tensor)

      # Now that we've potentially modified a result or tuple result, we
      # need to overwrite the original result.
      if tuple_index is None:
        result = target_tensor
      else:
        result = _tuple_assign(result, postcondition.tuple_index, target_tensor)
  return result


def _expand_dynamic_conditions(conditions: Sequence[Condition],
                               args_dict: Dict[Text, Any],
                               context: _Context) -> List[Condition]:
  result = []
  for condition in conditions:
    if isinstance(condition, Dynamic):
      result.extend(condition.generate(context=context, args_dict=args_dict))
    else:
      result.append(condition)
  return result


def contract(*conditions: Condition):
  """Decorator that specifies the tensor contract for a function.

  This is the primary API for tensor contracts.

  See README.md for example usage.

  Args:
    *conditions: A varargs of `Require` and `Ensure` conditions.

  Returns:
    Behind the scenes, returns the result of the decorated function, potentially
    with a result modified to dynamically do contract checking, if necessary.
  """

  def decorator(func):
    """Invoked by Python when `@contract` is used as a function decorator."""

    def check_preconditions_and_run_func(
        *args: Any, **kwargs: Dict[Text, Any]) -> Any:
      """Invoked by Python when `@contract` is used as a function decorator."""
      # If contracts are disable, just call the function normally without any
      # intervention.
      if _all_disabled:
        return func(*args, **kwargs)

      caller_name, caller_file, caller_line = _get_function_and_line(
          inspect.stack())
      pretty_func_name = "{}.{}".format(func.__module__, func.__name__)
      context = _Context(
          func_name=pretty_func_name,
          caller_name=caller_name,
          caller_file=caller_file,
          caller_line=caller_line)

      args_dict: Dict[Text, Any] = _get_func_args(func, args, kwargs)
      if RESULT in args_dict:
        raise ValueError(
            "Analyzing function `{}` at {}:{}: tf_contracts does not allow "
            "`{}` to be a parameter name.".format(pretty_func_name, caller_file,
                                                  caller_line, RESULT))

      # Generate any extra `Dynamic` conditions.
      expanded_conditions = _expand_dynamic_conditions(
          conditions=conditions, args_dict=args_dict, context=context)
      # IMPORTANT: Use only `expanded_conditions` hereafter. Unfortunately,
      # `conditions` apparently can't be `del`ed due to *args and/or capturing.

      named_dims = _get_named_dims(
          conditions=expanded_conditions, args_dict=args_dict, context=context)

      precondition_specs: List[Union[Require, RequireTrue]] = list()
      postcondition_specs: List[Union[Ensure, EnsureTrue]] = list()
      for condition in expanded_conditions:
        if isinstance(condition, (NamedDim, Dynamic)):
          pass
        # `Ensure` / `EnsureTrue` must be handled first since they're subclasses
        # of `Require` / `RequireTrue`.
        elif isinstance(condition, (Ensure, EnsureTrue)):
          postcondition_specs.append(condition)
        elif isinstance(condition, (Require, RequireTrue)):
          precondition_specs.append(condition)
        else:
          raise ValueError("Unrecognized `Condition` subclass: {}".format(
              type(condition)))

      precondition_ops = _check_preconditions(
          precondition_specs,
          args_dict=args_dict,
          named_dims=named_dims,
          context=context)
      if _add_function_variable_scopes:
        with tf.variable_scope(func.__name__):
          with tf.control_dependencies(precondition_ops):
            result = func(*args, **kwargs)
      else:
        with tf.control_dependencies(precondition_ops):
          result = func(*args, **kwargs)

      result = _check_postconditions(
          postcondition_specs,
          args_dict=args_dict,
          named_dims=named_dims,
          context=context,
          result=result)
      return result
    return check_preconditions_and_run_func
  return decorator


@contextlib.contextmanager
def _noop_context_manager():
  """A `ContextManager` that does nothing."""
  yield None


def local_invariant(*conditions: Condition) -> ContextManager[None]:
  """A variant of a contract that can be specified inline via a `with` block."""
  if _all_disabled:
    return _noop_context_manager()

  caller_name, caller_file, caller_line = _get_function_and_line(
      inspect.stack())
  pretty_func_name = "{}.local_invariant.{}".format(caller_name, caller_line)
  context = _Context(
      func_name=pretty_func_name,
      caller_name=caller_name,
      caller_file=caller_file,
      caller_line=caller_line)

  # We'll treat the local variables defined within the calling context
  # as the 'args' here.
  args_dict: Dict[Text, Any] = _get_caller_locals(inspect.stack())
  if RESULT in args_dict:
    raise ValueError(
        "Analyzing function `{}` at {}:{}: tf_contracts does not allow "
        "`{}` to be a local variable.".format(pretty_func_name, caller_file,
                                              caller_line, RESULT))

  named_dims = _get_named_dims(
      conditions=conditions, args_dict=args_dict, context=context)

  precondition_specs: List[Union[Require, RequireTrue]] = list()
  for condition in conditions:
    if isinstance(condition, NamedDim):
      pass
    # `Ensure` / `EnsureTrue` must be handled first since they're subclasses
    # of `Require` / `RequireTrue`.
    elif isinstance(condition, (Ensure, EnsureTrue)):
      raise ValueError(
          "Postconditions not allowed in `local_invariant`: {}".format(
              type(condition)))
    elif isinstance(condition, (Require, RequireTrue)):
      precondition_specs.append(condition)
    else:
      raise ValueError(
          "`Condition` subclass not supported in `local_invariant`: {}".format(
              type(condition)))

  precondition_ops = _check_preconditions(
      precondition_specs,
      args_dict=args_dict,
      named_dims=named_dims,
      context=context)

  # This will be used in the resulting `with` block.
  return tf.control_dependencies(precondition_ops)


def _is_named_tuple(x):
  """Check if we're dealing with a NamedTuple."""
  t = type(x)
  b = t.__bases__
  if len(b) != 1 or b[0] != tuple:
    return False
  f = getattr(t, "_fields", None)
  if not isinstance(f, tuple):
    return False
  return all(isinstance(n, str) for n in f)


def _tuple_assign(target: Tuple[Any], index: int, item: Any) -> Tuple[Any]:
  """Assign a result into an (immutable) tuple via copy-on-write."""

  # TODO(jhclark): Support `tuple_item` with string names for named tuples.
  result = list(target)
  result[index] = item
  if _is_named_tuple(target):
    # For NamedTuples, we need to still return another NamedTuple.
    result = type(target)._make(result)  # pytype: disable=attribute-error
  elif isinstance(target, list):
    pass
  else:
    result = tuple(result)
  # Make sure we didn't mutilate the type somehow.
  assert isinstance(result, type(target))
  return result


def func_name_scope():
  """Decorator that adds the current function name as a variable scope.

  WARNING: EXPERIMENTAL (may conflict with other decorators).

  This makes things like `input_fn`s easier to debug. Be aware that this will
  affect Variables containing parameters when used within a `model_fn` (and so
  using it there might be more brittle).

  Returns:
    Behind the scenes, returns the result of the decorated function, potentially
    with a result modified to dynamically do contract checking, if necessary.
  """

  def decorator(func):
    """Invoked by Python when `@contract` is used as a function decorator."""

    def add_scope(*args: Any, **kwargs: Dict[Text, Any]):
      pretty_func_name = "{}.{}".format(func.__module__, func.__name__)
      with tf.variable_scope(pretty_func_name):
        return func(*args, **kwargs)
    return add_scope
  return decorator
