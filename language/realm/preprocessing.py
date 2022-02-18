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
"""Preprocessing servicer."""


import grpc
from grpc.framework.foundation import logging_pool
from language.realm import preprocessing_pb2_grpc
from six.moves import queue
from tensorflow.compat import v1 as tf


# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import rpc as contrib_rpc
except ImportError:
  tf.logging.warn("tf.contrib.rpc is not available.")
# pylint: enable=g-import-not-at-top



class PreprocessingServicer(preprocessing_pb2_grpc.PreprocessingServicer):
  """A Preprocessing server."""

  def __init__(self, max_queue_size, *args, **kwargs):
    super(PreprocessingServicer, self).__init__(*args, **kwargs)
    self._queue = queue.Queue(maxsize=max_queue_size)

  def PopExample(self, request, context):
    return self._queue.get()

  def PushExample(self, example, timeout):
    try:
      self._queue.put(example, timeout=timeout)
    except queue.Full:
      tf.logging.info("Failed to push example. Queue is full.")
      return  # Ignore

  def QueueSize(self):
    return self._queue.qsize()


def create_and_start_server(port, servicer):
  """Create a server for pushing examples and starts it.


  Args:
    port: Port for the gRPC server to listen on.
    servicer: A PreprocessingServicer object.

  Returns:
    A grpc.Server object.
  """
  server = grpc.server(logging_pool.pool(max_workers=25))
  preprocessing_pb2_grpc.add_PreprocessingServicer_to_server(servicer, server)
  address = "[::]:%d" % port
  tf.logging.info("Create preprocessing server at %s", address)
  server.add_insecure_port(address)
  server.start()
  return server




def push_examples(example_generator, port, max_queue_size, queue_timeout):
  """Create a server and continuously push examples from a generator.

  Args:
    example_generator: Generator that yields examples.
    port: Port for the gRPC server to listen on.
    max_queue_size: Number of example to queue.
    queue_timeout: Queue timeout argument. None to block forever, otherwise time
      in seconds.
  """
  servicer = PreprocessingServicer(max_queue_size=max_queue_size)
  server = create_and_start_server(port, servicer)
  for i, example in enumerate(example_generator):
    servicer.PushExample(example, timeout=queue_timeout)
    if i % 500 == 0:
      tf.logging.info("Attempted to push %d examples.", i)
      tf.logging.info("Queue size is %d.", servicer.QueueSize())
  server.stop(120.0)  # The argument is grace period in seconds.


def _make_rpc_op(address, timeout_in_ms):
  """Makes RPC calls to a server."""
  protocol = "grpc"
  return contrib_rpc.try_rpc(
      address=address,
      method="/language.realm.Preprocessing/PopExample",
      request="",
      protocol=protocol,
      timeout_in_ms=timeout_in_ms)


def pop_example(address, timeout_in_ms=10 * 1000 * 60):
  """Pop an example from the given rpc server.

  This op intentionally fails when the server is unavailable. This is because
  the RPC op silently fails to recover when the server recovers.

  Args:
    address (string): Server created by `push_examples`.
    timeout_in_ms (in): How long we are willing to wait for the next example.

  Returns:
    response (Tensor<string> []): Serialized tf.Example.
  """
  result = _make_rpc_op(address, timeout_in_ms)
  assert_ok = tf.assert_equal(
      x=result.status_code,
      y=tf.errors.OK,
      data=["RPC call failed", result.status_message])
  with tf.control_dependencies([assert_ok]):
    return tf.identity(result.response)


def dataset_from_preprocessors(preprocessing_servers,
                               rpcs_per_tf_op = 5):
  """Get dataset of serialized examples from a preprocessing job.

  Args:
    preprocessing_servers: BNS addresses for all preprocessing tasks.
    rpcs_per_tf_op: Number of RPCs to issue at once to a single server in a
      single TF op. Each TF op has some overhead, so usually a value like 20
      works well.

  Returns:
    Tensorflow dataset which yields single serialized examples.
  """

  def _ragged_where(cond, x):
    assert len(cond.shape) == 1
    assert len(x.shape) == 1
    return tf.gather(x, tf.squeeze(tf.where(cond), axis=1))

  def _pop_and_filter_valid(addresses):
    assert len(addresses.shape) == 1, "Expected replicated address tensor."
    minutes_in_ms = 60000
    result = _make_rpc_op(addresses, 10 * minutes_in_ms)
    is_ok = tf.math.equal(result.status_code, tf.errors.OK)
    return _ragged_where(is_ok, result.response)

  def _get_single_server_dataset(server):
    d = tf.data.Dataset.from_tensors(
        tf.tile(tf.expand_dims(server, 0), [rpcs_per_tf_op]))
    d = d.repeat()
    return d.map(_pop_and_filter_valid)

  dataset = tf.data.Dataset.from_tensor_slices(preprocessing_servers)
  dataset = dataset.apply(
      tf.data.experimental.parallel_interleave(
          _get_single_server_dataset,
          sloppy=True,
          cycle_length=len(preprocessing_servers),
      ))
  dataset = dataset.apply(tf.data.experimental.unbatch())
  return dataset
