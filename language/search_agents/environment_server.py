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
"""gRPC server for the EnvironmentService defined by environment.proto.

This server must be implemented in order to run the Search Agents.

"""

from concurrent import futures

from absl import app
from absl import flags
from absl import logging
import grpc
from language.search_agents import environment_pb2
from language.search_agents import environment_pb2_grpc


flags.DEFINE_integer('port', 50055, 'port to listen on')

FLAGS = flags.FLAGS


class EnvironmentServicer(environment_pb2_grpc.EnvironmentServiceServicer):
  """A gRPC server for the Environment Service."""

  def __init__(self, *args, **kwargs):
    self._stub = None

  def GetDocuments(self, request: environment_pb2.GetDocumentsRequest,
                   context) -> environment_pb2.GetDocumentsResponse:
    raise NotImplementedError('GetDocuments must be implemented.')

  def GetQuery(self, request: environment_pb2.GetQueryRequest,
               context) -> environment_pb2.GetQueryResponse:
    raise NotImplementedError('GetQuery must be implemented.')


def main(_):
  logging.info('Loading server...')
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  environment_pb2_grpc.add_EnvironmentServiceServicer_to_server(
      EnvironmentServicer(), server)
  server_creds = grpc.local_server_credentials()
  server.add_secure_port('[::]:{}'.format(FLAGS.port), server_creds)
  server.start()
  logging.info('Running server on port %s...', FLAGS.port)
  server.wait_for_termination()


if __name__ == '__main__':
  app.run(main)
