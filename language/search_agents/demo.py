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
"""Demo for testing the environment server."""

from absl import app
from absl import flags
from absl import logging
import grpc

from language.search_agents import environment_pb2
from language.search_agents import environment_pb2_grpc

flags.DEFINE_string('server_address', 'localhost:50055',
                    'Address of the Environment Server.')
FLAGS = flags.FLAGS


def main(_):
  channel_creds = grpc.local_channel_credentials()
  channel = grpc.secure_channel(FLAGS.server_address, channel_creds)
  grpc.channel_ready_future(channel).result(timeout=10)
  stub = environment_pb2_grpc.EnvironmentServiceStub(channel)

  request = environment_pb2.GetQueryRequest()
  response = stub.GetQuery(request, timeout=10)
  logging.info('\n\nReceived GetQueryResponse:\n%s\n', response)


if __name__ == '__main__':
  app.run(main)
