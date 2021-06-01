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
# pylint: disable=missing-docstring
# pylint: disable=g-long-lambda
"""NQ Server."""

import functools
from typing import Optional

from absl import logging
import grpc
from language.search_agents.muzero import common_flags

from language.search_agents import environment_pb2
from language.search_agents import environment_pb2_grpc



def get_nq_server() -> 'NQServer':
  return NQServer(common_flags.ENVIRONMENT_SERVER_SPEC.value)


class NQServer:
  """Backend NQ environment server."""

  def __init__(self, nq_env_server: str):
    """Init NQServer with its adress.

    Args:
      nq_env_server: str, Adress of the search environment server.
    """

    channel_creds = grpc.local_channel_credentials()


    channel = grpc.secure_channel(nq_env_server, channel_creds)

    grpc.channel_ready_future(channel).result(
        timeout=common_flags.RPC_DEADLINE.value)
    self._stub = environment_pb2_grpc.EnvironmentServiceStub(channel)

  def _call_rpc(self, stub_method, request):
    for i in range(common_flags.MAX_RPC_RETRIES.value):
      try:
        response = stub_method(
            request, timeout=common_flags.RPC_DEADLINE.value)
        break
      except grpc.RpcError as exception:
        logging.warning('RPC Exception in RPC method "%s", request "%s": %s',
                        str(stub_method), str(request), str(exception))
        if i < (common_flags.MAX_RPC_RETRIES.value - 1):
          # try again
          continue
        raise exception
    return response

  @functools.lru_cache(maxsize=1024)
  def get_documents(
      self,
      query: str,
      original_query: str,
      num_documents: Optional[int] = None,
      num_ir_documents: Optional[int] = None,
  ) -> environment_pb2.GetDocumentsResponse:
    """Get k best documents from search environment for given query.

    Args:
      query: str, Query for the retrieval.
      original_query: str, Original query.
      num_documents: Number of documents to retrieve.
      num_ir_documents: Number of documents to retrieve from underlying IR.

    Returns:
      The k top scoring documents as a list of common_pb2.Documents.
    """
    retrieval_mode = environment_pb2.RetrievalRequestType.Value(
        common_flags.RETRIEVAL_MODE.value)
    reader_mode = environment_pb2.ReaderRequestType.Value(
        common_flags.READER_MODE.value)

    if not num_documents:
      num_documents = common_flags.NUM_DOCUMENTS_TO_RETRIEVE.value
    if not num_ir_documents:
      num_ir_documents = num_documents
    req_query = environment_pb2.Query(
        query_with_operations=query, query=original_query)
    request = environment_pb2.GetDocumentsRequest(
        request_type=retrieval_mode,
        query=req_query,
        max_num_results=num_documents,
        max_num_ir_results=num_ir_documents,
        reader_request_type=reader_mode)
    docs = self._call_rpc(stub_method=self._stub.GetDocuments, request=request)
    return docs

  def get_query(
      self,
      index: Optional[int] = None,
      dataset_type: str = 'TRAIN') -> environment_pb2.GetQueryResponse:
    """Get query by index.

    Args:
      index: int, Index of the query.
      dataset_type: str, Dataset to choose from in ['TRAIN', 'DEV', 'TEST'].

    Returns:
      The corresponding query as enviornment_pb2.Query.
    """
    dataset = environment_pb2.DataSet.Value(common_flags.DATASET.value)
    if index:
      req = environment_pb2.GetQueryRequest(
          index=index, dataset=dataset, dataset_type=dataset_type)
    else:
      req = environment_pb2.GetQueryRequest(
          dataset=dataset, dataset_type=dataset_type)
    query = self._call_rpc(self._stub.GetQuery, req)
    return query
