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
"""Run the T5 agent on a jsonl dataset."""

import functools
import json
from typing import Callable, Sequence, List

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam import metrics
import grpc
from language.search_agents import environment_pb2
from language.search_agents import environment_pb2_grpc
from language.search_agents.t5 import t5_agent_lib
import tensorflow as tf
import tensorflow_text as tf_text  # pylint: disable=unused-import

from google.protobuf import text_format


_NUM_RESULTS = flags.DEFINE_integer(
    'num_results', 5, 'How many documents are retrieved / stored.')
_EPISODE_LENGTH = flags.DEFINE_integer(
    'episode_length', 5,
    'Maximum number of steps per episode, including the initial step.')

_CONTEXT_LENGTH = flags.DEFINE_integer(
    'context_length', 70, 'Size of context snippet centered on answer.')
_TITLE_LENGTH = flags.DEFINE_integer(
    'title_length', 10, 'Max number of tokens for document titles.')
_MAX_DOCUMENTS = flags.DEFINE_integer('max_documents', 5,
                                      'Maximum number of documents in state.')
_AGGREGATE = flags.DEFINE_bool(
    'aggregate', True,
    'Aggregate results over steps, keeping top documents by MR score.')

_INPUT_JSONL = flags.DEFINE_string('input_jsonl', None, 'Input JSONL file.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Output path.')

_T5_SAVED_MODEL = flags.DEFINE_string('t5_saved_model', None,
                                      'Location of T5 saved model.')
_ENVIRONMENT_SERVER_SPEC = flags.DEFINE_string(
    'environment_server_spec', 'localhost:50055',
    'Address to the Environment Server.')

flags.mark_flag_as_required(_INPUT_JSONL.name)
flags.mark_flag_as_required(_OUTPUT_PATH.name)
flags.mark_flag_as_required(_T5_SAVED_MODEL.name)


def get_documents(
    query_for_reader: str,
    query_for_retriever: str,
    stub: environment_pb2_grpc.EnvironmentServiceStub,
    retrieval_type: environment_pb2.RetrievalRequestType = environment_pb2
    .LUCENE,
    reader_type: environment_pb2.ReaderRequestType = environment_pb2.DPR_READER,
    num_results: int = 5,
    num_retries: int = 3) -> List[environment_pb2.Document]:
  """Fetch documents and return them sorted by reader scores.

  Args:
    query_for_reader:  Query assumed by the machine reader.  This should be the
      "base" query, i.e. the original query (q0) for which an answer is
      requested and which does not contain operators.
    query_for_retriever:  A potentially modified version of the base query; in
      particular, this query can contain search operators such as + or - and
      field syntax.
    stub:  An EnvironmentServiceStub used for making requests.
    retrieval_type:  What kind fo retrieval is to be used.  Note that only
      LUCENE explicitly supports the use of search operators.
    reader_type:  Which machine reader is used for extracting short answers. The
      extracted answers - one per result - are used both for sorting the
      documents and for determining the "context" available for picking
      refinement terms.
    num_results:  How many documents are to be returned.
    num_retries:  Try at most that many times to obtain results.  If the RPC
      fails more than num_retries time, get_documents will (re)raise the
      RPCException.

  Returns:
    A list of document results for `query_for_retriever`, sorted by the answer
    scores for the answers extracted from the documents assuming
    `query_for_reader`.
  """

  request = environment_pb2.GetDocumentsRequest(
      request_type=retrieval_type,
      reader_request_type=reader_type,
      query=environment_pb2.Query(
          query=query_for_reader, query_with_operations=query_for_retriever),
      max_num_results=num_results)

  attempts = 0
  while True:
    try:
      response = stub.GetDocuments(request)
      break
    except grpc.RpcError as e:
      attempts += 1
      if attempts <= num_retries:
        continue
      raise e

  return sorted(
      response.documents, key=lambda x: x.answer.mr_score, reverse=True)


class RunT5EpisodeFn(beam.DoFn):
  """Generate a synthetic episode."""

  def __init__(self):
    super().__init__()

    self.get_documents_fn = ...  # type: Callable[[str, str], List[environment_pb2.Document]]
    self.run_episode_fn = ...  # type: Callable[[str, List[str]], environment_pb2.RelevanceFeedbackEpisodeExample]

    self.improved = metrics.Metrics.counter('t5agent', 'improved')

    self.started = metrics.Metrics.counter('t5agent', 'started')
    self.finished = metrics.Metrics.counter('t5agent', 'finished')
    self.steps = metrics.Metrics.counter('t5agent', 'steps')
    self.stop = metrics.Metrics.counter('t5agent', 'stop')

    self.no_documents = metrics.Metrics.counter('t5agent', 'no_documents')
    self.t5_unparseable = metrics.Metrics.counter('t5agent', 't5_unparseable')

    self.step_counters = []
    for steps in range(_EPISODE_LENGTH.value + 1):
      self.step_counters.append(
          metrics.Metrics.counter('t5agent', f't5_episode_length_{steps}'))

  def start_bundle(self):
    environment_stub = environment_pb2_grpc.EnvironmentServiceStub(
        grpc.secure_channel(_ENVIRONMENT_SERVER_SPEC.value,
                            grpc.local_channel_credentials()))

    self.get_documents_fn = functools.partial(
        get_documents, stub=environment_stub, num_results=_NUM_RESULTS.value)

    self.t5_saved_model = tf.saved_model.load(_T5_SAVED_MODEL.value, ['serve'])
    self.run_episode_fn = functools.partial(
        self.run_episode,
        get_documents_fn=self.get_documents_fn,
        query_t5_fn=self.predict_t5)

  def predict_t5(self, query_string: str) -> str:
    return self.t5_saved_model.signatures['serving_default'](tf.constant(
        [query_string]))['outputs'].numpy()[0].decode('utf-8')

  def run_episode(
      self, q0: str, answers: Sequence[str],
      get_documents_fn: Callable[[str, str], List[environment_pb2.Document]],
      query_t5_fn: Callable[[str], str]
  ) -> environment_pb2.RelevanceFeedbackEpisodeExample:
    self.started.inc()

    episode = environment_pb2.RelevanceFeedbackEpisodeExample()
    episode.q0 = q0
    episode.answer = '\t'.join(answers)

    addition_terms = []
    subtraction_terms = []
    or_terms = []
    all_docs = {}

    for i in range(_EPISODE_LENGTH.value):
      query_for_lucence = t5_agent_lib.make_query(episode.q0, addition_terms,
                                                  subtraction_terms, or_terms)
      new_documents = get_documents_fn(q0, query_for_lucence)
      new_documents.sort(key=lambda x: x.answer.mr_score, reverse=True)
      if not new_documents:
        self.no_documents.inc()
        break

      current_results = new_documents
      if _AGGREGATE.value:
        t5_agent_lib.add_to_dict(new_documents, all_docs)
        current_results = (
            t5_agent_lib.get_aggregated_documents(all_docs,
                                                  _MAX_DOCUMENTS.value))

      score = t5_agent_lib.ndcg(current_results, answers)
      episode.final_reward = score

      if i == 0:
        episode.initial_reward = score
        for d in new_documents:
          result_doc = episode.initial_documents.add()
          result_doc.CopyFrom(d)
      else:
        step = episode.steps.add()
        step.reward = score

        step.add_term.field = addition_terms[-1].field
        step.add_term.term = addition_terms[-1].term
        step.add_term.term_type = addition_terms[-1].term_type

        step.subtract_term.field = subtraction_terms[-1].field
        step.subtract_term.term = subtraction_terms[-1].term
        step.subtract_term.term_type = subtraction_terms[-1].term_type

        step.or_term.field = or_terms[-1].field
        step.or_term.term = or_terms[-1].term
        step.or_term.term_type = or_terms[-1].term_type

        for d in new_documents:
          result_doc = step.results.add()
          result_doc.CopyFrom(d)

      state = t5_agent_lib.state_from_documents(
          current_results,
          max_title_tokens=_TITLE_LENGTH.value,
          max_context_tokens=_CONTEXT_LENGTH.value)
      query_with_operators = t5_agent_lib.query_to_prompt(
          episode.q0, addition_terms, subtraction_terms, or_terms)
      t5_prompt = ' '.join((query_with_operators, state))
      t5_response = query_t5_fn(t5_prompt)

      success, stop, new_terms = t5_agent_lib.parse_t5_response(t5_response)
      if not success:
        self.t5_unparseable.inc()
        break
      if stop:
        self.stop.inc()
        break

      addition_term, subtraction_term, or_term = new_terms
      addition_terms.append(addition_term)
      subtraction_terms.append(subtraction_term)
      or_terms.append(or_term)

      self.steps.inc()

    episode.final_query = query_for_lucence
    self.finished.inc()
    self.step_counters[len(episode.steps)].inc()
    return episode

  def process(self, element: str):
    element_json = json.loads(element)
    result = self.run_episode_fn(element_json['question'],
                                 element_json['answer'])
    yield json.dumps({
        'query': element_json['question'],
        'episode': text_format.MessageToString(result)
    })


def pipeline(root):
  _ = (
      root
      | 'Read JSONL' >> beam.io.ReadFromText(file_pattern=_INPUT_JSONL.value)
      | 'Run Episodes' >> beam.ParDo(RunT5EpisodeFn())
      | 'Reshard' >> beam.Reshuffle()
      | 'Save' >> beam.io.WriteToText(_OUTPUT_PATH.value))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  beam_options = None
  with beam.Pipeline(beam_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
