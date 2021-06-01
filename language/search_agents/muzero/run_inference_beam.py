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
"""Run the MuZero agent on a jsonl dataset."""

import functools
import json
from typing import List, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam import metrics
import grpc
from language.search_agents import environment_pb2
from language.search_agents.muzero import agent_lib
from language.search_agents.muzero import env
import numpy as np
import tensorflow as tf

from muzero import core
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


flags.DEFINE_string('initial_inference_model_server_spec', 'localhost:10000',
                    'Model server address for initial inference.')
flags.DEFINE_string('initial_inference_model_name', 'default',
                    'Model name for initial inference.')

flags.DEFINE_string('recurrent_inference_model_server_spec', 'localhost:10001',
                    'Model server address for recurrent inference.')
flags.DEFINE_string('recurrent_inference_model_name', 'default',
                    'Model name for recurrent inference.')

_INPUT_JSONL = flags.DEFINE_string('input_jsonl', None, 'Input JSONL file.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Output path.')

_RUNS_PER_QUERY = flags.DEFINE_integer('runs_per_query', 1,
                                       'How often do we play each query.')
_NUM_RETRIES = flags.DEFINE_integer('num_retries', 20,
                                    'How often do we retry each query.')

flags.mark_flag_as_required(_INPUT_JSONL.name)
flags.mark_flag_as_required(_OUTPUT_PATH.name)

FLAGS = flags.FLAGS


def send_initial_inference_request(
    predict_service: prediction_service_pb2_grpc.PredictionServiceStub,
    inputs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> core.NetworkOutput:
  """Initial inference for the agent, used at the beginning of MCTS."""
  input_ids, input_type_ids, input_features, action_history = inputs

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.initial_inference_model_name
  request.model_spec.signature_name = 'initial_inference'

  request.inputs['input_ids'].CopyFrom(
      tf.make_tensor_proto(values=np.expand_dims(input_ids, axis=0)))
  request.inputs['segment_ids'].CopyFrom(
      tf.make_tensor_proto(values=np.expand_dims(input_type_ids, axis=0)))
  request.inputs['features'].CopyFrom(
      tf.make_tensor_proto(values=np.expand_dims(input_features, axis=0)))
  request.inputs['action_history'].CopyFrom(
      tf.make_tensor_proto(values=np.expand_dims(action_history, axis=0)))
  response = predict_service.Predict(request)

  # Parse and `unbatch` the response.
  map_names = {
      f'output_{i}': v for (i, v) in enumerate([
          'value', 'value_logits', 'reward', 'reward_logits', 'policy_logits',
          'hidden_state'
      ])
  }
  outputs = {
      map_names[k]: tf.make_ndarray(v).squeeze()
      for k, v in response.outputs.items()
  }

  return core.NetworkOutput(**outputs)


def send_recurrent_inference_request(
    hidden_state: np.ndarray, action: np.ndarray,
    predict_service: prediction_service_pb2_grpc.PredictionServiceStub
) -> core.NetworkOutput:
  """Recurrent inference for the agent, used during MCTS."""
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.recurrent_inference_model_name
  request.model_spec.signature_name = 'recurrent_inference'

  request.inputs['hidden_state'].CopyFrom(
      tf.make_tensor_proto(values=tf.expand_dims(hidden_state, axis=0)))
  request.inputs['action'].CopyFrom(
      tf.make_tensor_proto(
          values=np.expand_dims(action, axis=0).astype(np.int32)))
  response = predict_service.Predict(request)

  # Parse and `unbatch` the response.
  map_names = {
      f'output_{i}': v for (i, v) in enumerate([
          'value', 'value_logits', 'reward', 'reward_logits', 'policy_logits',
          'hidden_state'
      ])
  }
  outputs = {
      map_names[k]: tf.make_ndarray(v).squeeze()
      for k, v in response.outputs.items()
  }

  return core.NetworkOutput(**outputs)


def run_episode(
    episode: core.Episode, mzconfig: core.MuZeroConfig,
    initial_inference_service: prediction_service_pb2_grpc.PredictionServiceStub,
    recurrent_inference_service: prediction_service_pb2_grpc.PredictionServiceStub,
    initial_inference_cache: List[core.NetworkOutput],
    counter: metrics.Metrics.DelegatingCounter) -> None:
  """Runs and mutates `episode`.  Returns MCTS visualizations for each step."""
  step_num = 0
  while not episode.terminal() and len(episode.history) < mzconfig.max_moves:
    # Get observation.
    current_observation = episode.make_image(state_index=-1)

    # Prepare MCTS.
    # We may already have played this, so avoid swamping the predict servers.
    if len(initial_inference_cache) >= step_num + 1:
      initial_inference_output = initial_inference_cache[step_num]
      counter.inc()
    else:
      initial_inference_output = send_initial_inference_request(
          predict_service=initial_inference_service, inputs=current_observation)
      initial_inference_cache.append(initial_inference_output)

    legal_actions = episode.legal_actions()
    root = core.prepare_root_node(
        config=mzconfig,
        legal_actions=legal_actions,
        initial_inference_output=initial_inference_output)

    # Run MCTS.
    core.run_mcts(
        config=mzconfig,
        root=root,
        action_history=episode.action_history(),
        legal_actions_fn=episode.legal_actions,
        recurrent_inference_fn=functools.partial(
            send_recurrent_inference_request,
            predict_service=recurrent_inference_service))

    # Pick action.
    action = core.select_action(
        config=mzconfig,
        num_moves=len(episode.history),
        node=root,
        train_step=0,
        use_softmax=mzconfig.use_softmax_for_action_selection)

    episode.apply(action)
    step_num += 1


class EpisodeFn(beam.DoFn):
  """Maps a query id to the corresponding episode."""

  def __init__(self):
    super().__init__()

    self.environment = ...  # type: env.NQEnv
    self.mzconfig = ...  # type: core.MuZeroConfig
    self.stub = ...  # type: prediction_service_pb2.PredictionServiceStub

    self.started = metrics.Metrics.counter('agent_inference', 'episode_started')
    self.completed = metrics.Metrics.counter('agent_inference',
                                             'episode_completed')
    self.cache_hit = metrics.Metrics.counter('agent_inference',
                                             'initial_inference_cache_hit')
    self.failed = metrics.Metrics.counter('agent_inference', 'episode_failed')
    self.retries = metrics.Metrics.counter('agent_inference', 'episode_retries')

  def start_bundle(self):
    env_descriptor = env.get_descriptor()
    self.environment = env.create_environment(
        task=42,
        training=False,
        stop_after_seeing_new_results=FLAGS.stop_after_seeing_new_results)
    self.mzconfig = agent_lib.muzeroconfig_from_flags(
        env_descriptor=env_descriptor)
    self.mzconfig.max_num_action_expansion = 100

    self.initial_inference_stub = (
        prediction_service_pb2_grpc.PredictionServiceStub(
            grpc.secure_channel(FLAGS.initial_inference_model_server_spec,
                                grpc.local_channel_credentials())))
    self.recurrent_inference_stub = (
        prediction_service_pb2_grpc.PredictionServiceStub(
            grpc.secure_channel(FLAGS.recurrent_inference_model_server_spec,
                                grpc.local_channel_credentials())))

  def process(self, element, *args, **kwargs):
    # We really want to get results for all episodes.  So if an episode fails,
    # we will retry up to `_NUM_RETRIES` time.
    # We still log all failed episodes so we could run a smaller follow-up
    # job and join the data, but hopefully, 20 retries does the trick.

    element_json = json.loads(element)
    test_query = environment_pb2.GetQueryResponse(
        query=element_json['question'], gold_answer=element_json['answer'])

    for _ in range(_RUNS_PER_QUERY.value):
      self.started.inc()

      attempts = 0
      initial_inference_cache = []
      while True:
        attempts += 1
        # Note:  Any interaction with the environment might fail, even resetting
        #        an episode.  So move everything inside the try-body.
        try:
          episode = self.mzconfig.new_episode(
              environment=self.environment, index=test_query)
          run_episode(
              episode=episode,
              mzconfig=self.mzconfig,
              initial_inference_service=self.initial_inference_stub,
              recurrent_inference_service=self.recurrent_inference_stub,
              initial_inference_cache=initial_inference_cache,
              counter=self.cache_hit)
          metrics_dict = self.environment.special_episode_statistics_learner(
              return_as_dict=True)
          self.completed.inc()

          state_dict = self.environment.state.json_repr()
          output = {
              'query': self.environment.state.original_query.query,
              'state': state_dict,
              'metrics': metrics_dict,
              'gold_answer': element_json['answer'],
          }
          yield json.dumps(output)
          break
        except (grpc.RpcError, core.RLEnvironmentError) as e:
          if attempts > _NUM_RETRIES.value:
            logging.info('Episode permanently failed: %s', e)
            self.failed.inc()
            yield beam.pvalue.TaggedOutput(tag='failed', value=element)
            break
          else:
            logging.info('Episode failed: %s, retrying', e)
            self.retries.inc()


def pipeline(root):
  """Maps all dev-set query ids to their episodes."""

  completed, failed = (
      root
      | 'Read JSONL' >> beam.io.ReadFromText(file_pattern=_INPUT_JSONL.value)
      | 'RunEpisodes' >> beam.ParDo(EpisodeFn()).with_outputs(
          'failed', main='completed'))
  _ = (completed | 'SaveCompleted' >> beam.io.WriteToText(_OUTPUT_PATH.value))
  _ = (
      failed
      | 'SaveFailed' >> beam.io.WriteToText(f'{_OUTPUT_PATH.value}_failed'))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  beam_options = None
  with beam.Pipeline(beam_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
