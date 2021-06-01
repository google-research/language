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
"""Shared flags."""

from absl import flags
from language.search_agents import environment_pb2

# Core MuZero Architecture - see
# http://github.com/google-research/google-research/muzero/network.py.
# The `dynamics` function.  This is used to `evolve` the initial hidden_state
# during MCTS.
N_LSTM_LAYERS = flags.DEFINE_integer(
    'n_lstm_layers', 1,
    'Number of LSTM layers. Defines the `dynamics` function.')
LSTM_SIZE = flags.DEFINE_integer(
    'lstm_size', 512,
    'Size for each LSTM layer. Defines the `dynamics` function.')
# The policy, reward, and value heads, defined on top of the latent hidden
# state.
N_HEAD_HIDDEN_LAYERS = flags.DEFINE_integer(
    'n_head_hidden_layers', 1, 'Number of hidden layers in head MLPs.')
HEAD_HIDDEN_SIZE = flags.DEFINE_integer(
    'head_hidden_size', 32, 'Sizes of each head hidden layer in head MLPs.')

# NQ-Encoder Architecture - see
# http://github.com/google-research/language/search_agents/muzero/network.py.
BERT_SEQUENCE_LENGTH = flags.DEFINE_integer('bert_sequence_length', 512,
                                            'Sequence length for BERT input.')
BERT_CONFIG = flags.DEFINE_string(
    'bert_config', 'bert_config.json', 'Filename of the BERT config json. '
    'Download from gs://search_agents/bert/bert_config.json')
BERT_INIT_CKPT = flags.DEFINE_string(
    'bert_init_ckpt', 'bert_model.ckpt',
    'Filename of the BERT init checkpoint. '
    'Download from gs://search_agents/bert/bert_model.ckpt.data-00000-of-00001 '
    'and gs://search_agents/bert/bert_model.ckpt.index')
BERT_VOCAB = flags.DEFINE_string(
    'bert_vocab', 'vocab.txt', 'Filename of vocab.txt. '
    'Download from gs://search_agents/bert/vocab.txt')
ACTION_ENCODER_HIDDEN_SIZE = flags.DEFINE_integer(
    'action_encoder_hidden_size', 32,
    'The hidden size for the RNN action encoder. To disable action encoding '
    'use a value <= 0')
N_ACTIONS_ENCODING = flags.DEFINE_integer(
    'n_actions_encoding', 20, 'Maximum number of past actions to encode.')

# Optimizer Settings.
OPTIMIZER = flags.DEFINE_string('optimizer', 'adam',
                                'One of [sgd, adam, rmsprop, adagrad]')
LEARNING_RATE = flags.DEFINE_float('optimizer_learning_rate', 1e-3,
                                   'Learning rate.')
MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum')
LR_DECAY_FRACTION = flags.DEFINE_float('lr_decay_fraction', .999,
                                       'Final LR as a fraction of initial.')
LR_WARM_RESTARTS = flags.DEFINE_integer(
    'lr_warm_restarts', 0, 'If true (1), do warm restarts for LR decay.')
LR_DECAY_STEPS = flags.DEFINE_integer(
    'lr_decay_steps', int(1e6),
    'Decay steps for the cosine learning rate schedule.')

# MuZero hyper-parameters.
NUM_SIMULATIONS = flags.DEFINE_integer('num_simulations', 100,
                                       'Number of simulations per MCTS.')

NUM_UNROLL_STEPS = flags.DEFINE_integer(
    'num_unroll_steps', 5, 'Number of unroll steps when training.')

ONE_MINUS_DISCOUNT = flags.DEFINE_float('one_minus_discount', 0.1,
                                        '1-discount factor used by MuZero.')
DIRICHLET_ALPHA = flags.DEFINE_float('dirichlet_alpha', .15, 'Dirichlet alpha.')
ROOT_EXPLORATION_FRACTION = flags.DEFINE_float('root_exploration_fraction', 0.2,
                                               'Root exploration fraction.')
PB_C_BASE = flags.DEFINE_integer(
    'pb_c_base', 19652,
    'PB C Base, see equation (2) / the Upper Confidence Bound part of MCTS in '
    'https://arxiv.org/pdf/1911.08265.pdf.')
PB_C_INIT = flags.DEFINE_float(
    'pb_c_init', 1.25,
    'PB C Init, see equation (2) / the Upper Confidence Bound part of MCTS')
VALUE_ENCODER_STEPS = flags.DEFINE_integer(
    'value_encoder_steps', 8,
    'Number of buckets used for discretizing value in MuZero. If 0, assumes 1 '
    'step per integer.')
REWARD_ENCODER_STEPS = flags.DEFINE_integer(
    'reward_encoder_steps', None,
    'Number of buckets used for discretizing reward in MuZero. If 0, assumes 1 '
    'step per integer.  If None, uses `value_encoder_steps`.')
PLAY_MAX_AFTER_MOVES = flags.DEFINE_integer(
    'play_max_after_moves', -1,
    'Play the argmax after this many game moves. -1 means never play argmax.')
USE_SOFTMAX_FOR_ACTION_SELECTION = flags.DEFINE_integer(
    'use_softmax_for_action_selection', 1,
    'Whether to use softmax (1) for regular histogram sampling (0).')
TEMPERATURE = flags.DEFINE_float('temperature', 1.0,
                                 'for softmax sampling of actions')
EMBED_ACTIONS = flags.DEFINE_integer(
    'embed_actions', 0,
    'use action embeddings (1) instead of one hot encodings (0)')

# Pretraining specific.
PRETRAINING_NUM_UNROLL_STEPS = flags.DEFINE_integer(
    'pretraining_num_unroll_steps', 5,
    'Number of unroll steps for the pretraining.  Ideally it is 5.')
PRETRAINING_SKIP_IMMEDIATE_STOP_PROBABILITY = flags.DEFINE_float(
    'pretraining_skip_immediate_stop_probability', 0.0,
    'Probability with which examples that immediately stop (with the initial '
    'results) are skipped.')
PRETRAINING_TRUNCATE_EXAMPLES = flags.DEFINE_boolean(
    'pretraining_truncate_examples', False,
    'If true, the supervised episodes will be truncated to the optimal '
    'stopping point.')

# Configures env.py.
IDF_LOOKUP_PATH = flags.DEFINE_string(
    'idf_lookup_path', 'word_to_idf_reduced.pickle',
    'Filename of the sstable for the idf scores. '
    'Download from gs://search_agents/word_to_idf_reduced.pickle')
GLOBAL_TRIE_PATH = flags.DEFINE_string(
    'global_trie_path', None,
    'If set, unpickles the global trie to speed up starting time.  If not set, '
    'builds the trie from scratch.')
EXCLUDE_PUNCTUATION_FROM_TRIE = flags.DEFINE_integer(
    'exclude_punctuation_from_trie', 1,
    'If 1, excludes punctuation from the trie.')

MAX_NUM_ACTIONS = flags.DEFINE_integer(
    'max_num_actions', 100,
    'Maximum number of actions that go into a single tree contruction.')
MAX_NUM_REQUESTS = flags.DEFINE_integer(
    'max_num_requests', 5, 'Maximum number of requests made to lucene.')
MASK_ILLEGAL_ACTIONS = flags.DEFINE_integer(
    'mask_illegal_actions', 1,
    'If true (1), provide a mask for legal next actions.')
RESTRICT_VOCABULARY = flags.DEFINE_integer(
    'restrict_vocabulary', 1,
    'If true (1), restrict the vocabulary to the words seen in the context.')
RELEVANCE_FEEDBACK_RESTRICT = flags.DEFINE_integer(
    'relevance_feedback_restrict', 0,
    'If true (1), further restrict the vocabulary using the relevance '
    'feedback method.')
FRACTION_OF_RELEVANCE_FEEDBACK_RESTRICTED_STEPS = flags.DEFINE_float(
    'fraction_of_relevance_feedback_restricted_steps', 0.5,
    'The fraction of steps that use relevance feedback restricted vocabulary '
    'during MCTS.')
MAX_RELEVANCE_FEEDBACK_RESTRICTED_STEPS = flags.DEFINE_integer(
    'max_relevance_feedback_restricted_steps', int(1e6),
    'The max number of steps it takes to decrease the '
    'fraction_of_relevance_feedback_restricted_steps to a value of 0.')
SPLIT_VOCABULARY_BY_TYPE = flags.DEFINE_integer(
    'split_vocabulary_by_type', 1,
    'If true (1), the vocabulary is restricted to the words seen in the '
    'context, i.e. restrict_vocabulary=1. The vocabulary is split into '
    'query|passage|answer types based on where it is in the context.')
CONTEXT_TITLE_SIZE = flags.DEFINE_integer(
    'context_title_size', 10,
    'The max size for the title context that goes into the state string.')
CONTEXT_WINDOW_SIZE = flags.DEFINE_integer(
    'context_window_size', 70,
    'The window size for the context that goes into the state string.')
USE_AGGREGATED_DOCUMENTS = flags.DEFINE_integer(
    'use_aggregated_documents', 1,
    'If 1, return aggregated documents collected throughout the episode. '
    'The documents are sorted by mr_score and num_documents_to_retrieve is '
    'returned.')
USE_DOCUMENT_TITLE = flags.DEFINE_integer(
    'use_document_title', 1,
    'If 1, include the document title in the state as well as enabling _Wt_ '
    'vocabulary type.')

GRAMMAR_TYPE = flags.DEFINE_enum(
    'grammar_type', 'one_term_at_a_time_with_add_term_only', [
        'bert', 'one_term_at_a_time', 'add_term_only',
        'one_term_at_a_time_with_add_term_only'
    ], 'What are the available terminals for reformulations.  `bert` uses '
    'WordPieces directly.')

# Scoring.
REWARD = flags.DEFINE_enum('reward', 'curiosity+dcg', ['curiosity+dcg'],
                           'The reward for which we optimize.')
REWARD_INTERPOLATION_VALUE = flags.DEFINE_float(
    'reward_interpolation_value', 0.0,
    'Interpolation weight for the rewards if it consists of two individual. '
    'NOTE:  For a composite reward with name `score1+score2`, the '
    'final score will be computed as (reward_interpolation_value * score1 + '
    '(1-reward_interpolation_value) * score2).  So setting it to 1.0 puts all '
    'weight on the first component, and setting it to 0.0 puts all weight on '
    'the second component.')
INCOMPLETE_TREE_PENALTY = flags.DEFINE_float(
    'incomplete_tree_penalty', 0,
    'Penalty on the score for not finishing a tree in the max number of steps.')
EMPTY_RESULTS_PENALTY = flags.DEFINE_float(
    'empty_results_penalty', -1,
    'Penalty on the score for producing a query that results in zero retrieved '
    'documents.')
ADD_FINAL_REWARD = flags.DEFINE_float(
    'add_final_reward', 1,
    'If true (1), additionaly to the intermediate reward the final reward is '
    'added when the stop-action is chosen.')

STOP_AFTER_SEEING_NEW_RESULTS = flags.DEFINE_integer(
    'stop_after_seeing_new_results', 0,
    'If non-zero, uses pen-ultimate results when [stop]-ing, rather than the '
    'latest set of results.')
INACTION_PENALTY = flags.DEFINE_float(
    'inaction_penalty', -1.0,
    'Reward penalty on the learner for not even issuing a query at all.')

SKIP_SUCCESSFUL_TRAIN_EPISODE_PROB = flags.DEFINE_float(
    'skip_successful_train_episode_prob', 0.0,
    'Probability with which we skip an already successful training episode.')

# Statistics.
VISUALIZE_MCTS = flags.DEFINE_integer('visualize_mcts', 1, '')
VISUALIZE_MIN_VISIT_COUNT = flags.DEFINE_integer(
    'visualize_min_visit_count', 1,
    'Threshold on visit count for the visualization of the MCTS.')

# Mostly to do with server.py.
# Basic server configuration. These are not vizier tuned.
ENVIRONMENT_SERVER_SPEC = flags.DEFINE_string(
    'environment_server_spec', 'localhost:50055',
    'Server spec of the Environment server.')
RETRIEVAL_MODE = flags.DEFINE_enum(
    'retrieval_mode', 'LUCENE',
    list(environment_pb2.RetrievalRequestType.keys()),
    'Retrieval mode. Values defined in environment.proto')
READER_MODE = flags.DEFINE_enum(
    'reader_mode', 'DPR_READER', list(environment_pb2.ReaderRequestType.keys()),
    'Reader mode. Values defined in environment.proto')
DATASET = flags.DEFINE_enum(
    'dataset', 'NATURAL_QUESTIONS', list(environment_pb2.DataSet.keys()),
    'Dataset to be used. Values defined in environment.proto')
RPC_DEADLINE = flags.DEFINE_integer('rpc_deadline', 120,
                                    'RPC deadline in seconds.')
MAX_RPC_RETRIES = flags.DEFINE_integer('max_rpc_retries', 3,
                                       'Max number of retries.')

NUM_DOCUMENTS_TO_RETRIEVE = flags.DEFINE_integer(
    'num_documents_to_retrieve', 5,
    'Number of documents to retrieve from the environment. '
    'This determines the maximum number of the docs_gold_answer index.')
NUM_IR_DOCUMENTS_TO_RETRIEVE = flags.DEFINE_integer(
    'num_ir_documents_to_retrieve', 5,
    'Number of documents to retrieve from IR. When this is larger than '
    'num_documents_to_retrieve, only num_documents_to_retrieve with the '
    'highest MR score is returned.')
K = flags.DEFINE_integer('k', 5, 'The value of k for computing various scores.')
