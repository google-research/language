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
"""Util functions to handle preprocessing, shared across scripts."""
import collections
import random
import numpy as np

import tensorflow.compat.v1 as tf

gfile = tf.gfile

MAX_SEQUENCE_LENGTH = 60

task_input_indices = {"sst2": [0], "mnli": [8, 9]}


def build_vocab(sents_data,
                task_name,
                vocab_mode="downstream_vocab",
                vocab_path=None):
  """find all words in input corpus to build a vocabulary."""
  if vocab_mode == "bert_vocab":
    # Use a custom vocab to carry out filtering (such as BERT's word piece)
    with gfile.Open(vocab_path, "r") as f:
      vocab = f.read().strip().split("\n")
    # Filter out special tokens
    vocab = [x for x in vocab if x[0] != "[" and x[-1] != "]"]
    probs = [1.0 / len(vocab) for _ in vocab]

  elif vocab_mode == "full_corpus":
    # Use all words in a corpus of text to find out the vocabulary
    vocab = collections.Counter("\n".join(sents_data).split())
    vocab = [(k, v) for k, v in vocab.items()]
    vocab.sort(key=lambda x: x[1], reverse=True)
    vocab_total = sum([x[1] for x in vocab])

    probs = [float(x[1]) / vocab_total for x in vocab]
    vocab = [x[0] for x in vocab]

  elif "full_corpus_top_" in vocab_mode:
    full_vocab = collections.defaultdict(int)
    for sent in sents_data:
      for word in sent.split():
        full_vocab[word] += 1
    # Sort the vocabulary words according to their frequency
    full_vocab = sorted([(k, v) for k, v in full_vocab.items()],
                        key=lambda x: x[1],
                        reverse=True)
    # Take the top-k values from the vocabulary for the final list
    top_k_val = int(vocab_mode[len("full_corpus_top_"):])
    vocab = [x[0] for x in full_vocab[:top_k_val]]
    probs = [1.0 / len(vocab) for _ in vocab]

  elif vocab_mode == "downstream_vocab":
    vocab = collections.defaultdict(int)
    for sent in sents_data:
      for index in task_input_indices[task_name]:
        original_sent = sent.split("\t")[index].split()
        for word in original_sent:
          vocab[word] += 1

    vocab = [(k, v) for k, v in vocab.items()]
    vocab.sort(key=lambda x: x[1], reverse=True)
    vocab_total = sum([x[1] for x in vocab])

    probs = [float(x[1]) / vocab_total for x in vocab]
    vocab = [x[0] for x in vocab]

  else:
    probs = None
    vocab = None

  return vocab, probs


def get_lengths_pool(sents_data):
  return [len(x.split()) for x in sents_data]


def detokenize(tokens, vocab_mode="downstream_vocab"):
  if vocab_mode == "bert_vocab":
    string = " ".join(tokens)
    return string.replace(" ##", "")
  else:
    return " ".join(tokens)


def get_length(original_sequence, thief_lengths_pool, lengths_scheme):
  if lengths_scheme == "uniform_random":
    return random.randint(1, MAX_SEQUENCE_LENGTH)
  elif lengths_scheme == "thief_random":
    length = random.choice(thief_lengths_pool)
    while length >= MAX_SEQUENCE_LENGTH:
      length = random.choice(thief_lengths_pool)
    return length
  else:
    return len(original_sequence)


def sanitize_sentence(sample_tokens, vocab, vocab_dict):
  """Replace all OOV tokens with random tokens from vocab."""
  sanitized_sample_tokens = []
  for token in sample_tokens:
    if token in vocab_dict:
      sanitized_sample_tokens.append(token)
    else:
      sanitized_sample_tokens.append(random.choice(vocab))
  return sanitized_sample_tokens


def sample_thief_data(thief_data,
                      threshold=50,
                      sanitize=False,
                      vocab=None,
                      vocab_dict=None):
  """Sample sentences from thief dataset. Remove OOV if sanitize is True."""
  thief_sample = random.choice(thief_data)
  while len(thief_sample.split()) > threshold:
    thief_sample = random.choice(thief_data)

  if sanitize:
    thief_sample_words = thief_sample.split()
    sanitized_thief_sample_words = sanitize_sentence(
        sample_tokens=thief_sample_words, vocab=vocab, vocab_dict=vocab_dict)
    return " ".join(sanitized_thief_sample_words)
  else:
    return thief_sample


def sample_next_token(vocab, probs, scheme):
  if scheme.endswith("_uniform"):
    return random.choice(vocab)
  elif scheme.endswith("_freq") or scheme.endswith("_freq_thresh"):
    return vocab[np.argmax(np.random.multinomial(1, probs))]


def sample_next_sequence(vocab, probs, seq_length, scheme):
  """Sample a full sequence of random tokens using uniform or multinomials."""
  if scheme.endswith("_uniform"):
    return [random.choice(vocab) for _ in range(seq_length)]

  elif scheme.endswith("_freq"):
    bow_sent = np.random.multinomial(seq_length, probs)
    nonzero_indices = np.nonzero(bow_sent)[0]
    bow_indices = []
    for index, freq in zip(nonzero_indices, bow_sent[nonzero_indices]):
      bow_indices.extend([index for _ in range(freq)])
    assert len(bow_indices) == seq_length
    random.shuffle(bow_indices)
    return [vocab[i] for i in bow_indices]

  elif scheme.endswith("_freq_thresh"):
    freq_seq_length = int(np.sum(probs[:10000]) * seq_length)
    # use frequency sampling for first freq_seq_length tokens
    bow_indices = []
    if freq_seq_length > 0:
      bow_sent = np.random.multinomial(seq_length, probs[:10000])
      nonzero_indices = np.nonzero(bow_sent)[0]
      for index, freq in zip(nonzero_indices, bow_sent[nonzero_indices]):
        bow_indices.extend([index for _ in range(freq)])

    # Fill in remaining tokens with uniform random indices
    while len(bow_indices) < seq_length:
      bow_indices.append(random.randint(0, len(vocab) - 1))

    random.shuffle(bow_indices)
    return [vocab[i] for i in bow_indices]
  else:
    return None


def token_replace(token_list, vocab, probs, num_changes):
  """Replace tokens at random one at a time from the vocab."""
  original_len = len(token_list)
  for _ in range(num_changes):
    random_index = random.randint(0, len(token_list) - 1)
    token_list[random_index] = sample_next_token(vocab, probs, "_uniform")
  assert len(token_list) == original_len
  return token_list


def token_add(token_list, vocab, probs, num_changes):
  """Add new tokens at random positions in the original token_list."""
  original_len = len(token_list)
  for _ in range(num_changes):
    random_index = random.randint(0, len(token_list))
    token_list.insert(random_index, sample_next_token(vocab, probs, "_uniform"))
  assert len(token_list) == original_len + num_changes
  return token_list


def token_drop(token_list, num_changes):
  """Drop tokens at random from the original token_list."""
  original_len = len(token_list)
  for _ in range(num_changes):
    random_index = random.randint(0, len(token_list) - 1)
    del token_list[random_index]
  assert len(token_list) == original_len - num_changes
  return token_list
