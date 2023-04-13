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
"""Custom callbacks for Fever."""
import collections
import json
import os
import sys


from absl import logging
from fever_scorer.scorer import fever_score
from language.serene import constants
from language.serene import types
from language.serene import util
import numpy as np
import tensorflow.compat.v2 as tf



class ClaimEvidence(NamedTuple):
  match_prob: float
  verify_pred: int  # The index of the class, taken from probs.argmax()
  metadata: types.Json
  match_label: int
  claim_label: int


FlatClaimEvidence = List[ClaimEvidence]


def unwrap_tensor(tensor):
  try:
    value = tensor.numpy()
    if isinstance(value, bytes):
      return value.decode('utf8')
    else:
      return value
  except AttributeError:
    return tensor


def unwrap_np(array):
  if isinstance(array, np.ndarray):
    return array.tolist()
  else:
    return array


def flatten_and_filter_evidence_to_fever(claim_id,
                                         flat_evidence,
                                         max_evidence):
  """Convert flattened, ranked evidence to fever format.

  This filters out gold examples not present in the retrieval
  scrape, and converts to the format expected by the fever scoring script.

  Args:
    claim_id: The claim_id
    flat_evidence: The flattened, ranked evidence predictions
    max_evidence: The maximum evidence to use

  Returns:
    Fever formatted evidence
  """
  candidate_evidence = [
      evidence for evidence in flat_evidence
      if evidence.metadata['tfidf_candidate']
  ]

  # Get the first document, then get the fever verification prediction.
  verify_pred = candidate_evidence[0][1]
  return {
      'id':
          claim_id,
      'predicted_label':
          constants.FEVER_CLASSES[verify_pred],
      'predicted_evidence': [[
          candidate.metadata['wikipedia_url'], candidate.metadata['sentence_id']
      ] for candidate in candidate_evidence[:max_evidence]]
  }


def compute_claim_recall(
    claim_json,
    ranked_claim_evidence):
  """Compute the recall position for a single claim.

  Args:
    claim_json: The json of the fever claim
    ranked_claim_evidence: A ranked list of evidence

  Returns:
    The recall position
  """
  gold_evidence: Set[Tuple[Text, int]] = set()
  for evidence_set in claim_json['evidence']:
    for evidence in evidence_set:
      if evidence is not None:
        if evidence[2] is not None and evidence[3] is not None:
          gold_evidence.add((evidence[2], evidence[3]))

  retrieved_evidence = []
  for example in ranked_claim_evidence:
    if example.metadata['tfidf_candidate']:
      retrieved_evidence.append(example.metadata)

  if not retrieved_evidence:
    return None
  recall_position = None
  for idx, doc in enumerate(retrieved_evidence, start=1):
    wikipedia_url = doc['wikipedia_url']
    sentence_id = doc['sentence_id']
    if (wikipedia_url, sentence_id) in gold_evidence:
      recall_position = idx
      break

  if recall_position is None:
    return sys.maxsize
  else:
    return recall_position


def model_recall_by_position(
    verifiable_dev,
    ranked_evidence):
  """Compute the evidence recall for each claim and list of ranked evidence.

  Args:
    verifiable_dev: The subset of validation set that is verifiable.
    ranked_evidence: Per claim, ranked evidence.

  Returns:
    For each claim, the index of the first correct document, or sys.maxsize
    if there were none.
  """
  recall = []
  n_missing_evidence = 0
  for claim_json in verifiable_dev:
    claim_id = claim_json['id']
    if claim_id in ranked_evidence:
      recall_position = compute_claim_recall(claim_json,
                                             ranked_evidence[claim_id])
      recall.append(recall_position)
    else:
      n_missing_evidence += 1
      recall.append(sys.maxsize)
  logging.info('N Empty Evidence: %s', n_missing_evidence)

  return recall  # pytype: disable=bad-return-type  # always-use-return-annotations


def partition_preds_by_scrape_type(verify_predictions,
                                   evidence_predictions,
                                   val_examples):
  """Partition predictions by which scrape_type they come from.

  The validation fold contains four sets of evidence: drqa, lucene, ukp_pred,
  and ukp_wiki. The intention is in this function to partition these into
  four sets so that they can each be scored separately to measure the
  difference between them on models that are trained on one of these
  (train_scrape).

  Args:
    verify_predictions: Claim verification predictions to partition, a 3-dim
      tensor of probabilities (one for each class)
    evidence_predictions: Evidence predictions to partition, a scalar
      probability of matching
    val_examples: Validation examples, typically all of
      FeverMetricsCallback._validation_flat

  Returns:
    Predictions and examples partitioned by scrape type
  """
  partitioned_verify = collections.defaultdict(list)
  partitioned_match = collections.defaultdict(list)
  partitioned_example = collections.defaultdict(list)
  for verify_probs, match_prob, example in zip(verify_predictions,
                                               evidence_predictions,
                                               val_examples):
    struct, _ = example
    metadata = json.loads(unwrap_tensor(struct['metadata']))
    scrape_type = metadata['scrape_type']
    partitioned_verify[scrape_type].append(verify_probs)
    partitioned_match[scrape_type].append(match_prob)
    partitioned_example[scrape_type].append(example)
  return partitioned_verify, partitioned_match, partitioned_example


def group_claim_predictions(verify_predictions,
                            evidence_predictions,
                            val_examples):
  """Group predictions by their claim_id, sorted by score.

  Args:
    verify_predictions: Predictions from the model, [[p0], [p1],...]
    evidence_predictions: Predictions from the model, [[p0, p1, p2], ...]
    val_examples: Validation examples, a subset from self._validation_flat

  Returns:
    The grouped, ranked evidence per claim
  """
  claim_predictions = {}
  for verify_probs, match_prob, example in zip(verify_predictions,
                                               evidence_predictions,
                                               val_examples):
    struct, label = example
    verify_pred = verify_probs.argmax()
    metadata = json.loads(unwrap_tensor(struct['metadata']))
    claim_id = int(metadata['claim_id'])
    if claim_id not in claim_predictions:
      claim_predictions[claim_id] = []

    claim_predictions[claim_id].append(
        ClaimEvidence(match_prob, verify_pred, metadata,
                      unwrap_tensor(label['evidence_matching']),
                      unwrap_tensor(label['claim_classification'])))
  ranked_evidence = {}
  for claim_id, docs in claim_predictions.items():
    sorted_docs = sorted(docs, key=lambda x: x[0], reverse=True)
    ranked_evidence[claim_id] = sorted_docs

  return ranked_evidence


class FeverMetricsCallback(tf.keras.callbacks.Callback):
  """Callback to compute Fever metrics at end of epoch."""

  def __init__(self,
               *,
               validation_batched,
               fever_dev_path,
               max_evidence,
               max_recall_n = 15,
               checkpoint_dir,
               debug = False):
    """Compute the fever metrics from the batched dataset.

    Args:
      validation_batched: Batched dataset to compute metrics from.
      fever_dev_path: Path to fever dev data
      max_evidence: Max evidence to use
      max_recall_n: Stop computing recall after this position
      checkpoint_dir: If not none, then write validation predictions to disk
        here every epoch
      debug: Whether to enable debug handling of metrics
        since (some fail/error without full data)
    """
    super().__init__()
    self._validation_batched = validation_batched
    self._fever_dev_path = fever_dev_path
    self._max_recall_n = max_recall_n
    self._max_evidence = max_evidence
    self._checkpoint_dir = checkpoint_dir
    self._debug = debug
    self._validation_flat = list(self._validation_batched.unbatch())
    self._dev = util.read_jsonlines(fever_dev_path)
    self._verifiable_dev = [
        claim for claim in self._dev
        if claim['label'] != constants.NOT_ENOUGH_INFO]
    self._verifiable_dev_lookup = {
        claim['id']: claim for claim in self._verifiable_dev}

  def _save_predictions(self, epoch, examples, claim_predictions,
                        evidence_predictions):
    if self._checkpoint_dir is not None:
      out_path = os.path.join(self._checkpoint_dir,
                              f'val_preds_epoch_{epoch}.json')
      predictions = []
      for example, claim_pred, evidence_pred in zip(examples, claim_predictions,
                                                    evidence_predictions):
        # JSON does not work with tf.Tensor or np.ndarray
        claim_pred = unwrap_np(unwrap_tensor(claim_pred))
        evidence_pred = unwrap_np(unwrap_tensor(evidence_pred))
        # The second field is the label, which we don't need
        struct, _ = example
        metadata = json.loads(unwrap_tensor(struct['metadata']))
        predictions.append({
            'claim_pred': claim_pred,
            'evidence_pred': evidence_pred,
            'metadata': metadata,
        })
      with tf.io.gfile.GFile(out_path, 'w') as f:
        json.dump({'predictions': predictions}, f)

  def on_epoch_end(self, epoch, logs=None):
    """At the end of the epoch, trigger computing/saving fever metrics.

    Args:
      epoch: The current epoch
      logs: The logs dictionary containing metrics to read/write
    """
    predictions = self.model.predict(self._validation_batched)
    # This is a list of tensors, one for each output, since keras does not work
    # with dictionaries. The correct places for predictions is set by
    # sorted(['claim_classification', 'evidence_matching']), so the zero
    # index contains claim predictions, the first index evidence predictions
    claim_predictions = predictions['claim_classification']
    evidence_predictions = predictions['evidence_matching']
    self._save_predictions(epoch, self._validation_flat, claim_predictions,
                           evidence_predictions)
    partitioned_claim_preds, partitioned_evidence_preds, partitioned_example = partition_preds_by_scrape_type(
        claim_predictions, evidence_predictions, self._validation_flat)
    for scrape_type in constants.DOC_TYPES:
      scrape_claim_preds = partitioned_claim_preds[scrape_type]
      scrape_evidence_preds = partitioned_evidence_preds[scrape_type]
      scrape_examples = partitioned_example[scrape_type]
      logging.info('scrape_type=%s n_preds=%s', scrape_type,
                   len(scrape_examples))
      ranked_evidence = group_claim_predictions(scrape_claim_preds,
                                                scrape_evidence_preds,
                                                scrape_examples)
      self._record_recall_metrics(
          ranked_evidence=ranked_evidence,
          epoch=epoch,
          logs=logs,
          scrape_type=scrape_type)
      for n_evidence in range(1, 6):
        self._record_fever_metrics(
            ranked_evidence=ranked_evidence,
            epoch=epoch,
            logs=logs,
            scrape_type=scrape_type,
            n_evidence=n_evidence,
        )

  def _debug_metrics(self, logs):
    metrics = {
        'val_fever_strict_score': 0,
        'val_fever_accuracy_score': 0,
        'val_fever_precision': 0,
        'val_fever_recall': 0,
        'val_fever_f1': 0,
    }
    for name, value in metrics.items():
      if logs is not None:
        logs[name] = value

  def _record_fever_metrics(self,
                            *,
                            ranked_evidence,
                            epoch,
                            n_evidence,
                            scrape_type,
                            logs = None):
    """Compute and record fever metrics.

    Args:
      ranked_evidence: Evidence grouped and sorted by claim_id
      epoch: Current epoch
      n_evidence: Number of evidence, if None default to self._max_evidence
      scrape_type: The source scrape type to record in metrics
      logs: logs dictionary to read/write
    """
    if len(ranked_evidence) != len(self._verifiable_dev_lookup):
      logging.warning(
          'Mismatched lengths for gold/pred claims: %s vs %s',
          len(ranked_evidence), len(self._verifiable_dev_lookup))
    if self._debug:
      self._debug_metrics(logs)
      return
    # Compute formatted predictions and gold for fever score function
    formatted_predictions = []
    actual = []
    max_evidence = n_evidence if n_evidence is not None else self._max_evidence
    for claim_id, claim in self._verifiable_dev_lookup.items():
      if claim_id in ranked_evidence:
        prediction = flatten_and_filter_evidence_to_fever(
            claim_id, ranked_evidence[claim_id], max_evidence)
        formatted_predictions.append(prediction)
        actual.append({'evidence': claim['evidence'], 'label': claim['label']})
      else:
        logging.warning(
            'Missing prediction for claim_id=%s type=%s',
            claim_id, type(claim_id))

    # Compute fever metrics
    strict_score, accuracy_score, precision, recall, f1 = fever_score(
        formatted_predictions, actual)
    metrics = {
        f'val_fever_strict_score_{scrape_type}_{max_evidence}':
            strict_score,
        f'val_fever_accuracy_score_{scrape_type}_{max_evidence}':
            accuracy_score,
        f'val_fever_precision_{scrape_type}_{max_evidence}':
            precision,
        f'val_fever_recall_{scrape_type}_{max_evidence}':
            recall,
        f'val_fever_f1_{scrape_type}_{max_evidence}':
            f1
    }
    for name, value in metrics.items():
      logging.info(
          'epoch %s: %s=%s',
          epoch, name, value)
      if logs is not None:
        logs[name] = value  # pytype: disable=container-type-mismatch  # always-use-return-annotations

  def _record_recall_metrics(self,
                             *,
                             ranked_evidence,
                             epoch,
                             scrape_type,
                             logs = None):
    """Compute and record recall metrics.

    Args:
      ranked_evidence: Grouped and sorted evidence
      epoch: Current epoch
      scrape_type: The source scrape type to log metrics with
      logs: logs dictionary
    """
    if self._debug:
      metrics = {
          'val_recall_1': 0,
      }
      for name, value in metrics.items():
        if logs is not None:
          logs[name] = value
      return
    recall_positions = model_recall_by_position(
        self._verifiable_dev, ranked_evidence)
    recall_metrics = {}
    for idx in range(1, self._max_recall_n + 1):
      total = sum(1.0 for n in recall_positions if n <= idx)
      recall_metrics[idx] = total / len(recall_positions)
    for idx in range(1, 11):
      if idx in recall_metrics:
        metric_name = f'val_recall_{scrape_type}_{idx}'
        if logs is not None:
          logs[metric_name] = recall_metrics[idx]
        logging.info(
            'epoch %s: %s=%s',
            epoch, metric_name, recall_metrics[idx])


