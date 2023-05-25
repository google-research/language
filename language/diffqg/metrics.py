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
"""Computes metrics from paired annotations."""
import dataclasses
import enum
import json
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from bleurt import score as bleurt_score
from language.diffqg import annotation
from rouge_score import rouge_scorer
import sentence_transformers


class Label(enum.Enum):
  FALSE_POSITIVE = 1
  FALSE_NEGATIVE = 2
  TRUE_POSITIVE = 3
  TRUE_NEGATIVE = 4

  def is_positive(self) -> bool:
    return self == Label.TRUE_POSITIVE or self == Label.FALSE_NEGATIVE


@dataclasses.dataclass
class Score:
  """Evaluation of a single example pair."""

  # pylint: disable=invalid-name because L is more readable here.
  rouge_L: float
  rouge_1: float
  label: Label
  bleurt: Optional[float] = None
  qsim: Optional[bool] = None

  def to_dict(self):
    """Translates class into dictionary, excluding unspecified values."""
    scores = {
        "rougeL": self.rouge_L,
        "rouge_1": self.rouge_1,
        "label": f"{self.label}",
    }
    if self.qsim is not None:
      scores["qsim"]: self.qsim
    if self.bleurt is not None:
      scores["bleurt"] = self.bleurt
    return scores

  def __str__(self):
    return json.dumps(self.to_dict())


@dataclasses.dataclass
class ScoredPair:
  score: Score
  paired_annotation: annotation.PairedAnnotation

  def to_dict(self):
    return dataclasses.asdict(self.paired_annotation) | self.score.to_dict()

  def __str__(self):
    return json.dumps(self.to_dict())


class QsimModel:
  """Wrapper around huggingface CrossEncoder model."""

  model: sentence_transformers.CrossEncoder
  threshold: float

  def __init__(
      self,
      model_name: str = "cross-encoder/quora-roberta-large",
      threshold: float = 0.5,
  ) -> None:
    self.model = sentence_transformers.CrossEncoder(model_name)
    self.threshold = threshold

  def is_sentence_duplicate(self, sentence1: str, sentence2: str) -> bool:
    return self.are_sentences_duplicate([(sentence1, sentence2)])[0]

  def are_sentences_duplicate(
      self, paired_sentences: List[Tuple[str, str]]
  ) -> List[bool]:
    # Query similarity model does not return True for two empty strings.
    negative_preds = {}
    for i, (sentence1, sentence2) in enumerate(paired_sentences):
      if not sentence1 and not sentence2:
        negative_preds[i] = True
      elif not sentence1 or not sentence2:
        negative_preds[i] = False
    scores = self.model.predict(paired_sentences)
    return [
        score > self.threshold if i not in negative_preds else negative_preds[i]
        for (i, score) in enumerate(scores)
    ]

  def is_paired_anno_duplicate(self, paired_anno: annotation.PairedAnnotation):
    return self.is_sentence_duplicate(
        paired_anno.gold_question, paired_anno.pred_question
    )

  def are_paired_annos_duplicate(
      self, paired_annos: List[annotation.PairedAnnotation]
  ) -> List[bool]:
    paired_sentences = [
        (anno.gold_question, anno.pred_question) for anno in paired_annos
    ]
    return self.are_sentences_duplicate(paired_sentences)


def _normalize_and_get_label(paired_anno: annotation.PairedAnnotation) -> Label:
  """Helper method to compute label of diff detection.

  Args:
    paired_anno: Predicted and gold annotation to compute label of.

  Returns:
    Label of diff detection
  """
  if paired_anno.gold_question is None and paired_anno.pred_question is None:
    label = Label.TRUE_NEGATIVE
    paired_anno.gold_question = ""
    paired_anno.pred_question = ""
  elif paired_anno.pred_question is None:
    label = Label.FALSE_NEGATIVE
    paired_anno.pred_question = ""
  elif paired_anno.gold_question is None:
    label = Label.FALSE_POSITIVE
    paired_anno.gold_question = ""
  else:
    label = Label.TRUE_POSITIVE
  return label


class Scorer:
  """Scores each pair of annotation and prediction.

  rouge_scorer: Library to produce rouge scores. Note that rouge is untrained,
    the scorer object simplify specifies which metrics to use.
  bleurt_scorer: Wrapped BLEURT checkpoint to produce BLEURT scores. Note that
    loading this checkpoint and running this model is expensive.
  qsim_model: Wrapped huggingface RoBERTa cross encoder model trained on Quora
    Question Pairs. Note that loading and running this model is expensive.
  """

  def __init__(
      self,
      bleurt_checkpoint: Optional[str] = None,
      qsim_model: Optional[QsimModel] = None,
  ) -> None:
    self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"])
    self.bleurt_scorer = (
        bleurt_score.BleurtScorer(bleurt_checkpoint)
        if bleurt_checkpoint is not None
        else None
    )
    self.qsim_model = qsim_model

  def score_batch(
      self, paired_annos: List[annotation.PairedAnnotation]
  ) -> Iterable[ScoredPair]:
    """Scores multiple annotations at once, useful for batching model calls.

    Note this method does not handle batching, and the length of the list will
    be used as the batch size.

    Args:
      paired_annos: A single batch of annotations to score.

    Yields:
      ScoredPair objects consisting of the annotation with metrics populated.
        Note that bleurt and qsim might be None if the models were not loaded.
    """
    labels = [_normalize_and_get_label(anno) for anno in paired_annos]
    nones = [None] * len(paired_annos)
    bleurts = (
        self.batch_bleurt(paired_annos)
        if self.bleurt_scorer is not None
        else nones
    )
    qsims = (
        self.qsim_model.are_paired_annos_duplicate(paired_annos)
        if self.qsim_model is not None
        else nones
    )
    for i, paired_anno in enumerate(paired_annos):
      rouge = self.get_rouge_L(paired_anno)
      f1 = self.get_f1(paired_anno)
      score = Score(rouge, f1, labels[i], bleurts[i], qsims[i])
      yield ScoredPair(score, paired_anno)

  def score_pair(self, paired_anno: annotation.PairedAnnotation) -> ScoredPair:
    """Scores paired annotation for standard metrics.

    Note that the query similarity based metrics are not included here.

    Args:
      paired_anno: The paired annotation and prediction to score.

    Returns:
      A ScoredPair object wrapping the example and the scores.
    """
    label = _normalize_and_get_label(paired_anno)
    rouge = self.get_rouge_L(paired_anno)
    bleurt = (
        self.get_bleurt(paired_anno) if self.bleurt_scorer is not None else None
    )
    f1 = self.get_f1(paired_anno)
    qsim = (
        self.qsim_model.is_paired_anno_duplicate(paired_anno)
        if self.qsim_model is not None
        else None
    )
    score = Score(rouge, f1, label, bleurt, qsim)
    return ScoredPair(score, paired_anno)

  # pylint: disable=invalid-name because L is more readable here.
  def get_rouge_L(self, paired_anno: annotation.PairedAnnotation) -> float:
    if not paired_anno.pred_question and not paired_anno.gold_question:
      return 1.0
    elif not paired_anno.pred_question or not paired_anno.gold_question:
      return 0.0
    return self.rouge_scorer.score(
        paired_anno.gold_question, paired_anno.pred_question
    )["rougeL"].fmeasure

  def batch_bleurt(
      self, paired_annos: List[annotation.PairedAnnotation]
  ) -> List[float]:
    """Batches calls to the wrapped BLEURT model for efficiency.

    Note this method does not handle batching and only a single batch should be
    passed.

    Args:
      paired_annos: A single batch of examples.

    Returns:
      A list of scores returned by the BLEURT model, using 0.0 or 1.0 to
        replace empty string matches.
    """
    negative_preds = {}
    for i, paired_anno in enumerate(paired_annos):
      if not paired_anno.pred_question and not paired_anno.gold_question:
        negative_preds[i] = 1.0
      elif not paired_anno.pred_question or not paired_anno.gold_question:
        negative_preds[i] = 0.0
    references = [paired_anno.gold_question for paired_anno in paired_annos]
    candidates = [paired_anno.pred_question for paired_anno in paired_annos]
    bleurt_scores = self.bleurt_scorer.score(
        references=references, candidates=candidates, batch_size=len(references)
    )
    return [
        bleurt_scores[i] if i not in negative_preds else negative_preds[i]
        for (i, score) in enumerate(bleurt_scores)
    ]

  def get_bleurt(self, paired_anno: annotation.PairedAnnotation) -> float:
    if not paired_anno.pred_question and not paired_anno.gold_question:
      return 1.0
    elif not paired_anno.pred_question or not paired_anno.gold_question:
      return 0.0
    return self.bleurt_scorer.score(
        references=[paired_anno.gold_question],
        candidates=[paired_anno.pred_question],
        batch_size=1,
    )[0]

  def get_f1(self, paired_anno: annotation.PairedAnnotation) -> float:
    """Computes token-level overlap of the two questions.

    This is equivalent to Rouge-1's f-measure or the commonly reported
    SQuAD F1 in question answering datasets.

    Args:
      paired_anno: pair of annotation and prediction.

    Returns:
      token-level f1 of the overlap between predicted and annotated.
    """
    gold_set = set(paired_anno.gold_question.split())
    pred_set = set(paired_anno.pred_question.split())
    if not pred_set and not gold_set:
      return 1.0
    elif not pred_set or not gold_set:
      return 0.0
    overlap_words = gold_set & pred_set
    recall = len(overlap_words) / len(gold_set)
    precision = len(overlap_words) / len(pred_set)
    if (summed := recall + precision) == 0.0:
      return 0.0
    f1 = 2 * recall * precision / summed
    return f1


def compute_f1_from_labels(labels: List[Label]) -> Dict[str, float]:
  """Aggregates the metrics to overall change detection metrics.

  Args:
    labels: Label of each example

  Returns:
    Aggregated metrics over the change detection portion of the task.
  """
  true_pos = labels.count(Label.TRUE_POSITIVE)
  false_pos = labels.count(Label.FALSE_POSITIVE)
  false_neg = labels.count(Label.FALSE_NEGATIVE)
  true_neg = labels.count(Label.TRUE_NEGATIVE)
  if not true_pos:
    print("WARNING: No true positive predictions detected!")
    print(f"FP ({false_pos}) FN ({false_neg}) TN ({true_neg})")
    precision, recall, f1 = 0.0, 0.0, 0.0
  else:
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * recall * precision / (recall + precision)
  acc = (true_pos + true_neg) / (false_pos + false_neg + true_pos + true_neg)
  return {
      "label_precision": precision,
      "label_recall": recall,
      "label_f1": f1,
      "label_acc": acc,
  }


def _aggregate_subset_scores(
    subset: List[Score], suffix: str
) -> Dict[str, float]:
  """Computes the average of each metric over the provided subset.

  Note that BLEURT and Query Similarity will be skipped if any of the values are
  None. This should only happen when a checkpoint/path is not provided.

  Args:
    subset: Subset of scores to compute metrics over.
    suffix: Suffix to append to this subset in the final metrics dictionary.

  Returns:
    Dictionary of metric names to metric values.
  """
  metrics = {}
  length = float(len(subset))
  if not length:
    return metrics
  metrics[f"len_{suffix}"] = length
  metrics[f"rouge_{suffix}"] = sum(m.rouge_L for m in subset) / length
  metrics[f"text_f1_{suffix}"] = sum(m.rouge_1 for m in subset) / length
  if not any(m.bleurt is None for m in subset):
    metrics[f"bleurt_{suffix}"] = sum(m.bleurt for m in subset) / length
  if not any(m.qsim is None for m in subset):
    metrics[f"qsim_{suffix}"] = sum(float(m.qsim) for m in subset) / length
  return metrics


def calculate_metrics(
    scored_annotations: List[ScoredPair],
    subsets: Optional[Dict[str, Callable[[ScoredPair], bool]]] = None,
) -> Dict[str, float]:
  """Computes metrics for the scored_annotations for each subset.

  Standard subsets can include all of the data, only the true positives labeled
  by the system, all positive examples, or only the human portion of the data.

  Args:
    scored_annotations: The list of scored annotations to calculate metrics
      over.
    subsets: A dictionary of subset names to filter functions that take a
      ScoredAnnotation and return true or false. If None, will run the single
      set consisting of all of the annotations.

  Returns:
    Dictionary of metric names to metric values, with metric names having the
      subset appended.
  """
  metrics = {}
  if not subsets:
    # Just process the entire set!
    subsets = {"all": lambda _: True}
  for subset_name, subset_filter in subsets.items():
    subset = [anno.score for anno in scored_annotations if subset_filter(anno)]
    metrics.update(_aggregate_subset_scores(subset, subset_name))
  metrics.update(
      compute_f1_from_labels([anno.score.label for anno in scored_annotations])
  )
  return metrics


def score_annotations(
    paired_annotations: Iterable[annotation.PairedAnnotation],
    qsim_model_name: Optional[str] = None,
    bleurt_checkpoint: Optional[str] = None,
    batch_size: int = 1,
    qsim_threshold: float = 0.5,
    num_batches: int = 0,
) -> Iterable[ScoredPair]:
  """Computes scored annotations from the annotation pairs.

  Args:
    paired_annotations: Pairs of gold and predicted annotations.
    qsim_model_name: A huggingface transformers model name. If query similarity
      is desired, should generally use cross-encoder/quora-roberta-large.
      Otherwise, set it to None to skip query similarity computation.
    bleurt_checkpoint: A path to a BLEURT checkpoint directory. Should generally
      use BLEURT-20 (storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip).
    batch_size: Batching for query similarity and BLEURT to speed up
      computation. Has no effect on other metrics or if BLEURT and query
      similarity are both None. Most GPUs probably have to set this to 1.
    qsim_threshold: The threshold to consider a label positive. Keep this at 0.5
      for standard metrics computation, but provided for convenience for
      experimentation.
    num_batches: The number of batches to run. Set low for testing purposes. At
      zero, all batches will be run.

  Yields:
    ScoredPairs, consisting of all desired metrics and the input annotation
      pairs.
  """
  qsim_model = (
      QsimModel(qsim_model_name, qsim_threshold)
      if qsim_model_name is not None
      else None
  )
  scorer = Scorer(bleurt_checkpoint, qsim_model)
  batch = []
  scored_batches = 0
  for paired_anno in paired_annotations:
    batch.append(paired_anno)
    if len(batch) == batch_size:
      yield from scorer.score_batch(batch)
      batch.clear()
      scored_batches += 1
      if num_batches and scored_batches == num_batches:
        return
  if batch:
    yield from scorer.score_batch(batch)
