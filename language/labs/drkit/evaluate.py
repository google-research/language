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
"""Evaluate lazy slot filling results."""

import codecs
import collections
import gzip
import json
import random
import re
import string
import unicodedata

from absl import app
from absl import flags
from bert import tokenization
from language.labs.drkit import input_fns
import numpy as np
import tensorflow.compat.v1 as tf

PUNCTUATION = frozenset(string.punctuation)

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("ground_truth_file", None,
                    "File with ground truth answers.")

flags.DEFINE_string("predicted_answers_file", None,
                    "File with predicted answers from model.")

flags.DEFINE_string("relation_counts_file", None,
                    "JSON file with relation counts.")


class NumpyEncoder(json.JSONEncoder):
  """Special json encoder for numpy types."""

  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # This is the fix
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def wikimovie_eval_fn(dataset, results, name_map, output_prediction_file,
                      **kwargs):
  """Compute evaluation metrics for OneHopDataset or TwoHopDataset.

  Args:
    dataset: An object of type OneHopDataset.
    results: A list of result dicts from running estimator.predict.
    name_map: A mapping from prediction indices to text strings.
    output_prediction_file: File to store predictions to.
    **kwargs: Variable keyword arguments.

  Returns:
    metrics: A dict mapping metric names to values.
  """
  del kwargs

  # Collect ground truth answers.
  gt_answer = {ex.qas_id: ex.answer_entity for ex in dataset.examples}
  gt_ques = {ex.qas_id: ex.question_text for ex in dataset.examples}
  gt_entity = {ex.qas_id: ex.subject_entity[0] for ex in dataset.examples}
  inf_chain = {ex.qas_id: ex.inference_chain for ex in dataset.examples}

  # Compute basic metrics.
  num_correct = 0.
  all_predictions = {}
  chain2stats = {ch: [0., 0.] for ch in inf_chain.values()}
  incorrect_results, correct_results = [], []
  for result in results:
    qas_id = result["qas_ids"]
    prediction = result["predictions"]
    if prediction in gt_answer[qas_id]:
      num_correct += 1
      chain2stats[inf_chain[qas_id]][0] += 1
      correct_results.append({
          "qas_id": result["qas_ids"],
          "question": gt_ques[qas_id],
          "answers": gt_answer[qas_id],
          "subject": gt_entity[qas_id],
          "inf-chain": inf_chain[qas_id],
          "predictions": result["predictions"],
      })
      for hop in range(3):
        if "sparse_%d" % hop in result:
          correct_results[-1].update({
              "sparse_%d" % hop: result["sparse_%d" % hop],
              "dense_%d" % hop: result["dense_%d" % hop],
              "mention_%d" % hop: result["mention_%d" % hop],
              "entity_%d" % hop: result["entity_%d" % hop],
              "sparse_scores_%d" % hop: result["sparse_scores_%d" % hop],
              "dense_scores_%d" % hop: result["dense_scores_%d" % hop],
              "mention_scores_%d" % hop: result["mention_scores_%d" % hop],
              "entity_scores_%d" % hop: result["entity_scores_%d" % hop],
          })
    else:
      incorrect_results.append({
          "qas_id": result["qas_ids"],
          "question": gt_ques[qas_id],
          "answers": gt_answer[qas_id],
          "subject": gt_entity[qas_id],
          "inf-chain": inf_chain[qas_id],
          "predictions": result["predictions"],
      })
      for hop in range(3):
        if "sparse_%d" % hop in result:
          incorrect_results[-1].update({
              "sparse_%d" % hop: result["sparse_%d" % hop],
              "dense_%d" % hop: result["dense_%d" % hop],
              "mention_%d" % hop: result["mention_%d" % hop],
              "entity_%d" % hop: result["entity_%d" % hop],
              "sparse_scores_%d" % hop: result["sparse_scores_%d" % hop],
              "dense_scores_%d" % hop: result["dense_scores_%d" % hop],
              "mention_scores_%d" % hop: result["mention_scores_%d" % hop],
              "entity_scores_%d" % hop: result["entity_scores_%d" % hop],
          })
    chain2stats[inf_chain[qas_id]][1] += 1
    all_predictions[qas_id] = name_map[str(prediction)]
  accuracy = num_correct / len(all_predictions)
  json.dump(all_predictions, tf.gfile.Open(output_prediction_file, "w"))
  json.dump(
      random.sample(incorrect_results, 100),
      tf.gfile.Open(output_prediction_file + ".incorrect", "w"),
      cls=NumpyEncoder)
  json.dump(
      random.sample(correct_results, 100),
      tf.gfile.Open(output_prediction_file + ".correct", "w"),
      cls=NumpyEncoder)

  # Return metrics.
  metrics = {
      "accuracy": accuracy,
  }
  for ch, stats in chain2stats.items():
    metrics["inference-chains-acc/" + ch] = stats[0] / stats[1]
  return metrics


def multihop_eval_fn(dataset,
                     results,
                     name_map,
                     output_prediction_file,
                     supervision="mention",
                     **kwargs):
  """Compute evaluation metrics for OneHopDataset or TwoHopDataset.

  Args:
    dataset: An object of type OneHopDataset.
    results: A list of result dicts from running estimator.predict.
    name_map: A mapping from prediction indices to text strings.
    output_prediction_file: File to store predictions to.
    supervision: Type of supervision used in the model.
    **kwargs: Variable keyword arguments.

  Returns:
    metrics: A dict mapping metric names to values.
  """
  del kwargs

  # Collect ground truth answers.
  gt_mentions = {ex.qas_id: ex.answer_mention[0] for ex in dataset.examples}
  if supervision == "mention":
    gt_answer = gt_mentions
  else:
    gt_answer = {ex.qas_id: ex.answer_entity[0] for ex in dataset.examples}

  # Compute basic metrics.
  num_correct = 0.
  all_predictions = {}
  for result in results:
    qas_id = result["qas_ids"]
    prediction = result["predictions"]
    if prediction == gt_answer[qas_id]:
      num_correct += 1
    all_predictions[qas_id] = name_map[str(prediction)]
  accuracy = num_correct / len(all_predictions)

  # Compute advanced metrics.
  json.dump(all_predictions, tf.gfile.Open(output_prediction_file, "w"))
  micro, macro, _, _ = compute_scores(dataset.gt_file, output_prediction_file)

  # Return metrics.
  metrics = {
      "accuracy": accuracy,
      "micro-p": micro[0],
      "micro-r": micro[1],
      "micro-f": micro[2],
      "macro-p": macro[0],
      "macro-r": macro[1],
      "macro-f": macro[2],
  }
  return metrics


def hotpot_eval_fn(dataset, results, name_map, output_prediction_file,
                   **kwargs):
  """Compute evaluation metrics for HotpotQADataset.

  Args:
    dataset: An object of type HotpotQADataset.
    results: A list of result dicts from running estimator.predict.
    name_map: A mapping from prediction indices to text strings.
    output_prediction_file: File to store predictions to.
    **kwargs: Variable keyword arguments.

  Returns:
    metrics: A dict mapping metric names to values.
  """
  del kwargs

  # Collect ground truth answers.
  gt_answer = {ex.qas_id: ex.answer_entity for ex in dataset.examples}
  gt_types = {ex.qas_id: ex.inference_chain for ex in dataset.examples}

  # Compute basic metrics.
  num_correct = {2: 0., 5: 0., 10: 0., 20: 0.}
  aps = []
  no_answer = 0.
  all_predictions = {}
  bridge_acc, comp_acc = 0., 0.
  bridge_tot, comp_tot = 0, 0
  single_acc = 0.
  layer_weights = np.zeros_like(results[0]["layer_probs"])
  num_layer_entities = {i: 0. for i in range(layer_weights.shape[0])}
  num_new_entities = {i: 0. for i in range(layer_weights.shape[0])}
  for result in results:
    qas_id = result["qas_ids"].decode("utf-8")
    preds = result["top_idx"]
    scores = result["top_vals"]
    ans = gt_answer[qas_id]
    my_type = gt_types[qas_id]
    if my_type == "bridge":
      bridge_tot += 1
    else:
      comp_tot += 1
    ranks = np.where(np.in1d(preds, ans))[0]
    ranks = np.sort(ranks)
    ap = 0.
    cnt = 0.
    if any(rr < 10 for rr in ranks):
      single_acc += 1
    if ranks.shape[0] == 0:
      no_answer += 1
    for rr in ranks:
      cnt += 1
      ap += cnt / (rr + 1)
    if ans:
      aps.append(ap / len(ans))
    else:
      aps.append(0.)
    found = False
    for key in [2, 5, 10, 20]:
      if found or np.in1d(ans, preds[:key]).all():
        num_correct[key] += 1
        found = True
        if key == 10:
          if my_type == "bridge":
            bridge_acc += 1
          else:
            comp_acc += 1
    # Non-accuracy stats
    layer_weights += result["layer_probs"]
    layer_entities = {i: set() for i in range(layer_weights.shape[0])}
    all_predictions[qas_id] = {}
    for i in range(layer_weights.shape[0]):
      layer_entities[i] = set(
          [ee for ee in result["layer_%d_ent" % i] if ee != -1])
      num_layer_entities[i] += len(layer_entities[i])
      num_new_entities[i] += len(layer_entities[i] - layer_entities[0])
      # all_predictions[qas_id]["layer_%d" % i] = [
      #     name_map[str(ee)] for ee in layer_entities[i]]
    all_predictions[qas_id]["predictions"] = [
        (name_map[str(pred)], str(scores[i])) for i, pred in enumerate(preds)
    ]
  tf.logging.info("Evaluated %d items", len(all_predictions))
  accuracy = {
      key: (num_correct[key] / len(all_predictions)) for key in num_correct
  }

  # Compute advanced metrics.
  json.dump(all_predictions, tf.gfile.Open(output_prediction_file, "w"))

  # Return metrics.
  metrics = {"eval/@%d" % key: accuracy[key] for key in accuracy}
  metrics["accuracy"] = accuracy[10]
  metrics["eval/map"] = sum(aps) / len(all_predictions)
  metrics["eval/bridge_accuracy"] = bridge_acc / bridge_tot
  metrics["eval/comparison_accuracy"] = comp_acc / comp_tot
  metrics["analysis/single_accuracy"] = single_acc / len(all_predictions)
  metrics["analysis/no_answers"] = no_answer / len(all_predictions)
  for i in range(layer_weights.shape[0]):
    metrics["analysis/layer_weight_%d" %
            i] = layer_weights[i] / len(all_predictions)
    metrics["analysis/num_entities_%d" %
            i] = num_layer_entities[i] / len(all_predictions)
    metrics["analysis/num_new_entities_%d" %
            i] = num_new_entities[i] / len(all_predictions)

  return metrics


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  """Compute F1 score."""
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def exact_match_score(prediction, ground_truth):
  """Compute EM score."""
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    my_score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(my_score)
  return max(scores_for_ground_truths)


def read_predictions(prediction_file):
  with tf.gfile.Open(prediction_file) as f:
    predictions = json.load(f)
  return predictions


def read_answers(gold_file):
  """Read ground truth answers."""
  answers = {}
  f = tf.gfile.Open(gold_file)
  if gold_file.endswith(".gz"):
    f = gzip.GzipFile(fileobj=f)
  for i, line in enumerate(f):
    example = json.loads(line)
    if i == 0 and "header" in example:
      continue
    for qa in example["qas"]:
      answers[qa["qid"]] = qa["answers"]
  f.close()
  return answers


def evaluate(answers, predictions, skip_no_answer=False):
  """Compute F1 and EM scores."""
  f1 = exact_match = total = 0
  for qid, ground_truths in answers.items():
    if qid not in predictions:
      if not skip_no_answer:
        message = "Unanswered question %s will receive score 0." % qid
        print(message)
        total += 1
      continue
    total += 1
    prediction = predictions[qid]
    exact_match += metric_max_over_ground_truths(exact_match_score, prediction,
                                                 ground_truths)
    f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return {"exact_match": exact_match, "f1": f1}


def mrqa_eval_fn(dataset_file, predictions_file, skip_no_answer=True):
  answers = read_answers(dataset_file)
  predictions = read_predictions(predictions_file)
  return evaluate(answers, predictions, skip_no_answer)


def compute_scores(ground_truth_file, predicted_answers_file):
  """Read predictions and ground truth and return P, R, F."""
  telemetry, incorrect = read_results(ground_truth_file, predicted_answers_file)
  micro = aprf(telemetry)
  relationwise = aprf_relationwise(telemetry)
  macro = sum([val[0] for _, val in relationwise.items()])
  macro = macro / len(relationwise)
  return micro, macro, relationwise, incorrect


def read_results(ground_truth_file, predicted_answers_file):
  """Read results and ground truth and return data structure with stats."""
  with codecs.getreader("utf-8")(tf.gfile.GFile(ground_truth_file,
                                                "r")) as read:
    data_ = {}
    for line in read:
      item = json.loads(line.strip())
      if isinstance(item["relation"], dict):
        relation = item["relation"]["wikidata_id"]
      elif isinstance(item["relation"], list):
        relation = (
            item["relation"][0]["wikidata_id"] + "_" +
            item["relation"][1]["wikidata_id"])
      data_[item["id"]] = [relation, item["subject"]["wikidata_id"]]
      if "is_impossible" in item and item["is_impossible"]:
        continue
      if item["object"] is None:
        continue
      if isinstance(item["object"]["mention"], dict):
        data_[item["id"]] += [item["object"]["mention"]["text"]]
      if "name" in item["object"]:
        data_[item["id"]] += [item["object"]["name"]]
      if "aliases" in item["object"]:
        data_[item["id"]] += item["object"]["aliases"].keys()
  with codecs.getreader("utf-8")(tf.gfile.GFile(predicted_answers_file,
                                                "r")) as fin:
    predictions = json.load(fin)

    telemetry, incorrect = [], []
    n = 0
    for key in data_:
      if key not in predictions:
        continue
      g = data_[key][2:]
      a = predictions[key]
      m = data_[key][:2]
      stats = score(g, a)
      telemetry.append([m[0], m[1], g, a, stats])
      if stats[0] == 0. and stats[3] > 0.:
        incorrect.append(key)
      n += 1
    return telemetry, incorrect


def aprf_relationwise(g):
  """Returns precision, recall and F score for each relation."""
  rel_to_stats = collections.defaultdict(list)
  for item in g:
    rel_to_stats[item[0]].append(item)
  rel_to_scores = {}
  for rel, stats in rel_to_stats.items():
    rel_to_scores[rel] = [aprf(stats), len(stats)]
  return rel_to_scores


def aprf(g):
  """Returns precision, recall and F of the given statistics."""
  tp, _, sys_pos, real_pos = sum([x[-1] for x in g])
  if tp == 0:
    p = r = f = 0.0
  else:
    p = tp / float(sys_pos) if sys_pos > 0 else 0.
    r = tp / float(real_pos) if real_pos > 0 else 0.
    f = 2 * p * r / (p + r)
  return np.asarray([p, r, f])


def score(gold, answer):
  """Compares answer to ground truth to return TP / FP stats."""
  if gold:
    gold = set([simplify(g) for g in gold])
  answer = simplify(answer)
  result = np.zeros(4)
  if gold:
    result[3] += 1
    if answer in gold:
      result[0] += 1
  else:
    if not answer:
      result[1] += 1
  if answer:
    result[2] += 1
  return result


def strip_accents_and_punct(text):
  """Strips accents from a piece of text."""
  text = unicodedata.normalize("NFD", text)
  output = []
  for char in text:
    if char in PUNCTUATION:
      continue
    cat = unicodedata.category(char)
    if cat == "Mn":
      continue
    output.append(char)
  return "".join(output)


def simplify(answer):
  """Pre-process answer string."""
  toks = []
  articles = {"the", "a", "an", "and", ""}
  for t in answer.strip().lower().split():
    tok = strip_accents_and_punct(t)
    if tok not in articles:
      toks.append(tok)
  return "".join(toks)


def rare_relation_scores(relationwise, relation2counts):
  """Print statistics of rare relations for different thresholds."""
  for thresh in [5, 100, 500, 1000]:
    freq_stats, freq_total = np.array([0., 0., 0.]), 0
    rare_stats, rare_total = np.array([0., 0., 0.]), 0
    for relation, (stats, _) in relationwise.items():
      if relation2counts.get(relation, 0) < thresh:
        rare_stats += stats
        rare_total += 1
      else:
        freq_stats += stats
        freq_total += 1
    rare_stats /= rare_total
    freq_stats /= freq_total
    print(
        "Threshold =", thresh, "rare", rare_total,
        "Micro-P %.3f Micro-R %.3f Micro-F %.3f" %
        (rare_stats[0], rare_stats[1], rare_stats[2]), "freq", freq_total,
        "Micro-P %.3f Micro-R %.3f Micro-F %.3f" %
        (freq_stats[0], freq_stats[1], freq_stats[2]))




def main(_):
  eval_type = "hotpot"
  if eval_type == "hotpot":
    test_hotpot_eval()
  else:
    micro, macro, rwise, _ = compute_scores(FLAGS.ground_truth_file,
                                            FLAGS.predicted_answers_file)
    print("Micro", micro)
    print("Macro", macro)
    if FLAGS.relation_counts_file is not None:
      r2c = json.load(tf.gfile.Open(FLAGS.relation_counts_file))
      rare_relation_scores(rwise, r2c)


if __name__ == "__main__":
  app.run(main)
