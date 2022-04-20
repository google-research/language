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
"""Define evaluation metrics for EmQL.
"""

import pickle

from absl import flags
from language.emql.data_loader import DataLoader
import tensorflow.compat.v1.io.gfile as gfile

FLAGS = flags.FLAGS
flags.DEFINE_string('metrics_file', None, 'log file')


class Query2BoxMetrics(object):
  """Query2Box metrics.

  This function computes hits@k and MRR, and printed them to stdout / logs.
  """

  def __init__(self, task, root_dir, data_loader):
    self.task = task
    self.data_loader = data_loader
    # Load answers and hard_answers. Hard answers are defined as
    # answers that can only be inferred from the test kb that are
    # not exposed in train kb. Test kb is a superset of train kb.
    answer_file = root_dir + 'test_ans_%s.pkl' % task
    hard_answer_file = root_dir + 'test_ans_%s_hard.pkl' % task
    self.answers = pickle.load(gfile.GFile(answer_file, 'rb'))
    self.hard_answers = pickle.load(gfile.GFile(hard_answer_file, 'rb'))
    self.q2b_entity2id = pickle.load(
        gfile.GFile(root_dir + 'ent2ind.pkl', 'rb'))
    self.q2b_relation2id = pickle.load(
        gfile.GFile(root_dir + 'rel2ind.pkl', 'rb'))
    # Declare metrics.
    self.total = 0.0
    self.metrics = {'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0, 'mrr': 0.0}

  def convert_entid2q2b(self, entid):
    return self.q2b_entity2id[self.data_loader.id2entity[entid]]

  def convert_relid2q2b(self, relid):
    return self.q2b_relation2id[self.data_loader.id2relation[relid]]

  def eval(self, tf_prediction):
    """Eval one tf_prediction.

    Args:
      tf_prediction: a dictionary of predicted values

    """
    features = tf_prediction['query']
    pred_answers = tf_prediction['answer_ids']

    # Reformat queries into the Query2Box format. There's not a simple
    # way to automatically structure the input. Again, we hard code their
    # input format during parsing for simplicity.
    if self.task == '1c':
      ent, rel = features
      ent = self.convert_entid2q2b(ent)
      rel = self.convert_relid2q2b(rel)
      query = ((ent, (rel,)),)
    elif self.task == '2c':
      ent, rel1, rel2 = features
      ent = self.convert_entid2q2b(ent)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      query = ((ent, (rel1, rel2)),)
    elif self.task == '3c':
      ent, rel1, rel2, rel3 = features
      ent = self.convert_entid2q2b(ent)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      rel3 = self.convert_relid2q2b(rel3)
      query = ((ent, (rel1, rel2, rel3)),)
    elif self.task == '2i' or self.task == '2u':
      ent1, rel1, ent2, rel2 = features
      ent1 = self.convert_entid2q2b(ent1)
      ent2 = self.convert_entid2q2b(ent2)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      query = ((ent1, (rel1,)), (ent2, (rel2,)))
    elif self.task == '3i':
      ent1, rel1, ent2, rel2, ent3, rel3 = features
      ent1 = self.convert_entid2q2b(ent1)
      ent2 = self.convert_entid2q2b(ent2)
      ent3 = self.convert_entid2q2b(ent3)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      rel3 = self.convert_relid2q2b(rel3)
      query = ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (rel3,)))
    elif self.task == 'ic' or self.task == 'uc':
      ent1, rel1, ent2, rel2, rel3 = features
      ent1 = self.convert_entid2q2b(ent1)
      ent2 = self.convert_entid2q2b(ent2)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      rel3 = self.convert_relid2q2b(rel3)
      query = ((ent1, (rel1,)), (ent2, (rel2,)), rel3)
    elif self.task == 'ci':
      ent1, rel1, rel2, ent2, rel3 = features
      ent1 = self.convert_entid2q2b(ent1)
      ent2 = self.convert_entid2q2b(ent2)
      rel1 = self.convert_relid2q2b(rel1)
      rel2 = self.convert_relid2q2b(rel2)
      rel3 = self.convert_relid2q2b(rel3)
      query = ((ent1, (rel1, rel2)), (ent2, (rel3,)))
    else:
      raise ValueError

    all_ans = self.answers[query]
    hard_ans = self.hard_answers[query]
    easy_ans = all_ans - hard_ans
    assert len(easy_ans) == len(all_ans) - len(hard_ans)

    self.update_metrics(pred_answers, easy_ans, hard_ans)

  def update_metrics(self, pred_answers, easy_answers, hard_answers):
    """Compute and update metrics.

    Args:
      pred_answers: a list of predicted answers.
      easy_answers: a set of easy answers, i.e. answers can be inferred from
        train kb.
      hard_answers: a set of hard answers, i.e. answers can only be inferred
        from test kb.
    """
    # Check the correctness of the top 10 "hard" predictions.
    hits = [False] * 10
    i = 0
    mrr = 0.0
    for a in pred_answers:
      a = self.convert_entid2q2b(a)
      if i >= 10: break
      # Exclude easy answers (answers exposed in the training kb).
      if a in easy_answers:
        continue
      hits[i] = (a in hard_answers)
      i += 1
      mrr = max(1.0 / float(i) if a in hard_answers else 0.0, mrr)

    # Add to metrics.
    self.metrics['hits@1'] += float(any(hits[:1]))
    self.metrics['hits@3'] += float(any(hits[:3]))
    self.metrics['hits@10'] += float(any(hits[:10]))
    self.metrics['mrr'] += mrr
    self.total += 1.0

  def print_metrics(self):
    """Print metrics to stdout."""
    print('task: ', self.task)

    # Print to stdout.
    for k, v in self.metrics.items():
      print(k, v / self.total)

    # Print to log files if exist.
    if FLAGS.metrics_file is not None:
      with gfile.GFile(FLAGS.metrics_file, 'w') as f_out:
        f_out.write('task: %s\n' % self.task)
        f_out.write('top_k: %d\n' % FLAGS.intermediate_top_k)
        f_out.write('cm_width: %d\n' % FLAGS.cm_width)
        for k, v in self.metrics.items():
          f_out.write('%s: %f\n' % (k, v / self.total))
