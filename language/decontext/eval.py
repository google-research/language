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
"""Evaluating predictions of decontextualization task.

  It requires "annotation" and "prediction" file, both in jsonl format.

  The format of annotations file can be found in README.md

  Each line for predictions file contains:

  prediction: {example_id: 3245252,
               category: "DONE",
                   // category should be one of DONE, IMPOSSIBLE, UNNECESSARY.
               decontextualized_sentence: "Venus is one of the planets.",
                   // when category is "IMPOSSIBLE", this field is ignored.
               original_sentence: "It is one of the planets."}

"""

from absl import app
from absl import flags
from absl import logging
from language.decontext import decontext_util
from language.decontext import eval_util
import nltk

nltk.download("punkt")

FLAGS = flags.FLAGS

flags.DEFINE_string("annotations", None,
                    "Path to a annotations file, where each line is a JSON.")
flags.DEFINE_string("predictions", None,
                    "Path to a predictions file, where each line is a JSON.")


def main(argv):
  del argv
  annotation_dict = decontext_util.load_annotations(FLAGS.annotations)
  logging.info("%d examples in annotation.", len(annotation_dict))

  prediction_dict = decontext_util.load_predictions(FLAGS.predictions)
  pred_sent_dict = eval_util.get_sent_dict(prediction_dict)
  logging.info("%d examples in predicton.", len(prediction_dict))

  # Evaluates generated decontextualized sentence.
  logging.info("== Prediction score ==")
  (original_sent_dict, reference_sents_dict,
   median_human_sent_dict) = eval_util.process_annotation(annotation_dict)
  eval_util.compute_sentence_generation_scores(original_sent_dict,
                                               reference_sents_dict,
                                               pred_sent_dict)
  logging.info("== Reference human score ==")
  eval_util.compute_sentence_generation_scores(original_sent_dict,
                                               reference_sents_dict,
                                               median_human_sent_dict)

  # Evaluates decontextualization category classification

  logging.info("== Feasibility classification score ==")
  eval_util.score_classification(annotation_dict, prediction_dict)


if __name__ == "__main__":
  flags.mark_flag_as_required("annotations")
  flags.mark_flag_as_required("predictions")
  app.run(main)
