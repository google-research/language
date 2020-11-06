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
"""Compute QA scores for a saved model's predictions.

To get the predictions, see infer_captions.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from language.capwap.utils import experiment_utils
from language.capwap.utils import io_utils
from language.capwap.utils import metric_utils
from language.capwap.utils import text_utils
import tensorflow.compat.v1 as tf

DATA_DIR = os.getenv("QA2CAPTION_DATA", "data")

flags.DEFINE_string("caption_file", None, "Direct prediction file.")

flags.DEFINE_string("question_file", None, "QA evaluation file.")

flags.DEFINE_string("coco_annotations",
                    os.path.join(DATA_DIR, "COCO/annotations"),
                    "Path to raw annotations for COCO (i.e., the OOD data).")

flags.DEFINE_bool("eval_qa", True, "Do the QA evaluation.")

flags.DEFINE_bool("eval_reference", False,
                  "Measure reference metrics (w.r.t COCO)")

flags.DEFINE_string("output_file", None, "Path to write results to.")

flags.DEFINE_boolean("average", False, "Average beams.")

flags.DEFINE_float("no_answer_bias", -1e4, "No answer bias.")

flags.DEFINE_string("rc_model", os.path.join(DATA_DIR, "rc_model"),
                    "TF Hub handle for BERT QA model.")

flags.DEFINE_string("vocab_path", os.path.join(DATA_DIR, "uncased_vocab.txt"),
                    "Path to BERT directory.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info("***** Running QA evaluation *****")

  # Load vocab
  vocab = text_utils.Vocab.load(FLAGS.vocab_path)

  # If not averaging whatever is in the top-k predictions, we rewrite
  # the caption file just to be sure.
  if not FLAGS.average:
    tmpfile = experiment_utils.get_tempfile()
    with tf.io.gfile.GFile(tmpfile, "w") as f_out:
      with tf.io.gfile.GFile(FLAGS.caption_file, "r") as f_in:
        for line in f_in:
          entry = json.loads(line)
          entry["token_ids"] = entry["token_ids"][:1]
          f_out.write(json.dumps(entry) + "\n")
    FLAGS.caption_file = tmpfile

  # Get results.
  results = {}

  # Evaluate QA metrics (the main eval).
  if FLAGS.eval_qa:
    if not FLAGS.question_file:
      raise ValueError("Must provide a question-answer file.")
    qa_metrics = metric_utils.evaluate_questions(
        caption_file=FLAGS.caption_file,
        vocab=vocab,
        question_file=FLAGS.question_file,
        params=dict(
            rc_model=FLAGS.rc_model,
            question_length=64,
            context_length=64,
            num_input_threads=FLAGS.num_input_threads,
            no_answer_bias=FLAGS.no_answer_bias,
            use_tpu=FLAGS.use_tpu,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            prefetch_batches=1000))
    results.update(qa_metrics)

  # If this is a base OOD model (on COCO), evaluate useful
  # metrics such as reference BLEU/CIDEr/ROUGE.
  if FLAGS.eval_reference:
    if not FLAGS.coco_annotations:
      raise ValueError("Must provide path to COCO annotations.")
    coco_metrics = metric_utils.evaluate_captions(
        caption_file=FLAGS.caption_file,
        coco_annotations=FLAGS.coco_annotations,
        vocab=vocab)
    results.update(coco_metrics)

  # Write outputs.
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_file))
  with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
    json.dump(results, f, cls=io_utils.NumpyEncoder, indent=2, sort_keys=True)
  tf.logging.info(results)


if __name__ == "__main__":
  flags.mark_flag_as_required("caption_file")
  flags.mark_flag_as_required("output_file")
  tf.disable_v2_behavior()
  app.run(main)
