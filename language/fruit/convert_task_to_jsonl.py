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
r"""Convert a seqio task into jsonl format.

    This is a script that extracts the seqio task into jsonl format so
    later it can be used for inference and then latter call the
    evaluation script. Note that if you use t5x, the evaluation is
    builtin. This script is designed more for users who did not
    use the t5x package for training and evaluation.

    ** Input Splits **

    The typical use case will consider only three possible combinations.

    To output the validation set (from "Nov. 20, 2019" and "Nov. 20, 2020")

    --task_name="wikidiff_diff_all_text_reference" --split="validation"

    To output the test set (from "Nov. 20, 2020" and "Nov. 20, 2021").

    --task_name="wikidiff_diff_all_text_reference_test" --split="test"

    To print out the gold test (from "Nov. 20, 2020" and "Nov. 20, 2021",
       verified by annotators)

    --task_name="wikidiff_diff_all_text_reference_gold_test" --split="test"

    ** Output **

    This script will be used for generating two jsonl files.

    {prefix}_inputonly.jsonl

        The input only files will contain the input of Fruit5 for the
        chosen split. This file is used as input for genearting the prediction
        using your models.

    {prefix}_inputlabels.jsonl

        The input and labels file contains both the input and the
        targeted output. This one will later be in the evaluation script.

    ** Eval **

    Please see the info in the eval file (scripts/evaluate_direct_jsonls.py
) for more details.

"""
import ast
import functools
import json


from absl import app
from absl import flags
from language.fruit import rendering_utils
from language.fruit import tf_utils
import language.fruit.tasks  # pylint: disable=unused-import
import seqio
import t5.data
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "task_name", None,
    "the name of the task that is going to be converted into the jsonl format")
flags.DEFINE_string("split", None,
                    "the split that should be used for the conversion.")
flags.DEFINE_string(
    "sequence_length", "{'inputs': 1024, 'targets': 512}",
    "String representation of a dictionary for sequence length to be passed to Seqio `get_dataset`."
)
flags.DEFINE_string("output_prefix", None,
                    "The output name prefix for output the two jsonl files.")


def convert_to_jsonl(task, split):
  """Convert to jsonl from reading the task."""

  delimiter_range_pair = rendering_utils.get_default_delimiter_range_pair(
      task,
      rendering_utils.DelimiterType.text,
  )

  normalize_fn = functools.partial(
      rendering_utils.normalize,
      delimiter_range_pair=delimiter_range_pair,
      task=task,
  )

  print(f"***** {task} *****")
  vocab = t5.data.get_default_vocabulary()

  sequence_length = ast.literal_eval(FLAGS.sequence_length)
  dataset = task.get_dataset(
      sequence_length=sequence_length,
      split=split,
      use_cached=False,
      shuffle=False,
      num_epochs=1,
  )

  print(f"*** {split} ***")
  print("* features *")

  count = 0
  step = 500

  with tf.io.gfile.GFile(FLAGS.output_prefix + "_inputonly.jsonl",
                         "w") as inputonly_f:
    with tf.io.gfile.GFile(FLAGS.output_prefix + "_inputlabels.jsonl",
                           "w") as inputlabels_f:

      for raw_example in dataset:
        count += 1
        if count % step == 0:
          print(f"processed {count} examples..")
        inputs = tf_utils.maybe_decode(
            vocab.decode_tf(raw_example["inputs"]).numpy())
        targets = tf_utils.maybe_decode(
            vocab.decode_tf(raw_example["targets"]).numpy())

        normalized_inputs, normalized_targets = normalize_fn(inputs, targets)

        input_only_example = {
            "inputs": inputs,
            "normalized_inputs": normalized_inputs,
        }

        inputonly_f.write(json.dumps(input_only_example) + "\n")

        example = {
            "inputs": inputs,
            "targets": targets,
            "normalized_inputs": normalized_inputs,
            "normalized_targets": normalized_targets,
        }

        inputlabels_f.write(json.dumps(example) + "\n")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  all_task_names = list(seqio.TaskRegistry.names())

  assert FLAGS.task_name in all_task_names, "unknown task name!"

  seqio_task = seqio.get_mixture_or_task(FLAGS.task_name)

  print("Available splits are:" + " ".join(seqio_task.splits))
  assert FLAGS.split in seqio_task.splits, "unavailable split in the task!"

  convert_to_jsonl(seqio_task, FLAGS.split)


if __name__ == "__main__":
  app.run(main)
