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
r"""Train and eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

# This import triggers the decorations that register the problems.
import language.labs.consistent_zero_shot_nmt.data_generators.translate_europarl  # pylint: disable=unused-import
import language.labs.consistent_zero_shot_nmt.data_generators.translate_iwslt17  # pylint: disable=unused-import
import language.labs.consistent_zero_shot_nmt.data_generators.translate_uncorpus  # pylint: disable=unused-import
import language.labs.consistent_zero_shot_nmt.models.agreement  # pylint: disable=unused-import
import language.labs.consistent_zero_shot_nmt.models.basic  # pylint: disable=unused-import
import language.labs.consistent_zero_shot_nmt.utils.t2t_tweaks  # pylint: disable=unused-import

from tensor2tensor.bin import t2t_trainer
import tensorflow as tf

FLAGS = flags.FLAGS


def main(argv):

  if getattr(FLAGS, "brain_jobs", None):
    FLAGS.worker_job = "/job:%s" % FLAGS.brain_job_name

  return t2t_trainer.main(argv)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  tf.app.run()
