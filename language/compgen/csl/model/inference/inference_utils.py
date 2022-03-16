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
"""Utils for inference."""

import os

from language.compgen.csl.model.inference import inference_wrapper
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.targets import target_grammar
import tensorflow as tf


def get_checkpoint(wrapper, model_dir, checkpoint):
  """Return checkpoint path and step, or (None, None)."""
  if checkpoint:
    checkpoint = os.path.join(model_dir, checkpoint)
  else:
    checkpoint = tf.train.latest_checkpoint(model_dir)
  step = None
  if checkpoint is not None:
    wrapper.restore_checkpoint(checkpoint)
    step = wrapper.checkpoint.current_step.numpy().item()
  print("Using checkpoint %s at step %s" % (checkpoint, step))
  return checkpoint, step


def get_inference_wrapper(config,
                          rules,
                          target_grammar_file=None,
                          verbose=False):
  """Construct and return InferenceWrapper."""
  rules = qcfg_file.read_rules(rules)
  target_grammar_rules = None
  if target_grammar_file:
    target_grammar_rules = target_grammar.load_rules_from_file(
        target_grammar_file)
  wrapper = inference_wrapper.InferenceWrapper(
      rules, config, target_grammar_rules, verbose=verbose)
  return wrapper
