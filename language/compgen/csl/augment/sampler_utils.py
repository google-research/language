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
"""Utilities for sampling synthetic examples."""

import sys

from language.compgen.csl.augment import joint_sampler
from language.compgen.csl.augment import qcfg_sampler
from language.compgen.csl.common import json_utils
from language.compgen.csl.model.inference import inference_utils
from language.compgen.csl.model.inference import inference_wrapper
from language.compgen.csl.qcfg import qcfg_file
from language.compgen.csl.targets import target_grammar
import numpy as np
from tensorflow.io import gfile


def get_sampler_wrapper(augment_config,
                        model_dir,
                        model_config,
                        rules,
                        target_grammar_file,
                        checkpoint,
                        verbose=False):
  """Construct and return SamplerWrapper."""
  sampler_file = "%s-sampler.json" % rules
  augment_config = json_utils.json_file_to_dict(augment_config)
  rules = qcfg_file.read_rules(rules)
  model_config = (
      json_utils.json_file_to_dict(model_config) if model_dir else None)
  target_grammar_rules = None
  if target_grammar_file:
    target_grammar_rules = target_grammar.load_rules_from_file(
        target_grammar_file)
  return SamplerWrapper(
      augment_config=augment_config,
      model_dir=model_dir,
      model_config=model_config,
      rules=rules,
      target_grammar_rules=target_grammar_rules,
      checkpoint=checkpoint,
      sampler_file=sampler_file,
      verbose=verbose)


def _get_model_score_fn(wrapper):
  scores = np.exp(wrapper.application_scores)

  def score_fn(parent_rule, nt_idx, child_rule):
    rhs_idx = wrapper.rhs_emb_idx_map[(parent_rule, nt_idx)]
    lhs_idx = wrapper.lhs_emb_idx_map[child_rule]
    return scores[lhs_idx, rhs_idx]

  return score_fn


def _get_score_fn(nonterminal_bias, min_nonterminal_rule_arity):

  def score_fn(unused_parent_rule, unused_nt_idx, child_rule):
    if child_rule.arity >= min_nonterminal_rule_arity:
      return 1 + nonterminal_bias
    return 1

  return score_fn


class SamplerWrapper(object):
  """The sampler wrapper that wraps a QCFGSampler or a JointSampler.

  It is pickable as it delays initialization until after the process has been
  forked.
  """

  def __init__(self,
               augment_config,
               model_dir,
               model_config,
               rules,
               target_grammar_rules=None,
               checkpoint=None,
               sampler_file=None,
               verbose=False,
               joint_rules=None):
    self.sampler = None
    self.score_fn = None
    self.augment_config = augment_config
    self.model_dir = model_dir
    self.model_config = model_config
    self.rules = rules
    self.target_grammar_rules = target_grammar_rules
    self.checkpoint = checkpoint
    self.sampler_file = sampler_file
    self.verbose = verbose
    self.joint_rules = joint_rules

  def initialize(self):
    """Initialzie sampler."""

    temperature = self.augment_config.get("temperature", 1)
    nontermminal_bias = self.augment_config.get("nonterminal_bias", 0)
    min_nonterminal_rule_arity = self.augment_config.get(
        "min_nonterminal_rule_arity", 1)

    if self.model_dir:
      wrapper = inference_wrapper.InferenceWrapper(
          self.rules,
          self.model_config,
          self.target_grammar_rules,
          verbose=self.verbose)
      inference_utils.get_checkpoint(wrapper, self.model_dir, self.checkpoint)
      type_lhs_scores = wrapper.model.scoring_layer.type_lhs_scores.numpy()

      # Print out some statistics over model parameters.
      # These are potentially useful for selecting an appropriate range for bias
      # parameters.
      print("max(type_lhs_scores): %s" % np.max(type_lhs_scores))
      print("min(type_lhs_scores): %s" % np.min(type_lhs_scores))
      print("var(type_lhs_scores): %s" % np.var(type_lhs_scores))
      print("mean(type_lhs_scores): %s" % np.mean(type_lhs_scores))

      wrapper.compute_application_scores(
          temperature=temperature,
          nonterminal_bias=nontermminal_bias,
          min_nonterminal_rule_arity=min_nonterminal_rule_arity)
      self.score_fn = _get_model_score_fn(wrapper)
    else:
      self.score_fn = _get_score_fn(
          nonterminal_bias=nontermminal_bias,
          min_nonterminal_rule_arity=min_nonterminal_rule_arity)

    # If a sampler file already exists, load joint rules fom it.
    if self.sampler_file and gfile.exists(self.sampler_file):
      # TODO(pawelnow): Remove this branch and use the more efficient
      # `from_joint_rules`.
      self.sampler = joint_sampler.JointSampler.from_file(
          self.sampler_file,
          max_recursion=self.augment_config["max_recursions"],
          min_recursion=self.augment_config["min_recursions"],
          verbose=self.verbose)
    # If joint rules are passed explicitly, initialize using those.
    elif self.joint_rules:
      self.sampler = joint_sampler.JointSampler.from_joint_rules(
          self.joint_rules,
          max_recursion=self.augment_config["max_recursions"],
          min_recursion=self.augment_config["min_recursions"],
          verbose=self.verbose)
    # Otherwise, if target rules are provided, initialize new joint rules.
    # This step can be expensive.
    elif self.target_grammar_rules:
      self.sampler = joint_sampler.JointSampler.from_rules(
          self.target_grammar_rules,
          self.rules,
          max_recursion=self.augment_config["max_recursions"],
          min_recursion=self.augment_config["min_recursions"],
          max_single_nt_applications=self
          .augment_config["max_single_nt_applications"],
          verbose=self.verbose)
    # Finally, if no target rules are provided, simply sample from QCFG.
    else:
      self.sampler = qcfg_sampler.QCFGSampler(
          self.rules,
          max_recursion=self.augment_config["max_recursions"],
          min_recursion=self.augment_config["min_recursions"])

  def sample_example(self, i):
    """Sample an example."""
    if self.sampler is None:
      self.initialize()
    source_tokens, target_tokens = self.sampler.sample(score_fn=self.score_fn)
    target = " ".join(target_tokens)
    source = " ".join(source_tokens)
    print("Add example %d" % i)
    if self.verbose:
      print("%s\n%s" % (source, target))
    sys.stdout.flush()
    return source, target

  def save(self):
    if self.sampler is None:
      raise ValueError("Sampler is None, please call initialize() first.")
    self.sampler.save(self.sampler_file)
