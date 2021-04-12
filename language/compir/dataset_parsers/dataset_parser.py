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
"""Interface for transforming data to reversible and lossy transformations."""


import dataclasses


@dataclasses.dataclass
class Example:
  utterance: Text
  program: Text


class DatasetParserInterface():
  """Interface for transforming data to reversible and lossy transformations."""

  def __init__(self, train_examples_raw,
               test_examples_raw):
    self.train_examples = self.preprocess(train_examples_raw)
    self.test_examples = self.preprocess(test_examples_raw)

  def f_reversible(self, example):
    """Transforms a single program to its reversible IR."""
    raise NotImplementedError

  def f_lossy(self, program, is_rir):
    """Transforms a single program to its lossy IR."""
    raise NotImplementedError

  def f_reversible_inverse(self, program):
    """Inverts a program from its reversibe IR to its original representation."""
    raise NotImplementedError

  def preprocess_program(self, program):
    """Appiles preprocessing for a single program."""
    raise NotImplementedError

  def postprocess_program(self, program):
    """Postprocesses a single predicted program."""
    raise NotImplementedError

  def preprocess(self, examples):
    """Appiles preprocessing for programs in all examples."""
    return [
        Example(example.utterance, self.preprocess_program(example.program))
        for example in examples
    ]

  def postprocess_full(self, predictions,
                       is_rir):
    """Inverts predicted reversible IRs to programs and postprocesses them."""
    predictions_postprocessed = []
    for prediction in predictions:
      try:
        if is_rir:
          prediction_orig = self.f_reversible_inverse(prediction)
        else:
          prediction_orig = prediction
        prediction_postprocessed = self.postprocess_program(prediction_orig)
      except ValueError:
        # Postprocessing can fail when the prediction has a wrong syntax.
        print("Wrong prediction syntax : {}".format(prediction))
        prediction_postprocessed = None
      predictions_postprocessed.append(prediction_postprocessed)
    return predictions_postprocessed

  def no_transformation(self):
    """Applies no transformation by returning the raw data."""
    return self.train_examples, self.test_examples

  def to_reversible(self):
    """Transforms all examples to their reversible intermediate representation."""

    def get_rir_examples(examples):
      return [
          Example(example.utterance, self.f_reversible(example))
          for example in examples
      ]

    return get_rir_examples(self.train_examples), get_rir_examples(
        self.test_examples)

  def to_lossy(
      self,
      is_rir = False,
      train_examples = None,
      test_examples = None
  ):
    """Transforms examples to their lossy (and maybe also reversible) IR.

    Args:
      is_rir: if True, the input is a reversible IR, in comparison to a program
        in the original formalism.
      train_examples: to be supplied incase is_rir=True, and holds examples in
        their reversible IR.
      test_examples: to be supplied incase is_rir=True, and holds examples in
        their reversible IR.

    Returns:
      A tuple of train and test examples in their lossy IR (and maybe also
      reversible IR).
    """

    def get_lir_examples(examples):
      return [
          Example(example.utterance,
                  self.f_lossy(example.program, is_rir=is_rir))
          for example in examples
      ]

    if not train_examples or not test_examples:
      train_examples, test_examples = self.train_examples, self.test_examples
    return get_lir_examples(train_examples), get_lir_examples(test_examples)

  def program_to_lossy(self):
    """Transforms programs to a lossy IR."""
    return self.to_lossy()

  def rir_to_lossy(self):
    """Transforms programs to both a lossy and reversible IR."""
    return self.to_lossy(True, *self.to_reversible())

  def from_lossy(
      self,
      predictions,
      is_rir,
      is_direct,
      train_examples = None,
      test_examples = None
  ):
    """Prepares data to recover original programs from lossy IRs by seq2seq_2.

    Args:
      predictions: test set predictions, to be concatenated to the utterance as
        input to seq2seq_2. These are expected to be: (1) lossy IRs if
          is_direct=True and is_rir=False, (2) lossy+reversible IRs if
          is_direct=True and is_rir=True, (3) reversible IRs if is_direct=False
          and is_rir=True or (4) programs in the original formalism if
          is_direct=False and is_rir=False.
      is_rir: whether the representation from which to predict programs has gone
        through the reversible transformation.
      is_direct: If True, test predictions are expected to be lossy.
      train_examples: to be supplied incase is_rir=True, and holds examples in
        their reversible IR.
      test_examples: to be supplied incase is_rir=True, and holds examples in
        their reversible IR.

    Returns:
      A tuple of train and test examples used to train seq2seq_2 for predicting
      programs given the utterance and the lossy IR.
    """

    def concat_lirs(examples, lirs):
      """Concatenates the lossy IR to the utterance."""
      examples_concatenated = []
      for example, lir in zip(examples, lirs):
        source = "{} coarse: {}".format(example.utterance, lir)
        target = example.program
        examples_concatenated.append(Example(source, target))
      return examples_concatenated

    if not train_examples or not test_examples:
      train_examples, test_examples = self.train_examples, self.test_examples
    # Use gold lossy IRs for the training set.
    train_examples_lossy, _ = self.to_lossy(
        is_rir=is_rir,
        train_examples=train_examples,
        test_examples=test_examples)
    if is_direct:
      prediction_lirs = predictions
    # For the indirect approach, predictions are not in their lossy IR, so
    # the lossy transformation is applied.
    else:
      prediction_lirs = [
          self.f_lossy(prediction, is_rir) for prediction in predictions
      ]

    train_examples_concatenated = concat_lirs(
        train_examples, [example.program for example in train_examples_lossy])
    test_examples_concatenated = concat_lirs(test_examples, prediction_lirs)
    return train_examples_concatenated, test_examples_concatenated

  def program_from_lossy(
      self,
      predictions,
      is_direct,
  ):
    """Prepares data when the input is only lossy (and not reversible)."""
    return self.from_lossy(predictions, False, is_direct)

  def rir_from_lossy(
      self,
      predictions,
      is_direct,
  ):
    """Prepares data when the input is both lossy and reversible."""
    return self.from_lossy(predictions, True, is_direct, *self.to_reversible())
