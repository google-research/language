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
"""Calculate data impressions via discrete-level operations on some objective.

Using the idea mentioned in HotFlip (https://arxiv.org/pdf/1712.06751.pdf).
"""

import functools

from bert import modeling
from bert import tokenization

from bert_extraction.steal_bert_classifier.embedding_perturbations import embedding_util as em_util
from bert_extraction.steal_bert_classifier.models import run_classifier as rc

import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("total_steps", 100, "Total number of optimization steps.")

flags.DEFINE_integer("beam_size", 1,
                     "Beam size to carry out hotflip beam search.")

flags.DEFINE_integer("batch_size", 1, "Batch size for greedy decoding.")

flags.DEFINE_string("input_file_processor", "run_classifier",
                    "choice of data processor, run_classifier(_distillation).")

flags.DEFINE_string("input_template", "[EMPTY]<freq>*",
                    "CSV format to carry out inversion on templates of text.")

flags.DEFINE_string("prob_vector", None,
                    "probability vector to be used in some objectives.")

flags.DEFINE_string("obj_type", "max_self_entropy",
                    "The kind of loss function to optimize embeddings.")

flags.DEFINE_string("input_file", None,
                    "If there is an input file, ignore input_template.")

flags.DEFINE_string(
    "input_file_range", None,
    "For large input files, divide computation with this flag.")

flags.DEFINE_string("output_file", None,
                    "If an output file is specified, export results to it.")

flags.DEFINE_bool("accumulate_scores", False,
                  "During beam search add scores from the previous step.")

flags.DEFINE_bool("print_flips", False,
                  "Whether or not each flip should be logged.")

flags.DEFINE_string("flipping_mode", "greedy",
                    "Beam search, greedy or random perturbation.")

flags.DEFINE_string("stopping_criteria", None,
                    "Scheme adopted to stop the flip updates.")


def evaluate_stopping(stopping_criteria, obj_prob_vector, curr_prob_vector,
                      per_example_objective):
  """Evaluate whether flipping needs to stop or not."""
  if stopping_criteria == "hotflip":
    # Check whether the label has flipped or not
    return np.argmax(
        obj_prob_vector, axis=1) != np.argmax(
            curr_prob_vector, axis=1)

  elif stopping_criteria.startswith("greater_"):
    # Check whether the objective is greater than a threshold or not
    threshold = float(stopping_criteria[len("greater_"):])
    return per_example_objective >= threshold

  elif stopping_criteria.startswith("lesser_"):
    # Check whether the objective is lesser than a threshold or not
    threshold = float(stopping_criteria[len("lesser_"):])
    return per_example_objective <= threshold

  elif stopping_criteria.startswith("margin_lesser_"):
    # Check whether the margin between top two probabilities is lesser than
    # a threshold or not. This quantity is a measure of confidence.
    threshold = float(stopping_criteria[len("margin_lesser_"):])
    sorted_probs = np.sort(curr_prob_vector)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return margin <= threshold

  elif stopping_criteria.startswith("margin_greater_"):
    # Check whether the margin between top two probabilities is greater than
    # a threshold or not. This quantity is a measure of confidence.
    threshold = float(stopping_criteria[len("margin_greater_"):])
    sorted_probs = np.sort(curr_prob_vector)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return margin >= threshold

  else:
    tf.logging.warning("Stopping criteria not found!")
    return False


def greedy_updates(old_elements, grad_difference, biggest_indices,
                   max_seq_length):
  """Construct new beams with greedy updates based on gradient differences."""
  input_mask = np.array([elem["input_mask"] for elem in old_elements])
  # Mask tokens which cannot be flipped due to the input template constraint
  masked_gd = input_mask * np.reshape(grad_difference, [-1, max_seq_length])

  # Find the top flip score estimates by checking over tokens and word pieces
  top_indices = np.argmax(masked_gd, axis=1)
  top_scores = np.max(masked_gd, axis=1)

  new_elements = []
  for elem_num, (score, seq_num) in enumerate(zip(top_scores, top_indices)):
    old_elem = old_elements[elem_num]
    if old_elem["stopped"]:
      new_elements.append(old_elem)
    else:
      new_input_ids = list(old_elem["input_ids"])
      new_input_ids[seq_num] = biggest_indices[elem_num, seq_num, 0]
      new_elem = {
          "input_ids": np.array(new_input_ids),
          "original_input_ids": old_elem["original_input_ids"],
          "ip_num": old_elem["ip_num"],
          "score": score,
          "bert_input_mask": old_elem["bert_input_mask"],
          "input_mask": old_elem["input_mask"],
          "token_type_ids": old_elem["token_type_ids"],
          "prob_vector": old_elem["prob_vector"],
          "stopped": False,
          "steps_taken": old_elem["steps_taken"]
      }
      new_elements.append(new_elem)

  return new_elements


def beam_search(old_beams, grad_difference, biggest_indices, beam_size,
                accumulate_scores, max_seq_length):
  """Construct new beams using the old ones based on gradient differences."""
  input_mask = np.array([beam["input_mask"] for beam in old_beams])
  # Mask tokens which cannot be flipped due to the input template constraint
  masked_gd = input_mask * np.reshape(grad_difference, [-1, max_seq_length])
  if accumulate_scores:
    # Update scores to include old scores in previous beam
    # old_scores.shape = [beam_size, 1, 1], beam_size = 1 for first iteration
    old_scores = np.array([[[x["score"]]] for x in old_beams])
    updated_gd_scores = masked_gd + old_scores
  else:
    updated_gd_scores = masked_gd
  # Find the top flip score estimates by checking over tokens and word pieces
  masked_flat_gd = np.reshape(updated_gd_scores, -1)
  top_flat_indices = np.argsort(masked_flat_gd)[::-1][:beam_size]
  top_indices = np.unravel_index(top_flat_indices, dims=masked_gd.shape)
  top_scores = masked_flat_gd[top_flat_indices]

  new_beams = []
  for score, beam_num, seq_num, topk_num in zip(top_scores, top_indices[0],
                                                top_indices[1], top_indices[2]):
    old_beam = old_beams[beam_num]
    if old_beam["stopped"]:
      new_beams.append(old_beam)
    else:
      new_input_ids = list(old_beam["input_ids"])
      new_input_ids[seq_num] = biggest_indices[beam_num, seq_num, topk_num]
      new_beam = {
          "input_ids": np.array(new_input_ids),
          "original_input_ids": old_beam["original_input_ids"],
          "ip_num": old_beam["ip_num"],
          "score": score,
          "bert_input_mask": old_beam["bert_input_mask"],
          "input_mask": old_beam["input_mask"],
          "token_type_ids": old_beam["token_type_ids"],
          "prob_vector": old_beam["prob_vector"],
          "stopped": False,
          "steps_taken": old_beam["steps_taken"]
      }
      new_beams.append(new_beam)

  return new_beams


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.input_file_processor == "run_classifier":
    processors = {
        "sst-2": rc.SST2Processor,
        "mnli": rc.MnliProcessor,
    }
  elif FLAGS.input_file_processor == "run_classifier_distillation":
    processors = {
        "sst-2": rc.SST2ProcessorDistillation,
        "mnli": rc.MNLIProcessorDistillation,
    }
  else:
    raise ValueError("Invalid --input_file_processor flag value")

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  task_name = FLAGS.task_name.lower()
  processor = processors[task_name]()
  label_list = processor.get_labels()
  num_labels = len(label_list)

  input_ids_placeholder = tf.placeholder(
      dtype=tf.int32, shape=[None, FLAGS.max_seq_length])

  bert_input_mask_placeholder = tf.placeholder(
      dtype=tf.int32, shape=[None, FLAGS.max_seq_length])

  token_type_ids_placeholder = tf.placeholder(
      dtype=tf.int32, shape=[None, FLAGS.max_seq_length])

  prob_vector_placeholder = tf.placeholder(
      dtype=tf.float32, shape=[None, num_labels])

  one_hot_input_ids = tf.one_hot(
      input_ids_placeholder, depth=bert_config.vocab_size)

  input_tensor, _ = em_util.run_one_hot_embeddings(
      one_hot_input_ids=one_hot_input_ids, config=bert_config)

  flex_input_obj, per_eg_obj, probs = em_util.model_fn(
      input_tensor=input_tensor,
      bert_input_mask=bert_input_mask_placeholder,
      token_type_ids=token_type_ids_placeholder,
      bert_config=bert_config,
      num_labels=num_labels,
      obj_type=FLAGS.obj_type,
      prob_vector=prob_vector_placeholder)

  if FLAGS.obj_type.startswith("min"):
    final_obj = -1 * flex_input_obj
  elif FLAGS.obj_type.startswith("max"):
    final_obj = flex_input_obj

  # Calculate the gradient of the final loss function with respect to
  # the one-hot input space
  grad_obj_one_hot = tf.gradients(ys=final_obj, xs=one_hot_input_ids)[0]

  # gradients with respect to position in one hot input space with 1s in it
  # this is one term in the directional derivative of HotFlip,
  # Eq1 in https://arxiv.org/pdf/1712.06751.pdf
  #
  # grad_obj_one_hot.shape = [batch_size, seq_length, vocab_size]
  # input_ids_placeholder.shape = [batch_size, seq_length]
  # original_token_gradients.shape = [batch_size, seq_length]
  original_token_gradients = tf.gather(
      params=grad_obj_one_hot,
      indices=tf.expand_dims(input_ids_placeholder, -1),
      batch_dims=2)
  original_token_gradients = tf.tile(
      original_token_gradients, multiples=[1, 1, FLAGS.beam_size])

  # These are the gradients / indices whose one-hot position has the largest
  # gradient magnitude, the performs part of the max calculation in Eq10 of
  # https://arxiv.org/pdf/1712.06751.pdf
  biggest_gradients, biggest_indices = tf.nn.top_k(
      input=grad_obj_one_hot, k=FLAGS.beam_size)

  # Eq10 of https://arxiv.org/pdf/1712.06751.pdf
  grad_difference = biggest_gradients - original_token_gradients

  tvars = tf.trainable_variables()

  assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
      tvars, FLAGS.init_checkpoint)

  tf.logging.info("Variables mapped = %d / %d", len(assignment_map), len(tvars))

  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  if FLAGS.input_file:
    custom_examples = processor.get_custom_examples(FLAGS.input_file)
    custom_templates = [
        em_util.input_to_template(x, label_list) for x in custom_examples
    ]
  else:
    prob_vector = [float(x) for x in FLAGS.prob_vector.split(",")]
    custom_templates = [(FLAGS.input_template, prob_vector)]

  num_input_sequences = custom_templates[0][0].count("[SEP]")

  if FLAGS.flipping_mode == "beam_search":
    FLAGS.batch_size = 1

  detok_partial = functools.partial(em_util.detokenize, tokenizer=tokenizer)

  # Since input files will often be quite large, this flag allows processing
  # only a slice of the input file
  if FLAGS.input_file_range:
    start_index, end_index = FLAGS.input_file_range.split("-")
    if start_index == "start":
      start_index = 0
    if end_index == "end":
      end_index = len(custom_templates)
    start_index, end_index = int(start_index), int(end_index)
  else:
    start_index = 0
    end_index = len(custom_templates)

  tf.logging.info("Processing examples in range %d, %d", start_index, end_index)

  all_elements = []

  too_long = 0

  for ip_num, (ip_template, prob_vector) in enumerate(
      custom_templates[start_index:end_index]):
    # Parse the input template into a list of IDs and the corresponding mask.
    # Different segments in template are separated by " <piece> "
    # Each segment is associated with a word piece (or [EMPTY] to get flex
    # inputs) and a frequency. (which is separated by "<freq>"). * can be used
    # to choose a frequency till the end of the string
    #
    # Here is an example 2-sequence template for tasks like MNLI to optimize
    # 20 vectors, (10 for each sequence)
    # [CLS]<freq>1 <piece> [EMPTY]<freq>10 <piece> [SEP]<freq>1 <piece> \
    # [EMPTY]<freq>10 <piece> [SEP]<freq>1 <piece> [PAD]<freq>*
    (input_ids, input_mask, bert_input_mask,
     token_type_ids) = em_util.template_to_ids(
         template=ip_template,
         config=bert_config,
         tokenizer=tokenizer,
         max_seq_length=FLAGS.max_seq_length)

    if len(input_ids) > FLAGS.max_seq_length:
      # truncate them!
      input_ids = input_ids[:FLAGS.max_seq_length]
      input_mask = input_mask[:FLAGS.max_seq_length]
      bert_input_mask = bert_input_mask[:FLAGS.max_seq_length]
      token_type_ids = token_type_ids[:FLAGS.max_seq_length]
      too_long += 1

    all_elements.append({
        "input_ids": input_ids,
        "original_input_ids": list(input_ids),
        "ip_num": start_index + ip_num,
        "score": 0.0,
        "bert_input_mask": bert_input_mask,
        "input_mask": input_mask,
        "token_type_ids": token_type_ids,
        "prob_vector": prob_vector,
        "stopped": False,
        "steps_taken": 0
    })

  tf.logging.info("%d / %d were too long and hence truncated.", too_long,
                  len(all_elements))

  iteration_number = 0
  consistent_output_sequences = []

  while all_elements and iteration_number < 10:

    steps_taken = []
    output_sequences = []
    failures = []
    zero_step_instances = 0

    iteration_number += 1
    tf.logging.info("Starting iteration number %d", iteration_number)
    tf.logging.info("Pending items = %d / %d", len(all_elements),
                    len(custom_templates[start_index:end_index]))

    batch_elements = []
    for ip_num, input_object in enumerate(all_elements):
      batch_elements.append(input_object)
      # wait until the input has populated up to the batch size
      if (len(batch_elements) < FLAGS.batch_size and
          ip_num < len(all_elements) - 1):
        continue

      # optimize a part of the flex_input (depending on the template)
      for step_num in range(FLAGS.total_steps):
        feed_dict = {
            input_ids_placeholder:
                np.array([x["input_ids"] for x in batch_elements]),
            bert_input_mask_placeholder:
                np.array([x["bert_input_mask"] for x in batch_elements]),
            token_type_ids_placeholder:
                np.array([x["token_type_ids"] for x in batch_elements]),
            prob_vector_placeholder:
                np.array([x["prob_vector"] for x in batch_elements])
        }

        if FLAGS.flipping_mode == "random":
          # Avoiding the gradient computation when the flipping mode is random
          peo, pr = sess.run([per_eg_obj, probs], feed_dict=feed_dict)
        else:
          peo, gd, bi, pr = sess.run(
              [per_eg_obj, grad_difference, biggest_indices, probs],
              feed_dict=feed_dict)

        if FLAGS.print_flips:
          output_log = "\n" + "\n".join([
              "Objective = %.4f, Score = %.4f, Element %d = %s" %
              (obj, elem["score"], kk, detok_partial(elem["input_ids"]))
              for kk, (obj, elem) in enumerate(zip(peo, batch_elements))
          ])
          tf.logging.info("Step = %d %s\n", step_num, output_log)

        should_stop = evaluate_stopping(
            stopping_criteria=FLAGS.stopping_criteria,
            obj_prob_vector=np.array([x["prob_vector"] for x in batch_elements
                                     ]),
            curr_prob_vector=pr,
            per_example_objective=peo)

        for elem, stop_bool in zip(batch_elements, should_stop):
          if stop_bool and (not elem["stopped"]):
            if step_num == 0:
              # don't actually stop the perturbation since we want a new input
              zero_step_instances += 1
            else:
              elem["stopped"] = True
              elem["steps_taken"] = step_num

        if np.all([elem["stopped"] for elem in batch_elements]):
          steps_taken.extend([elem["steps_taken"] for elem in batch_elements])
          output_sequences.extend(list(batch_elements))
          batch_elements = []
          break

        if step_num == FLAGS.total_steps - 1:
          failures.extend(
              [elem for elem in batch_elements if not elem["stopped"]])
          steps_taken.extend([
              elem["steps_taken"] for elem in batch_elements if elem["stopped"]
          ])
          output_sequences.extend(
              [elem for elem in batch_elements if elem["stopped"]])
          batch_elements = []
          break

        # Flip a token / word-piece either systematically or randomly
        # For instances where hotflip was not successful, do some random
        # perturbations before doing hotflip
        if (FLAGS.flipping_mode == "random" or
            (iteration_number > 1 and step_num < iteration_number)):
          for element in batch_elements:
            # don't perturb elements which have stopped
            if element["stopped"]:
              continue

            random_seq_index = np.random.choice([
                ii for ii, mask_id in enumerate(element["input_mask"])
                if mask_id > 0.5
            ])

            random_token_id = np.random.randint(len(tokenizer.vocab))
            while (tokenizer.inv_vocab[random_token_id][0] == "[" and
                   tokenizer.inv_vocab[random_token_id][-1] == "]"):
              random_token_id = np.random.randint(len(tokenizer.vocab))

            element["input_ids"][random_seq_index] = random_token_id

        elif FLAGS.flipping_mode == "greedy":
          batch_elements = greedy_updates(
              old_elements=batch_elements,
              grad_difference=gd,
              biggest_indices=bi,
              max_seq_length=FLAGS.max_seq_length)

        elif FLAGS.flipping_mode == "beam_search":
          # only supported with a batch size of 1!
          batch_elements = beam_search(
              old_beams=batch_elements,
              grad_difference=gd,
              biggest_indices=bi,
              beam_size=FLAGS.beam_size,
              accumulate_scores=FLAGS.accumulate_scores,
              max_seq_length=FLAGS.max_seq_length)

        else:
          raise ValueError("Invalid --flipping_mode flag value")

      tf.logging.info("steps = %.4f (%d failed, %d non-zero, %d zero)",
                      np.mean([float(x) for x in steps_taken if x > 0]),
                      len(failures), len([x for x in steps_taken if x > 0]),
                      zero_step_instances)

    # measure consistency of final dataset - run a forward pass through the
    # entire final dataset and verify it satisfies the original objective. This
    # if the code runs correctly, total_inconsistent = 0
    tf.logging.info("Measuring consistency of final dataset")

    total_inconsistent = 0
    total_lossy = 0

    for i in range(0, len(output_sequences), FLAGS.batch_size):
      batch_elements = output_sequences[i:i + FLAGS.batch_size]
      feed_dict = {
          input_ids_placeholder:
              np.array([x["input_ids"] for x in batch_elements]),
          bert_input_mask_placeholder:
              np.array([x["bert_input_mask"] for x in batch_elements]),
          token_type_ids_placeholder:
              np.array([x["token_type_ids"] for x in batch_elements]),
          prob_vector_placeholder:
              np.array([x["prob_vector"] for x in batch_elements])
      }
      peo, pr = sess.run([per_eg_obj, probs], feed_dict=feed_dict)
      consistency_flags = evaluate_stopping(
          stopping_criteria=FLAGS.stopping_criteria,
          obj_prob_vector=np.array([x["prob_vector"] for x in batch_elements]),
          curr_prob_vector=pr,
          per_example_objective=peo)
      total_inconsistent += len(batch_elements) - np.sum(consistency_flags)

      # Next, apply a lossy perturbation to the input (conversion to a string)
      # This is often lossy since it eliminates impossible sequences and
      # incorrect tokenizations. We check how many consistencies still hold true
      all_detok_strings = [
          em_util.ids_to_strings(elem["input_ids"], tokenizer)
          for elem in batch_elements
      ]

      all_ip_examples = []
      if num_input_sequences == 1:
        for ds, be in zip(all_detok_strings, batch_elements):
          prob_vector_labels = be["prob_vector"].tolist()
          all_ip_examples.append(
              rc.InputExample(
                  text_a=ds[0],
                  text_b=None,
                  label=prob_vector_labels,
                  guid=None))
      else:
        for ds, be in zip(all_detok_strings, batch_elements):
          prob_vector_labels = be["prob_vector"].tolist()
          all_ip_examples.append(
              rc.InputExample(
                  text_a=ds[0],
                  text_b=ds[1],
                  label=prob_vector_labels,
                  guid=None))

      all_templates = [
          em_util.input_to_template(aie, label_list) for aie in all_ip_examples
      ]
      all_new_elements = []
      for ip_template, prob_vector in all_templates:
        (input_ids, input_mask, bert_input_mask,
         token_type_ids) = em_util.template_to_ids(
             template=ip_template,
             config=bert_config,
             tokenizer=tokenizer,
             max_seq_length=FLAGS.max_seq_length)

        if len(input_ids) > FLAGS.max_seq_length:
          input_ids = input_ids[:FLAGS.max_seq_length]
          input_mask = input_mask[:FLAGS.max_seq_length]
          bert_input_mask = bert_input_mask[:FLAGS.max_seq_length]
          token_type_ids = token_type_ids[:FLAGS.max_seq_length]

        all_new_elements.append({
            "input_ids": input_ids,
            "input_mask": input_mask,
            "bert_input_mask": bert_input_mask,
            "token_type_ids": token_type_ids,
            "prob_vector": prob_vector
        })
      feed_dict = {
          input_ids_placeholder:
              np.array([x["input_ids"] for x in all_new_elements]),
          bert_input_mask_placeholder:
              np.array([x["bert_input_mask"] for x in all_new_elements]),
          token_type_ids_placeholder:
              np.array([x["token_type_ids"] for x in all_new_elements]),
          prob_vector_placeholder:
              np.array([x["prob_vector"] for x in all_new_elements])
      }
      peo, pr = sess.run([per_eg_obj, probs], feed_dict=feed_dict)
      lossy_consistency_flags = evaluate_stopping(
          stopping_criteria=FLAGS.stopping_criteria,
          obj_prob_vector=np.array([x["prob_vector"] for x in all_new_elements
                                   ]),
          curr_prob_vector=pr,
          per_example_objective=peo)

      total_lossy += len(all_new_elements) - np.sum(lossy_consistency_flags)

      net_consistency_flags = np.logical_and(consistency_flags,
                                             lossy_consistency_flags)

      for elem, ncf in zip(batch_elements, net_consistency_flags):
        if ncf:
          consistent_output_sequences.append(elem)
        else:
          failures.append(elem)

    tf.logging.info("Total inconsistent found = %d / %d", total_inconsistent,
                    len(output_sequences))
    tf.logging.info("Total lossy inconsistent found = %d / %d", total_lossy,
                    len(output_sequences))
    tf.logging.info("Total consistent outputs so far = %d / %d",
                    len(consistent_output_sequences),
                    len(custom_templates[start_index:end_index]))

    # Getting ready for next iteration of processing
    if iteration_number < 10:
      for elem in failures:
        elem["input_ids"] = list(elem["original_input_ids"])
        elem["stopped"] = False
        elem["steps_taken"] = 0
        elem["score"] = 0.0
      all_elements = failures

  tf.logging.info("Giving up on %d instances!", len(failures))
  for elem in failures:
    consistent_output_sequences.append(elem)

  if FLAGS.output_file:
    final_output = []
    for op_num, elem in enumerate(consistent_output_sequences):
      detok_strings = em_util.ids_to_strings(elem["input_ids"], tokenizer)

      if num_input_sequences == 1:
        final_output.append("%d\t%d\t%s" %
                            (op_num, elem["ip_num"], detok_strings[0]))
      elif num_input_sequences == 2:
        final_output.append(
            "%d\t%d\t%s\t%s" %
            (op_num, elem["ip_num"], detok_strings[0], detok_strings[1]))

    if num_input_sequences == 1:
      header = "index\toriginal_index\tsentence"
    elif num_input_sequences == 2:
      header = "index\toriginal_index\tsentence1\tsentence2"

    final_output = [header] + final_output

    with tf.gfile.Open(FLAGS.output_file, "w") as f:
      f.write("\n".join(final_output) + "\n")

  return


if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()
