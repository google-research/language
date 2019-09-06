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
"""Extract features from VCR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import collections
import functools
import json
import os
import re
import zipfile

import absl
from bert import tokenization
import tensorflow as tf

flags = absl.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", os.path.expanduser("~/data/vcr"),
    "Directory containing downloaded VCR data.")

flags.DEFINE_string("vocab_path", None,
                    "BERT vocab.")

flags.DEFINE_string(
    "output_tfrecord", None,
    "Tf record file to write extracted features to.")

flags.DEFINE_integer(
    "shard", 0,
    "Shard number for parallel processing.")

flags.DEFINE_integer(
    "num_shards", 20,
    "Total number of shards for parallel processing.")

flags.DEFINE_integer("max_seq_length", 64, "Maximum sequence length for BERT.")

flags.DEFINE_integer("max_num_bboxes", 4,
                     "Maximum number of bounding boxes to consider.")

flags.DEFINE_bool(
    "append_all_bboxes", False,
    "Append all bboxes to end of token sequence, up to max_num_bboxes. We add "
    "the bboxes in order, including ones already mentioned in the text.")

flags.DEFINE_bool(
    "include_rationales", False,
    "Whether to include rationales as choices. VCR requires to predict the "
    "answer first without the rationale, so we train a separate model for "
    "this case.")

# A dummy JPEG to use in place of corrupted images.
BLANK_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh"
    "0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAA"
    "EDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIE"
    "AwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJi"
    "coKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWW"
    "l5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09f"
    "b3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQA"
    "AQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKj"
    "U2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJma"
    "oqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9"
    "oADAMBAAIRAxEAPwD3+iiigD//2Q==")


def make_int64_feature(v):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def make_float_feature(v):
  return tf.train.Feature(float_list=tf.train.FloatList(value=v))


def make_bytes_feature(v):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


def parse_image(height, width, image_string):
  """Decodes an image, resizes and scales RGB values to be in [0, 1]."""
  image_decoded = tf.image.decode_image(image_string, channels=3)
  image_decoded.set_shape([None, None, 3])
  image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
  image_resized = tf.image.resize_image_with_pad(image_float, height, width)
  image = tf.reshape(image_resized, [1, height, width, 3])
  return image


def tokenize(tokenizer, example, text):
  """Tokenize."""
  tokens = []
  for word_token in text:
    if isinstance(word_token, list):
      for i, object_id in enumerate(word_token):
        if i: tokens.append("and")
        tokens.extend(tokenizer.tokenize(example["objects"][object_id]))
        tokens.append("[OBJ-%d]" % object_id)
    else:
      tokens.extend(tokenizer.tokenize(word_token))
  tf.logging.info("Tokenization: %s", tokens)
  return tokens


def cap_length(q, a, r, length):
  """Cap total length of q, a, r to length."""
  assert length > 0, "length to cap too short: %d" % length
  while len(q) + len(a) + len(r) >= length:
    max_len = max(len(q), len(a), len(r))
    if len(r) == max_len:
      r = r[:-1]
    elif len(a) == max_len:
      a = a[:-1]
    else:
      q = q[:-1]
  return q, a, r


def get_bboxes(tokens, metadata):
  """Extracts bounding boxes relevant to the given tokens.

  Args:
    tokens: list of string tokens where objects with bounding boxes are denoted
      [OBJ-`object_id`]. `object_id` indexes in `metadata["boxes"]`.
    metadata: dictionary containing image metadata from the VCR dataset,
      including bounding boxes of referenced objects.

  Returns:
    bbox_positions_matrix: a `FLAGS.max_num_bboxes` x 4 int matrix, containing
            the first `FLAGS.max_num_bboxes` bounding boxes referenced in the
            tokens. Each row contains offset_height, offset_width,
            target_height and target_width of the bounding box.
    bbox_indices_vector: an int vector of length `len(tokens)`, containing
            the index of the bounding box relevant to each token, or -1 if no
            bounding box is referenced.
  """
  bbox_positions = {}
  bbox_indices = collections.defaultdict(list)
  for idx, t in enumerate(tokens):
    if len(bbox_positions) >= FLAGS.max_num_bboxes:
      break
    m = re.match(r"\[OBJ-(\d+)\]", t)
    if m:
      object_id = int(m.group(1))
      bbox_positions[object_id] = metadata["boxes"][object_id]
      bbox_indices[object_id].append(idx)

  # Adds dummy bounding boxes if there are fewer than `FLAGS.max_num_bboxes`.
  for idx in range(FLAGS.max_num_bboxes):
    if len(bbox_positions) == FLAGS.max_num_bboxes:
      break
    bbox_positions[-idx - 1] = [0, 0, 1, 1]

  bbox_positions_matrix = []
  bbox_indices_vector = [-1] * len(tokens)
  for idx, (object_id, bbox) in enumerate(bbox_positions.iteritems()):
    offset_height = int(bbox[1])
    offset_width = int(bbox[0])
    target_height = int(bbox[3] - bbox[1])
    target_width = int(bbox[2] - bbox[0])
    bbox_positions_matrix.extend(
        [offset_height, offset_width, target_height, target_width])
    for token_idx in bbox_indices[object_id]:
      bbox_indices_vector[token_idx] = idx

  tf.logging.info("Box positions: %s", bbox_positions_matrix)
  tf.logging.info("Box token indices: %s", bbox_indices_vector)
  return bbox_positions_matrix, bbox_indices_vector


def create_tf_examples(tokenizer,
                       example,
                       image_string,
                       metadata,
                       is_test=False):
  """Creates TF examples for the given VCR example and image feature vector."""
  tokenize_fn = functools.partial(tokenize, tokenizer, example)
  q = tokenize_fn(example["question"])

  if FLAGS.include_rationales:
    if is_test:
      answers = [tokenize_fn(a) for a in example["answer_choices"]]
    else:
      # Only get true answer.
      a = example["answer_choices"][example["answer_label"]]
      answers = [tokenize_fn(a)]
    rationales = [tokenize_fn(r) for r in example["rationale_choices"]]
  else:
    answers = [tokenize_fn(a) for a in example["answer_choices"]]
    rationales = [[]]
  y = collections.OrderedDict()
  y["image"] = make_bytes_feature([image_string])

  annot_id = int(re.match(".*-([0-9]*)", example["annot_id"]).group(1))
  img_id = int(re.match(".*-([0-9]*)", example["img_id"]).group(1))
  y["annot_id"] = make_int64_feature([annot_id])
  y["img_id"] = make_int64_feature([img_id])

  for i, a in enumerate(answers):
    for j, r in enumerate(rationales):
      # Create text input.
      extra_tokens = []
      if FLAGS.append_all_bboxes:
        # Append bboxes to end of tokens.
        num_appended_bboxes = min(len(metadata["boxes"]), FLAGS.max_num_bboxes)
        for idx in range(num_appended_bboxes):
          extra_tokens.extend(tokenizer.tokenize(example["objects"][idx]))
          extra_tokens.append("[OBJ-%d]" % idx)

      max_len = FLAGS.max_seq_length - 4 - len(extra_tokens)
      q, a, r = cap_length(q, a, r, max_len)

      tokens = ["[CLS]", "[IMAGE]"] + q + ["[SEP]"] + a + r + ["[SEP]"]
      tokens.extend(extra_tokens)

      tf.logging.info("Final tokens: %s", " ".join(tokens))

      bbox_positions_matrix, bbox_indices_vector = get_bboxes(tokens, metadata)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      segment_ids = [0] * (len(q) + 3) + [1] * (
          len(a) + len(r) + 1 + len(extra_tokens))
      input_mask = [1] * len(input_ids)

      # Pad.
      padding_len = FLAGS.max_seq_length - len(input_ids)
      input_ids.extend([0] * padding_len)
      segment_ids.extend([0] * padding_len)
      input_mask.extend([0] * padding_len)
      bbox_indices_vector.extend([-1] * padding_len)
      if (len(input_ids) != FLAGS.max_seq_length or
          len(segment_ids) != FLAGS.max_seq_length or
          len(input_mask) != FLAGS.max_seq_length or
          len(bbox_indices_vector) != FLAGS.max_seq_length):
        tf.logging.fatal("Bad feature lengths: %d, %d, %d, %d", len(input_ids),
                         len(segment_ids), len(input_mask),
                         len(bbox_indices_vector))

      # Is this the right choice?
      if "answer_label" in example and "rationale_label" in example:
        if FLAGS.include_rationales:
          positive = j == example["rationale_label"]
        else:
          positive = i == example["answer_label"]
      else:
        positive = False

      y["choice_id"] = make_int64_feature([i * 4 + j])
      y["input_ids"] = make_int64_feature(input_ids)
      y["segment_ids"] = make_int64_feature(segment_ids)
      y["input_mask"] = make_int64_feature(input_mask)
      y["label"] = make_int64_feature([int(positive)])
      y["bbox_pos"] = make_int64_feature(bbox_positions_matrix)
      y["bbox_idx"] = make_int64_feature(bbox_indices_vector)
      yield tf.train.Example(features=tf.train.Features(feature=y))


def main(_):
  tf.gfile.MakeDirs(os.path.dirname(FLAGS.output_tfrecord))
  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_path,
                                         do_lower_case=True)

  annotations_zipfn = os.path.join(FLAGS.data_dir, "vcr1annots.zip")
  images_zipfn = os.path.join(FLAGS.data_dir, "vcr1images.zip")

  # Generate data for all splits:
  for split in ["train", "val", "test"]:
    jsonl_file = split + ".jsonl"
    output_tfrecord = "-".join([FLAGS.output_tfrecord,
                                split,
                                "%05d" % FLAGS.shard,
                                "of",
                                "%05d" % FLAGS.num_shards])
    with tf.python_io.TFRecordWriter(output_tfrecord) as writer:
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        with zipfile.ZipFile(
            tf.gfile.Open(annotations_zipfn)) as annotations_zip:
          with zipfile.ZipFile(tf.gfile.Open(images_zipfn)) as images_zip:
            with annotations_zip.open(jsonl_file) as jsonl:
              for idx, line in enumerate(jsonl):
                if idx % FLAGS.num_shards != FLAGS.shard:
                  continue
                example = json.loads(line)
                meta_filename = "vcr1images/" + example["metadata_fn"]
                meta = json.loads(images_zip.open(meta_filename).read())
                del meta["segms"]

                try:
                  image_filename = "vcr1images/" + example["img_fn"]
                  tf.logging.info("Reading %s", image_filename)
                  with images_zip.open(image_filename) as image:
                    image_string = image.read()
                except zipfile.BadZipfile as e:
                  tf.logging.error("Bad Zip file: " + str(e))
                  image_string = BLANK_JPEG
                  for box in meta["boxes"]:
                    box[0] = 0.0
                    box[1] = 0.0
                    box[2] = 1.0
                    box[3] = 1.0

                is_test = (split == "test")
                for tf_example in create_tf_examples(tokenizer, example,
                                                     image_string, meta,
                                                     is_test=is_test):
                  writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
  tf.app.run()
