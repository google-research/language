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
from language.boolq.utils import ops
import tensorflow.compat.v1 as tf


class OpsTest(tf.test.TestCase):

  def test_lowercase(self):
    with self.test_session() as sess:
      test_str = [["Abc%@||", "DZ dzD", ""]]
      self.assertEqual([
          x.decode() for x in sess.run(
              ops.lowercase_op(tf.convert_to_tensor(test_str))).tolist()[0]
      ], [x.lower() for x in test_str[0]])

  def test_lowercase_unicode(self):
    with self.test_session() as sess:
      test_str = ["ŠČŽɬЩЮɦ"]
      self.assertEqual([
          sess.run(ops.lowercase_op(
              tf.convert_to_tensor(test_str))).tolist()[0].decode()
      ], [test_str[0].lower()])

  def test_bucket_by_quantiles(self):
    with self.test_session() as sess:
      data = tf.data.Dataset.from_tensor_slices(list(range(10))).repeat()
      data = data.apply(
          ops.bucket_by_quantiles(
              len_fn=lambda x: x,
              batch_size=4,
              n_buckets=2,
              hist_bounds=[2, 4, 6, 8]))
      # Turn off `inject_prefetch` optimization
      options = tf.data.Options()
      options.experimental_optimization.inject_prefetch = False
      data = data.with_options(options)
      it = data.make_initializable_iterator()
      sess.run(it.initializer)
      sess.run(tf.local_variables_initializer())
      next_op = it.get_next()

      # Let the model gather statistics, it sees 4*5=20 = 2 epochs,
      # so each bin should have a count of 4
      for _ in range(5):
        sess.run(next_op)

      counts = sess.run(tf.local_variables()[0])
      self.assertEqual(counts.tolist(), [4, 8, 12, 16, 20])

      # At this point the model should perfectly quantize the input
      for _ in range(4):
        out = sess.run(next_op)
        if out[0] < 5:
          self.assertAllInRange(out, 0, 5)
        else:
          self.assertAllInRange(out, 5, 10)


if __name__ == "__main__":
  tf.test.main()
