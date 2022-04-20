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
"""Implementation of Count-Min Sketch.

Implement a count-min sketch module that can create count-min sketch, check
membership of an element, and compute intersection and union of the sketches
of two sets.
"""


from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm

FLAGS = flags.FLAGS
np.random.seed(0)


class CountMinContext(object):
  """Definition of countmin sketch context.

  A CountMinContext hold the information needed to construct a count-min
  sketch. It caches the hash values of observed elements.
  """

  def __init__(self, width, depth, n = -1):
    """Initialize the count-min sketch context.

    Pre-compute the hashes of all elements if the number of elements is
    known (n>0).

    Args:
      width: width of the cm-sketch
      depth: depth of the cm-sketch
      n: number of elements, -1 if it's unknown
    """
    self.width = width
    self.depth = depth
    self.cache = dict()  # cache of hash value to a list of ids
    if n != -1:
      for e in tqdm(range(n)):
        e = str(e)
        self.cache[e] = [self._hash(e, i) for i in range(self.depth)]

  def _hash(self, x, i):
    """Get the i'th hash value of element x.

    Args:
      x: name or id in string
      i: the i'th hash function

    Returns:
      hash result
    """
    assert isinstance(x, str)
    assert isinstance(i, int)
    assert i >= 0 and i < self.depth
    hash_val = hash((i, x))
    return hash_val % self.width

  def get_hashes(self, x):
    """Get the hash values of x.

    Each element is hashed w times, where w is the width of the count-min
    sketch specified in the constructor of CountMinContext. This function
    returns w hash values of element x.

    Args:
      x: name or id in string

    Returns:
      a list of hash values with the length of depth
    """
    x = str(x)
    if x not in self.cache:
      self.cache[x] = [self._hash(x, i) for i in range(self.depth)]
    return self.cache[x]

  def get_sketch(self, xs = None):
    """Return a sketch for set xs (all zeros if xs not specified).

    This function takes a list of elements xs, take their hash values, and
    set 1.0 to the corresponding positions. It returns a 2d numpy array
    with width and depth declared in the constructor of CountMinContext.
    Values at unassigned positions remain 0.

    Args:
      xs: a set of name or id in string

    Returns:
      a sketch np.array()
    """
    sketch = np.zeros((self.depth, self.width), dtype=np.float32)
    if xs is not None:
      self.add_set(sketch, xs)
    return sketch

  def add(self, sketch, x):
    """Add an element to the sketch.

    Args:
      sketch: sketch to add x to
      x: name or id in string
    """
    assert isinstance(x, str)
    assert self.depth, self.width == sketch.shape

    if x not in self.cache:
      self.cache[x] = [self._hash(x, i) for i in range(self.depth)]
    for i in range(self.depth):
      sketch[i, self.cache[x][i]] += 1.0

  def add_set(self, sketch, xs):
    """Add a set of elements to the sketch.

    Args:
      sketch: sketch to add xs to
      xs: a set of name or id in string
    """
    assert self.depth, self.width == sketch.shape
    for x in xs:
      x = str(x)
      if not self.contain(sketch, x):
        self.add(sketch, x)

  def contain(self, sketch, x):
    """Check if the sketch contains x.

    Args:
      sketch: sketch to add xs to
      x: name or id in string

    Returns:
      True or False
    """
    assert self.depth, self.width == sketch.shape
    x = str(x)
    if x not in self.cache:
      self.cache[x] = [self._hash(x, i) for i in range(self.depth)]
    for i in range(self.depth):
      if sketch[i, self.cache[x][i]] == 0.0:
        return False
    return True

  def intersection(self, sk1, sk2):
    """Intersect two sketches.

    Args:
      sk1: first sketch
      sk2: second sketch

    Returns:
      a countmin sketch for intersection
    """
    assert sk1.shape == sk2.shape
    assert self.depth, self.width == sk1.shape
    sk_intersection = sk1 * sk2
    return sk_intersection

  def union(self, sk1, sk2):
    """Union two sketches.

    Args:
      sk1: first sketch
      sk2: second sketch

    Returns:
      a countmin sketch for union
    """
    assert sk1.shape == sk2.shape
    assert self.depth, self.width == sk1.shape
    sk_union = sk1 + sk2
    return sk_union


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
