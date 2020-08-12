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
# Lint as: python3
"""Tools for parallel processing.

Defines an Executor class, for parallel processing.

In newer versions of Python (>=3.7), this is subsumed by ProcessPoolExecutor,
whose constructor takes mp_context, initializer and init_args.
"""
import multiprocessing
import os
import queue
import time


def process_from_queue(input_queue, output_queue, worker, stop_event):
  """Processes items in a queue."""
  while not stop_event.is_set():
    try:
      # Keep trying to get an item.
      x = input_queue.get(timeout=0.05)
    except queue.Empty:
      continue

    # Do actual work.
    y = worker(x)

    while not stop_event.is_set():
      try:
        # Keep trying to push the item.
        output_queue.put(y, timeout=0.05)
        break
      except queue.Full:
        pass


def submit_generator_to_queue(input_queue, generator_fn, generator_kwargs,
                              stop_event):
  """Submits items from a generator into a queue."""
  generator = generator_fn(**generator_kwargs)
  for x in generator:
    while not stop_event.is_set():
      try:
        # Keep trying to push the item.
        input_queue.put(x, timeout=0.05)
        break
      except queue.Full:
        pass


class Executor(object):
  """Feeds tasks to multiple workers in parallel.

  General note on using the Python multiprocessing module:
  If there is any code in your entry-point script that leads to the creation of
  a Process, it should be nested under the following if-statement:

  if __name__ ==  '__main__':
    <your code>

  Every time a new Process is created, it imports __main__, so you don't want
  <your code> to be called in every new Process.

  """

  def __init__(self,
               create_worker,
               queue_size,
               num_workers=None,
               worker_kwargs=None):
    if num_workers is None:
      # Use all CPUs.
      num_workers = os.cpu_count()

    if worker_kwargs is None:
      worker_kwargs = {}

    self._mp = multiprocessing

    self._input_queue = self._mp.Queue(queue_size)
    self._output_queue = self._mp.Queue(queue_size)
    self._worker_processes = []
    self._submitter_processes = []
    self._stop_event = self._mp.Event()

    for i in range(num_workers):
      worker = create_worker(**worker_kwargs)
      process = self._mp.Process(
          target=process_from_queue,
          name='worker-{}'.format(i),
          args=(self._input_queue, self._output_queue, worker,
                self._stop_event))
      process.start()
      self._worker_processes.append(process)

  def _shutdown_queue(self, q):
    """Shuts down the multiprocessing Queue."""
    q.close()
    q.join_thread()

  def shutdown(self):
    """Shuts down all relevant resources."""
    # Send stop signal to worker and submit processes.
    self._stop_event.set()

    # We will force-terminate any process that is still alive 10 seconds after
    # we sent the stop signal.
    force_terminate_time = time.time() + 10

    self._shutdown_queue(self._input_queue)
    self._shutdown_queue(self._output_queue)

    # Wait until force_terminate_time for processes to gracefully stop.
    for proc in self._worker_processes + self._submitter_processes:
      wait_time = max(0, force_terminate_time - time.time())
      proc.join(wait_time)

      # If it's still alive, force terminate it.
      if proc.is_alive():
        proc.terminate()

  def submit(self, x):
    self._input_queue.put(x)

  def submit_from_generator(self, generator_fn, **generator_kwargs):
    process_idx = len(self._submitter_processes)
    process = self._mp.Process(
        target=submit_generator_to_queue,
        name='submitter-{}-{}'.format(generator_fn.__name__, process_idx),
        args=(self._input_queue, generator_fn, generator_kwargs,
              self._stop_event))
    process.start()
    self._submitter_processes.append(process)

  def results(self, max_to_yield=float('inf')):
    num_yielded = 0
    while True:
      if num_yielded >= max_to_yield:
        break
      y = self._output_queue.get()
      yield y
      num_yielded += 1

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.shutdown()
