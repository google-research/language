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
"""Implementation of multi-bootstrap for comparing model performance."""


import numpy as np
import pandas as pd


def _bsample(items, rng=np.random):
  """Sample n from n with replacement."""
  return rng.choice(items, size=len(items), replace=True)


def multibootstrap(run_info,
                   preds,
                   labels,
                   score_fn,
                   paired_seeds = False,
                   nboot = 500,
                   rng = np.random,
                   verbose = True,
                   progress_indicator=lambda x: x):
  """Compute bootstrap samples of score_fn using multibootstrap over seeds and examples.

  Args:
    run_info: DataFrame with columns 'seed' (any type) and 'intervention' (bool)
      Any other columns will be ignored, and all rows with the same value of
      (seed, intervention) will have their scores averaged. This is done as a
      macro-average, such that each seed contributes equally to the final
      average.
    preds: [num_runs, num_examples] matrix of predictions, aligned to rows of
      run_info.
    labels: [num_examples] array of target labels, aligned to columns of preds.
    score_fn: function that takes (y_true: List, y_pred: List) and returns a
      scalar.
    paired_seeds: if True, assumes base and expt are paired on the same set of
      seeds, and will use the same set of sampled seeds for both sides. If
      False, will sample seeds independently but will still use paired sets of
      examples.
    nboot: number of bootstrap samples to compute.
    rng: random number generator state, or an int to seed a new one.
    verbose: set to False to disable printing of intermediate info.
    progress_indicator: optional, pass-through iterator like tqdm to indicate
      progress, as computing many samples can take a while.

  Returns:
    List[Tuple[float, float]] of nboot paired bootstrap samples (L, L')
  """
  assert len(run_info) == len(preds)
  assert len(labels) == preds.shape[1]
  num_examples = len(labels)
  # Reset dataframe index to be row indices [0, 1, ..., num_runs]
  run_info = run_info.reset_index(drop=True)
  # Map seed -> list[int] of row indices into preds
  mask = run_info['intervention']
  base_seed_to_row = run_info[~mask].groupby('seed').groups
  expt_seed_to_row = run_info[mask].groupby('seed').groups

  if isinstance(rng, int):
    rng = np.random.RandomState(rng)

  if paired_seeds:
    if verbose:
      print(f'Multibootstrap (paired) on {num_examples} examples')
    # Get set of seed keys in intersection
    common_seeds = list(
        set(base_seed_to_row.keys()).intersection(expt_seed_to_row.keys()))
    if verbose:
      print(f'  Common seeds ({len(common_seeds)}): {str(common_seeds)}')
    # Warn if there are any seeds which don't overlap
    base_only_seeds = set(base_seed_to_row.keys()).difference(
        expt_seed_to_row.keys())
    if verbose and base_only_seeds:
      print(f'  Warning: {len(base_only_seeds)} seeds found on base side only. '
            f'These will be omitted from the analysis: {str(base_only_seeds)}')
    expt_only_seeds = set(expt_seed_to_row.keys()).difference(
        base_seed_to_row.keys())
    if verbose and expt_only_seeds:
      print(f'  Warning: {len(expt_only_seeds)} seeds found on expt side only. '
            f'These will be omitted from the analysis: {str(expt_only_seeds)}')
  else:
    base_seeds = list(sorted(base_seed_to_row.keys()))
    expt_seeds = list(sorted(expt_seed_to_row.keys()))
    if verbose:
      print(f'Multibootstrap (unpaired) on {num_examples} examples')
      print(f'  Base seeds ({len(base_seeds)}): {str(base_seeds)}')
      print(f'  Expt seeds ({len(expt_seeds)}): {str(expt_seeds)}')

  def _score_seeds(seeds_to_rows, seeds,
                   row_score_fn):
    """Helper to score a single bootstrap sample."""
    scores = []
    for s in seeds:
      ii = seeds_to_rows[s]
      seed_scores = [row_score_fn(i) for i in ii]
      scores.append(np.mean(seed_scores))  # mean over finetuning params, etc.
    return np.mean(scores)  # mean over seeds

  # Compute bootstrap samples
  bootstrap_samples = []
  for _ in progress_indicator(range(nboot)):
    # Sample seeds; we'll map these to row indices later.
    if paired_seeds:
      base_seed_sample = _bsample(common_seeds, rng)
      expt_seed_sample = base_seed_sample
    else:
      base_seed_sample = _bsample(base_seeds, rng)
      expt_seed_sample = _bsample(expt_seeds, rng)

    # Sample examples and select columns
    jj = rng.choice(num_examples, size=num_examples, replace=True)
    selected_labels = labels[jj]
    # pylint note: safe to close over jj, because this function is also
    # redefined on every loop iteration.
    score_row = lambda i: score_fn(selected_labels, preds[i, jj])  # pylint: disable=cell-var-from-loop

    # Select and score rows corresponding to selected seeds.
    base_score = _score_seeds(base_seed_to_row, base_seed_sample, score_row)
    expt_score = _score_seeds(expt_seed_to_row, expt_seed_sample, score_row)

    # Store this sample as a pair.
    bootstrap_samples.append((base_score, expt_score))

  return bootstrap_samples


def report_ci(samples,
              c = 0.95,
              abs_threshold = None,
              rel_threshold = None):
  """Report confidence interval and some other statistics from bootstrap samples.

  Note: abs_threshold and rel_threshold assume the effect of the intervention
  is positive; if this is not true you'll likely see a value close to zero.
  If you want to compute something like E[L' <= L - delta], you can switch
  base and expt in the input to this function using something like:
    samples = [(b,a) for (a,b) in samples]

  Note: rel_threshold is a good approximation only if L is not close to 0.

  Args:
    samples: List[Tuple[float, float]] as returned by multibootstrap()
    c: desired confidence level
    abs_threshold (optional): if given, will also report probability that the
      intervention has a positive effect of at least this value.
    rel_threshold (optional): if given, will also report probability that the
      intervention has a relative effect of at least this threshold. For
      instance, rel_threshold=0.01 will give the estimated confidence that the
      intervention improves by 1% over the baseline.

  Returns:
    stats as a dict
  """
  quantiles = [(1 - c) / 2, 1 - (1 - c) / 2]  # 95% -> [2.5%, 97.5%]
  ret = {}
  bdf = pd.DataFrame(samples, columns=['base', 'expt'])
  print(f'Bootstrap statistics from {len(bdf)} samples:')

  bu = bdf.base.mean()
  bmin, bmax = np.quantile(bdf.base, quantiles)
  print(f'  E[L]  = {bu:.3g} with {c:.0%} CI of ({bmin:.3g} to {bmax:.3g})')
  ret['base'] = {'mean': bu, 'low': bmin, 'high': bmax}

  eu = bdf.expt.mean()
  emin, emax = np.quantile(bdf.expt, quantiles)
  print(f"  E[L'] = {eu:.3g} with {c:.0%} CI of ({emin:.3g} to {emax:.3g})")
  ret['expt'] = {'mean': eu, 'low': emin, 'high': emax}

  deltas = bdf.expt - bdf.base
  u = np.mean(deltas)
  umin, umax = np.quantile(deltas, quantiles)
  pval = np.mean(deltas <= 0)
  print(f"  E[L'-L] = {u:.3g} with {c:.0%} CI of ({umin:.3g} to {umax:.3g})"
        f'; p-value = {pval:.3g}')
  ret['delta'] = {'mean': u, 'low': umin, 'high': umax, 'p': pval}

  if rel_threshold:
    pt = (deltas / bdf.base >= rel_threshold).mean()
    print(f"  E[(L'-L)/L ≥ {rel_threshold:.3g}] = {pt:0.2f}")

  if abs_threshold is not None:
    pt = (deltas >= abs_threshold).mean()
    print(f"  E[L' ≥ L + {abs_threshold:.3g}] = {pt:0.2f}")

  return ret
