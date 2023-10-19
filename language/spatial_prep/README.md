# A Benchmark for Reasoning with Spatial Prepositions

This folder contains the dataset described in the paper
 "A Benchmark for Reasoning with Spatial Prepositions" (EMNLP 2023).

## Download the data

* [English dataset](https://storage.googleapis.com/spatial-prepositions-dataset/spatial_prepositions_benchmark_en.tsv)
* [Romanian dataset](https://storage.googleapis.com/spatial-prepositions-dataset/spatial_prepositions_benchmark_ro.tsv)

## Data format

The datasets are provided in TSV (tab-separated) format.
 Each row contains an incongruent and a congruent example, both formed using
 the same prepositions, with the following tab-separated elements:
 `premise1_a`, `premise2_a`, `conclusion_a`,
 `no` (language-specific, indicating that `conclusion_a` is invalid),
 `premise1_b`, `premise2_b`, `conclusion_b`,
 `yes` (language-specific, indicating that `conclusion_b` is valid).

## Building examples

From each example, a question can be created by combining the two premises
 and the conclusion, as follows:
 "If `premise_1` and `premise_2`, does that imply that `conclusion`?"

The examples with invalid conclusions are designed such that a wrong
 interpretation of the spatial prepositions in the premises can make the
 conclusion appear valid. For example:

* If `John is in the crib` and `the crib is in the living room`,
 does that imply that `John is in the living room`? -> `yes`
 (congruent example with valid conclusion)

* If `John is in the newspaper` and `the newspaper is in the kitchen`,
 does that imply that `John is in the kitchen`? -> `no`
 (incongruent example with invalid conclusion)


## Citation

If you use this data, please cite:

```
@inproceedings{comsa2023,
    author = {Comșa, Iulia-Maria and Narayanan, Srini},
    title = "{A Benchmark for Reasoning with Spatial Prepositions}",
    publisher = "Association for Computational Linguistics",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = {2023},
}
```
