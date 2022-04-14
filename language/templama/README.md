# TempLAMA

This folder contains the code for constructing the TempLAMA dataset using [SLING](https://github.com/ringgaard/sling). The dataset is described in the paper [Time-Aware Language Models as Temporal Knowledge Bases](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00459/110012/Time-Aware-Language-Models-as-Temporal-Knowledge).

**Update**: You can now directly download the preprocessed data at:

```
https://storage.googleapis.com/gresearch/templama/train.json
https://storage.googleapis.com/gresearch/templama/val.json
https://storage.googleapis.com/gresearch/templama/test.json
```

## Prerequisites

You will need to first [install](https://github.com/ringgaard/sling/blob/master/doc/guide/install.md) SLING and then download the wikidata KB and its mapping to wikipedia. This will require around 12G disk space. We have provided a helper script to do these steps for you which can be run as:

```
bash install.sh <path_to_store_sling>
```

Note that the data and experiments described in the paper use the Wikidata dump from November 1, 2020, but these are no longer available for download. Instead the script above downloads the latest dumps from [Ringgaard Research](https://ringgaard.com/) which are updated nightly. So the final version of the TempLAMA data you construct might be slightly different based on the updates to Wikidata since.

## Constructing the data

After installing SLING please provide the path to it in the following script, along with the path where the output files should be placed:

```
bash prepare_data.sh <path_to_sling> <path_to_store_templama>
```

## Citation

If you use this data please cite:

```
@article{dhingra2022time,
    author = {Dhingra, Bhuwan and Cole, Jeremy R. and Eisenschlos, Julian Martin and Gillick, Daniel and Eisenstein, Jacob and Cohen, William W.},
    title = "{Time-Aware Language Models as Temporal Knowledge Bases}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {257-273},
    year = {2022},
    month = {03},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00459},
    url = {https://doi.org/10.1162/tacl\_a\_00459},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00459/2004543/tacl\_a\_00459.pdf},
}
```
