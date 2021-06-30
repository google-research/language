# TempLAMA

This folder contains the code for constructing the TempLAMA dataset using [SLING](https://github.com/ringgaard/sling). The dataset is described in the paper [Time-Aware Language Models as Temporal Knowledge Bases](https://arxiv.org/abs/2106.15110).

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
@misc{dhingra2021timeaware,
      title={Time-Aware Language Models as Temporal Knowledge Bases},
      author={Bhuwan Dhingra and Jeremy R. Cole and Julian Martin Eisenschlos and Daniel Gillick and Jacob Eisenstein and William W. Cohen},
      year={2021},
      eprint={2106.15110},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
