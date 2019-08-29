## Consistent Zero-shot NMT

### Installation

This code release has the following dependencies:

- `tensorflow==1.13.1` or `tensorflow-gpu==1.13.1`
- `tensor2tensor==1.11.0`

Make sure that the version of `tensor2tensor` library is `1.11.0` as the code
is not compatible with the later versions due API changes.

You can install the `language` package in your system (or virtual environment)
with all necessary dependencies by running the following command from the root
directory of this repository:

```bash
$ pip install -e ".[consistent-zero-shot-nmt]"
```

### Data preprocessing

Follow the instructions below to obtain preprocessed versions of the datasets
in the TFRecord format. To understand how the TFRecords entries are structured,
take a look at `data_generators/translate_multilingual.py`.

#### How to obtain TFRecords for IWSLT17*

IWSLT17* is a preprocessed/clean version of IWSLT17 introduced in the paper.
The official dataset had a large number of multi-parallel sentences in the
training data that resulted in a variety of spurious effects and made proper
evaluation of zero-shot performance of models challenging. To construct IWSLT17*,
we identify identical sentences across different training parallel corpora and
remove duplicates from the dataset.
 
Download and extract the official dataset to `/tmp/data/iwslt17-star/original`
and identify overlaps between the training corpora using the following script:
```bash
$ scripts/download_and_preproc_iwslt17.sh --data-dir=/tmp/data --dataset-name=iwslt17-star
```
Generate TFRecords that will be used for training models as follows:
```bash
$ scripts/datagen_iwslt17.sh \
    --data-dir=/tmp/data \
    --dataset-name=iwslt17-star \
    --problem-name=translate_iwslt17_nonoverlap
```
**Note:** To generate TFRecords for the official IWSLT17 dataset (without any cleaning),
simply change `--problem-name` argument to `translate_iwslt17`. 

#### Preprocessing of other datasets

To download and preprocess other datasets, follow the same steps as for IWSLT17*
above but use the corresponding scripts `scripts/download_and_preproc_<dataset_name>.sh`
and `scripts/datagen_<dataset_name>.sh`.

### Training and evaluation

Once TFRecords for the problem of interest have been generated, train different
models using one of the provided `run_[basic|agreement]_nmt_<dataset>.sh` scripts.
E.g., to train a basic multilingual NMT model on IWSLT17*, run the following:
```bash
$ scripts/run_nmt_experiment.sh \
    --data-dir=/tmp/data \
    --dataset-name=iwslt17-star \
    --model-name=basic_multilingual_nmt \
    --problem-name=translate_iwslt17_nonoverlap \
    --results-dir=/tmp/results
```
**Note:** Make sure to edit the script and adjust parameters as desired
(e.g., the number of training steps, logging frequency, etc.).
Different named model configurations are defined in `models/basic.py` and
`models/agreement.py`, respectively. 

### Citation

```bibtex
@inproceedings{alshedivat2019consistency,
  title={Consistency by Agreement in Zero-zhot Neural Machine Translation},
  author={Al-Shedivat, Maruan and Parikh, Ankur},
  booktitle={
    Proceedings of the 2019 Conference of the 
    North American Chapter of the Association for Computational Linguistics: 
    Human Language Technologies, Volume 1 (Long and Short Papers)
  },
  pages={1184--1197},
  year={2019},
}
```