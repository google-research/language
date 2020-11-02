# ToTTo Supplementary Repository

This code repository serves as a supplementary for the main repository. The main repository can be found [here](https://github.com/google-research-datasets/ToTTo/).

## ToTTo Dataset

ToTTo is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description.

During the dataset creation process, tables from English Wikipedia are matched with (noisy) descriptions. Each table cell mentioned in the description is highlighted and the descriptions are iteratively cleaned and corrected to faithfully reflect the content of the highlighted cells.

We hope this dataset can serve as a useful research benchmark for high-precision conditional text generation.

You can find more details, analyses, and baseline results in [our paper](https://arxiv.org/abs/2004.14373). You can download the data from the [main repository](https://github.com/google-research-datasets/ToTTo/) and cite our paper as follows:

```
@article{parikh2020totto,
  title={ToTTo: A Controlled Table-To-Text Generation Dataset},
  author={Parikh, Ankur P and Wang, Xuezhi and Gehrmann, Sebastian and Faruqui, Manaal and Dhingra, Bhuwan and Yang, Diyi and Das, Dipanjan},
  journal={arXiv preprint arXiv:2004.14373},
  year={2020}
```

## Clone the repository
First clone the research language repository and `cd` into that directory. All
commands should be run from that location.

```
git clone https://github.com/google-research/language.git language_repo
cd language_repo
```

## Prerequisites

The code requires python 3 and a few libraries. Please make sure that you have all the necessary libraries installed. You can use your favorite python environment manager (e.g., virtualenv or conda) to install the requirements listed in `eval_requirements.txt`.
```
pip3 install -r language/totto/eval_requirements.txt
```



## Visualizing sample data

To help understand the dataset, you can find a sample of the train and dev sets in the `sample/` folder. We additionally provide the `create_table_to_text_html.py` script that visualizes an example, the output of which you can also find in the `sample/` folder.

A sample command is given below:

```
python3 -m language.totto.create_table_to_text_html --input_path="language/totto/sample/train_sample.jsonl" --output_dir="."
```

## Running the evaluation scripts locally

To encourage comparability of results between different systems, we encourage researchers to evaluate their systems using the scripts provided in this repository. For an all-in-one solution, you can call `totto_eval.sh` with the following arguments:

- `--prediction_path`: Path to your model's predictions, one prediction text per line. [Required]
- `--target_path`: Path to the `totto_dev_data.jsonl` you want to evaluate. [Required]
- `--output_dir`: where to save the downloaded scripts and formatted outputs. [Default: `./temp/`]


You can test whether you are getting the correct outputs by running it on our provided development samples in the `sample/` folder, which also contains associated `sample_outputs.txt`. To do so, please run the following command:

```
bash language/totto/totto_eval.sh --prediction_path language/totto/sample/output_sample.txt --target_path language/totto/sample/dev_sample.jsonl
```
(If you get an error regarding `getopt` it is likely that you need to install `gnu-getopt`.)

You should see the following output:

```
======== EVALUATE OVERALL ========
Computing BLEU (overall)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 45.5 86.0/63.5/44.7/31.0 (BP = 0.869 ratio = 0.877 hyp_len = 57 ref_len = 65)
Computing PARENT (overall)
Evaluated 5 examples.
Precision = 0.7611 Recall = 0.4383 F-score = 0.5334
======== EVALUATE OVERLAP SUBSET ========
Computing BLEU (overlap subset)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 37.2 84.8/56.7/37.0/25.0 (BP = 0.809 ratio = 0.825 hyp_len = 33 ref_len = 40)
Computing PARENT (overlap subset)
Evaluated 3 examples.
Precision = 0.7140 Recall = 0.3135 F-score = 0.4134
======== EVALUATE NON-OVERLAP SUBSET ========
Computing BLEU (non-overlap subset)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 58.3 87.5/72.7/55.0/38.9 (BP = 0.959 ratio = 0.960 hyp_len = 24 ref_len = 25)
Computing PARENT (non-overlap subset)
Evaluated 2 examples.
Precision = 0.8317 Recall = 0.6256 F-score = 0.7135
```

## Testing the evaluation result

If you want to ensure that the results from `totto_eval.sh` are as expected, please run:

```
python3 -m language.totto.eval_pipeline_test
```

## Computing the BLEURT score.

First install BLEURT from [here](https://github.com/google-research/bleurt). For the leaderboard we used the BLEURT-base 128 checkpoint.

To run the BLEURT as part of the evaluation script add an additional argument for the BLEURT checkpoint path:

```
bash language/totto/totto_eval.sh --prediction_path language/totto/sample/output_sample.txt --target_path language/totto/sample/dev_sample.jsonl --bleurt_ckpt <BLEURT checkpoint path>
```

Note that running BLEURT is considerably slower than the other metrics, and is faster on GPU.


## Baseline preprocessing
For reproducibility, we supply our basic table linearization code used for the baselines in the paper. The code takes as input a jsonl file and will augment each json example with additional fields: `full_table_str`, `subtable_str`, `full_table_metadata_str`, and `subtable_metadata_str` for each of the table linearizations described in the paper.

Please note that given the complexity of some of the tables in our dataset (e.g. cells that span multiple rows and columns), many aspects of this preprocessing code (such as header extraction or the linearization scheme) may not be optimal. Developing new techniques of effectively representing open domain structured input is a part of the challenge of this dataset.

A sample command is given below:

```
python3 -m language.totto.baseline_preprocessing.preprocess_data_main --input_path="language/totto/sample/train_sample.jsonl" --output_path="./processed_train_sample.jsonl"
```
