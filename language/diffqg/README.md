# DiffQG: Generating Questions to Summarize Factual Changes
DiffQG is a dataset that consists of questions that summarize the factual change
between two versions of the same Wikipedia passage. This requires detecting
whether a factual change occurred, given a noun phrase, and if so, generating
a question that summarizes it.

DiffQG is a small, evaluation-only dataset consisting of expert annotations.
More details can be found in our EACL paper (to appear).


## Installation
The install_and_test.sh file will install all dependencies (including BLEURT),
as well as download the data.

```
git clone https://github.com/google-research/language.git
bash language/language/diffqg/install_and_test.sh language
```

If it is working correctly, all metrics should display.

## Data
Gold annotations can be found here:
https://storage.googleapis.com/gresearch/diffqg/gold_annotations.jsonl

Example predictions can be found here:
https://storage.googleapis.com/gresearch/diffqg/sample_output.jsonl

Predicted annotations should have the following fields:
q: Empty string or generated questions
a: Answer to the (potential) question
base: Base Passage Text
target: Target Passage Text

Note that 'base', 'target', and 'a' must exactly match the gold annotation file.

## Running the Evaluation
In general, the main entry point to the file is run_metrics.py.

Sample Invocation:
```
python3 -m language.diffqg.run_metrics \
  --gold_annotations=language/diffqg/data/gold_annotations.jsonl \
  --predicted_annotations=language/diffqg/data/model1_annotations.jsonl \
  --output_scores=language/diffqg/data/model1_scores.txt \
  --output_metrics=language/diffqg/data/model1_metrics.txt \
  --bleurt_checkpoint=language/diffqg/bleurt/BLEURT-20 \
  --run_qsim=True
```

Gold and predicted file paths must be specified.

Note that the models we use should be able to run on a CPU, but will be slow!
They should be able to run with batch size 1 on a single V100, however.

## How to Cite
If you found this data or code helpful, please cite:
```
@inproceedings{cole2023diffqg,
  title = {DiffQG: Generating Questions to Summarize Factual Changes},
  author = {Cole, Jeremy R. and Jain, Palak and Eisenschlos, Julian and Zhang, Michael J.Q. and Choi, Eunsol and Dhingra, Bhuwan},
  year = {2023},
  booktitle = {Proceedings of EACL}
}
```
