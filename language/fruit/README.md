# FRUIT: Faithfully Reflecting Updated Information in Text

This is the release for the paper: [FRUIT: Faithfully Reflecting Updated Information in Text](https://arxiv.org/abs/2112.08634).


## News

* (5/12/2022) The data is available now, and the evaluation scripts will be released later.

## Dataset

### Download

To download the FRUIT dataset, run:

```
mkdir fruit_dataset
gsutil cp -R gs://gresearch/FRUIT/dataset fruit_dataset
```

Note: this requires [gsutil](https://cloud.google.com/storage/docs/gsutil).

There are three subfolers: train, test, gold_test. Note that the train folders contain train and dev sets.

* The train folder contains the files that are constructed using the snapshots from "Nov. 20, 2019" and "Nov. 20, 2020".
* The test folder contains the files that are constructed using the snapshots from "Nov. 20, 2020" and "Nov. 20, 2021".
* The gold_test folder constructed using the same time period as the test folder, but the edits are verified with the human annotators (containing only processed file, as explained below).

### Processed Examples (used in EdiT5)

The processed file are with the named such as
article_pairs.update.diff.all.text.reference.tfrecords-00000-of-00010 is a
[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) file.
In the train folder, we use the first 9 files as the training set and the 10th file (00009) file as our
development set. There are processed files in the test and gold_test folders as well.

The fields of the processed files are as the follows:

 * id: id of the example
 * input: input to the EdiT5 examples.

 The input sequence contains the source version of the article and the new evidence from the other pages. An example is shown below:

```
[0] Aoraki / Mount Cook, often referred to as Mount Cook Village, is located within New Zealand's ... [CONTEXT] (0) Aoraki_/_Mount_Cook_National_Park INTRODUCTION Aoraki / Mount Cook, New Zealand's highest mountain, and Aoraki/Mount Cook Village lie within the park. (1) ...
```

Each sentence in the source article has a marker with a pair of square brackets (e.g. [1]). Later there is a context separator ([CONTEXT]). Each context has a marker with a pair of round brackets (e.g. (0)).

 * output: The output of the EdiT5 example. For example,

 The output of EdiT5 contains the extra marker to help the model to focus on updating the article and the referencing the context. For example,

```
(0) (1) Aoraki / Mount Cook, often referred to as Mount Cook Village, is located within New Zealand's Aoraki / Mount Cook National Park at the end of ... [2] [3] [4]
```

This means that the first sentence is updated using the context item (0) and (1). Note that the reference is calculated heuristically (excepted for the gold_test data). '[2] [3] [4]' means that the these sentences are copied directly from the source article.

** Raw file

Files with name such as article_pairs.update.jsonl-?????-of-00251 are generated using our data extraction pipeline. Each line is a json object describing the source article, the target article and the entity annotations we used to compute the context. The raw files are only in the train folder and the test folder, given the gold_test folder is a subset of the test folder where we clean the data even further with human annotations.

The main purpose of the raw files is for the users who want to create a different version of the input/output files than the EdiT5 processed files.

## Eval Scripts

The eval scripts will be release later.
