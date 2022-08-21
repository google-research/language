# FRUIT: Faithfully Reflecting Updated Information in Text

This is the release for the paper: [FRUIT: Faithfully Reflecting Updated Information in Text](https://arxiv.org/abs/2112.08634).


## News

* (5/12/2022) The data is available now, and the evaluation scripts will be released later.

## FRUIT-Wiki Dataset

### Download

To download the FRUIT-Wiki dataset, run:

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

## Raw Files

Files with name such as article_pairs.update.jsonl-?????-of-00251 are generated using our data extraction pipeline. Each line is a json object describing the source article, the target article and the entity annotations we used to compute the context. The raw files are only in the train folder and the test folder, given the gold_test folder is a subset of the test folder where we clean the data even further with human annotations.

The main purpose of the raw files is for the users who want to create a different version of the input/output files than the EdiT5 processed files.

## Data Collection Pipeline

To enable researchers to apply our data collection pipeline to future Wikipedia snapshots, this repository also contains our data processing code. Please refer to the [README_PIPELINE.md](./README_PIPELINE.md) file for detailed instructions on how to run the different pipeline steps.


## Eval Scripts without t5x

The NER related metric will be released later.

For t5x users, simply load the task  and evalution for UpdateRouge should work out-of-box.

### Step 1: Generate jsonl files.

    Run convert_task_to_jsonl.py

    The typical use case will consider only three possible combinations.

    To output the validation set (from "Nov. 20, 2019" and "Nov. 20, 2020")

    --task_name="wikidiff_diff_all_text_reference" --split="validation"

    To output the test set (from "Nov. 20, 2020" and "Nov. 20, 2021").

    --task_name="wikidiff_diff_all_text_reference_test" --split="test"

    To print out the gold test (from "Nov. 20, 2020" and "Nov. 20, 2021",
       verified by annotators)

    --task_name="wikidiff_diff_all_text_reference_gold_test" --split="test"

    ** Output **

    This script will be used for output two jsonl files.

    {output_prefix}_inputonly.jsonl

        The input only files will contain the input of EdiT5 for the
        chosen split. This file is used for genearting the prediction
        using your models.

    {output_prefix}_inputlabels.jsonl

        The input and labels file contains both the input and the
        targeted output.

### Step 2: Apply your own model to generate output

    Every line in {output_prefix}_inputonly.jsonl is the input example and your
    model should try to generate the output. The output format should be like
    the one in scripts/sample_data/pred.json.

### Step 3: Run evaluation script

    Use scripts/evaluate_direct_jsonls.py with the following arguments
    --input_labels_jsonl={output_prefix}_inputlabels.jsonl
    --prediction_jsonl=pred.jsonl
    --task_name=wikidiff_diff_all_text_reference

    to get the final results.

## Evaluating with t5x

To run evaluation with t5x first install the dependencies. This involves following the instructions in [t5x](https://github.com/google-research/t5x), and pip installing rouge_score and tqdm.

Then an example evaluation run is
```
 python t5x/t5x/eval.py \
   --gin_file language/fruit/t5x/configs/t5_large_eval.gin \
   --gin.MIXTURE_OR_TASK_NAME=\"wikidiff_diff_all_text_reference_test\" \
   --gin.CHECKPOINT_PATH=\"$(pwd)/checkpoint\" \
   --gin.EVAL_OUTPUT_DIR=\"/tmp/eval\" \
   --gin.DROPOUT_RATE=0
```
