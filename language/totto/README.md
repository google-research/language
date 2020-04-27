# ToTTo Dataset

ToTTo is a dataset for the controlled generation of descriptions of tabular data comprising over 100,000 examples. Each example is a aligned pair of a highlighted table and the description of the highlighted content.

During the dataset creation process, tables from English Wikipedia are matched with (noisy) descriptions. Each table cell mentioned in the description is highlighted and the descriptions are iteratively cleaned and corrected to faithfully reflect the content of the highlighted cells.

By providing multiple different descriptions from the same table, this dataset can be utilized as a testbed for the controllable generation of table description.

You can find more details, analyses, and baseline results in [our paper](#). You can cite it as follows:

```
@inproceedings{parikh2020totto,
  title =     {ToTTo: A Controlled Table-To-Text Generation Dataset},
  author =    {XXX},
  booktitle = {XXX},
  year =      {2020},
}
```

## Dataset Description

The ToTTo dataset consists of three `.jsonl` files, where each line is a JSON dictionary with the following format:

```json
{
  "table_page_title": "'Weird Al' Yankovic",
  "table_section_title": "Television",
  "table_section_text": "",
  "table": "[Described below]",
  "highlighted_cells": [[22, 2], [22, 3], [22, 0], [22, 1], [23, 3], [23, 1], [23, 0]],
  "example_id": 12345678912345678912,
  "annotations": [{"original_sentence": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Mr. Peanutbutter's brother, Captain Peanutbutter, and was hired to voice the lead role in the 2016 Disney XD series Milo Murphy's Law.",
                  "sentence_after_deletion": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter, and was hired to the lead role in the 2016 series Milo Murphy's Law.",
                  "sentence_after_ambiguity": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter, and was hired for the lead role in the 2016 series Milo Murphy's 'Law.",
                  "final_sentence": "In 2016, Al appeared in 2 episodes of BoJack Horseman as Captain Peanutbutter and was hired for the lead role in the 2016 series Milo Murphy's Law."}],
}
```

The `table` field is a `List[List[Dict]]`. The outer lists represents rows and the inner lists columns. Each `Dict` has the fields `column_span: int`, `is_header: bool`, `row_span: int`, and `value: str`. The first two rows for the example above look as follows:

```json
[
  [
    {    "column_span": 1,
         "is_header": true,
         "row_span": 1,
         "value": "Year"},
    {    "column_span": 1,
         "is_header": true,
         "row_span": 1,
         "value": "Title"},
    {    "column_span": 1,
         "is_header": true,
         "row_span": 1,
         "value": "Role"},
    {    "column_span": 1,
         "is_header": true,
         "row_span": 1,
         "value": "Notes"}
  ],
  [
    {    "column_span": 1,
         "is_header": false,
         "row_span": 1,
         "value": "1997"},
    {    "column_span": 1,
         "is_header": false,
         "row_span": 1,
         "value": "Eek! The Cat"},
    {    "column_span": 1,
         "is_header": false,
         "row_span": 1,
         "value": "Himself"},
    {    "column_span": 1,
         "is_header": false,
         "row_span": 1,
         "value": "Episode: 'The FugEektive'"}
  ], ...
]
```

To help understand the dataset, you can find a sample of the train and dev sets in the `sample/` folder. We additionaly provide the `create_table_to_text_html.py` script that visualizes an example, the output of which you can also find in the `sample/` folder.


### Dev and Test Set

The dev and test set have three references for each example, which are added to the list at the `annotations` key. The test set annotations are *private* and thus not included in the data. If you want us to evaluate your model on the private test set, please email us at `totto@google.com`. By emailing us, you consent to being contacted about this dataset.

We provide two splits within the dev and test sets - one uses previously seen combinations of table headers and one uses unseen combinations. The sets are marked using the `overlap_subset: bool` flag that is added to the JSON representation. By filtering the evaluation to examples with the flag set to `true`, you will be able to test the generalization ability of your model.

****

# Leaderboard

We are maintaining a leaderboard with official results on our blind test set:

<table>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">Overall</th>
    <th colspan="2">Overlap Subset</th>
    <th colspan="2">Non-Overlap Subset</th>
  </tr>
  <tr>
    <th></th>
    <th>Model</th>
    <th>Link</th>
    <th>Uses Wiki</th>
    <th>BLEU</th>
    <th>PARENT</th>
    <th>BLEU</th>
    <th>PARENT</th>
    <th>BLEU</th>
    <th>PARENT</th>
  </tr>
  <tr>
    <td>1.</td>
    <td>BERT-to-BERT (Wiki+Books)</td>
    <td><a href="https://arxiv.org/abs/1907.12461">[Rothe et al., 2019]</a></td>
    <td>yes</td>
    <td><b>44.0</b></td>
    <td><b>52.6</b></td>
    <td><b>52.7</b></td>
    <td><b>58.4</b></td>
    <td><b>35.1</b></td>
    <td><b>46.8</b></td>
  </tr>
  <tr>
    <td>2.</td>
    <td>BERT-to-BERT (Books)</td>
    <td><a href="https://arxiv.org/abs/1907.12461">[Rothe et al., 2019]</a></td>
    <td>no</td>
    <td>43.9</td>
    <td><b>52.6</b></td>
    <td><b>52.7</b></td>
    <td><b>58.4</b></td>
    <td>34.8</td>
    <td>46.7</td>
  </tr>
  <tr>
    <td>3.</td>
    <td>Pointer Generator</td>
    <td><a href="https://www.aclweb.org/anthology/P17-1099/">[See et al., 2017]</a></td>
    <td>no</td>
    <td>41.6</td>
    <td>51.6</td>
    <td>50.6</td>
    <td>58.0</td>
    <td>32.2</td>
    <td>45.2</td>
  </tr>
  <tr>
    <td>4.</td>
    <td>Content Planner</td>
    <td><a href="https://www.aaai.org/ojs/index.php/AAAI/article/view/4668">[Puduppully et al., 2019]</a></td>
    <td>no</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## Leaderboard Submission

If you want to submit test outputs, please format your predictions as a single `.txt` file with line-separated predictions. The predictions should be in the same order as the examples in the `test.jsonl` file.
Please email the prediction file to `totto@google.com` and state that you want your result to be included in the leaderboard. By emailing us, you consent to being contacted about this dataset.

## Getting Started
**Download the TOTTO data**

```
 wget URL
```

**Run the evaluation scripts locally**

To encourage comparability of results between different systems, we encourage researchers to evaluate their systems using the scripts provided in this repository. For an all-in-one solution, you can call `totto_eval.sh` with the following arguments:

- `--prediction_path`: Path to your model's predictions, one prediction text per line. [Required]
- `--example_path`: Path to the `public_dev_data.jsonl` you want to evaluate. [Required]
- `--output_dir`: where to save the downloaded scripts and formatted outputs. [Default: `./temp/`]

`totto_eval.sh` requires python 3 and a few libraries. Please make sure that you have all the necessary libraries installed. You can use your favorite python environment manager (e.g., virtualenv or conda) to install the requirements listed in `eval_requirements.txt`.

You can test whether you are getting the correct outputs by running it on our provided development samples in the `sample/` folder, which also contains associated `sample_outputs.txt`. To do so, please run the following command:

```
totto_eval.sh --prediction_path sample/sample_outputs.txt --example_path sample/dev_sample.jsonl
```

You should see the following output:

```
======== EVALUATE OVERALL ========
Computing BLEU (overall)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 76.0 91.9/79.0/71.1/64.8 (BP = 1.000 ratio = 1.024 hyp_len = 86 ref_len = 84)
Computing PARENT (overall)
Evaluated 5 examples.
Precision = 0.9463 Recall = 0.7051 F-score = 0.8062
======== EVALUATE OVERLAP SUBSET ========
Computing BLEU (overlap subset)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 68.0 90.0/71.4/61.5/54.2 (BP = 1.000 ratio = 1.071 hyp_len = 30 ref_len = 28)
Computing PARENT (overlap subset)
Evaluated 2 examples.
Precision = 0.9696 Recall = 0.7558 F-score = 0.8494
======== EVALUATE NON-OVERLAP SUBSET ========
Computing BLEU (non-overlap subset)
BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.7 = 80.1 92.9/83.0/76.0/70.2 (BP = 1.000 ratio = 1.000 hyp_len = 56 ref_len = 56)
Computing PARENT (non-overlap subset)
Evaluated 3 examples.
Precision = 0.9308 Recall = 0.6713 F-score = 0.7775
```

**Test the evaluation result**

If you want to ensure that the results from totto_eval.sh are as expected, please run `pytest` inside of this folder. This will run the tests provided in the `test_eval_pipeline.py` file.
