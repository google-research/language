# ASQA: Long-Form Answers for Ambiguous Open-domain Factoid Questions
Ivan Stelmakh (CMU), Yi Luan (Google Research), Bhuwan Dhingra (Google Research, Duke), Ming-Wei Chang (Google Research)

## Abstract
ASQA is the first long-form question answering dataset that focuses on ambiguous factoid questions. Different from previous long-form answers datasets, each question is annotated with both long-form answers and extractive question-answer pairs, which should be answerable by the generated passage. A generated long-form answer will be evaluated using both ROUGE and QA accuracy. We showed that these evaluation metrics correlated with human judgment well.
In this repostory we release the ASQA dataset, together with the evaluation code.


## ASQA Dataset
### Download
To download the ASQA dataset, run:

```
mkdir dataset
gsutil cp -R gs://gresearch/ASQA/ASQA.json dataset
```

Note: this requires [gsutil](https://cloud.google.com/storage/docs/gsutil).

### Format
The data consists of two subsets: train and dev. Each subset is a dictionary of the form {key: instance} where "key" is the id of the corresponding AmbigQA instance and "instance" is the instance of the ASQA dataset.

Each instance is a dictionary with the following attributes:

1. "ambiguous_question". The original Ambiguous Question of the ASQA dataset.
2. "qa_pairs". List of disambiguated QA pairs from the AmbigQA dataset enhanced with the additional context we construct. Each entry in the list corresponds to a disambiguated QA pair from AmbigQA and is represented by the following dictionary: {"question": disambiguated question from AmbigQA, "short_answers": list of short answers from AmbigQA, "context": additional context we provide, "wikipage": title of the Wikipedia page the additional context was taken from}.
3. "wikipages". List of Wikipedia pages visited by AmbigQA annotators. Each entry in the list is a dictionary of the following structure: {"title": title of the Wikipedia page, "url": link to the Wikipedia page}.
4. "annotations". One (for "train" set) or two (for "dev" set) long-form answers to the ambiguous question constructed by ASQA annotators. Each annotation is a dictionary of the following structure: {"long_answer": annotation, "knowledge": list of additional knowledge pieces}. Each piece of additional knowledge is represented by a separate dictionary: {"content": a passage from Wikipedia, "wikipage": title of the Wikipedia page the passage was taken from}.

Note: We do not release the questions from the test set of ASQA as these questions would reveal a subset of questions from the AmbigQA test set which are not publicly available.

Note: A leaderboard will be set up soon. Stay tuned.

### Annotation guideline
The annotation guideline is provided in [this slide](https://drive.google.com/drive/folders/1IS8J7fBlYvD7a1GrQlx85Ls6Y6mlcKje?usp=sharing).

## Automatic Evaluation
Given predicted long answers from a model it is straightforward to compute ROUGE
by comparing to the reference answers, but for computing the QA accuracy we need
run a separate reading comprehension model. For this we use huggingface's implementation
of Roberta-large trained on Squad 2.0 and fine-tuned on the ASQA training set.

All codes under this repository are compatible with Python 3, and a Unix-like system.

### Setup
1. You might want to setup a virtual environment before installation.

2. Install PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/#start-locally).

3. Install python packages and download the Roberta checkpoint by running:

```
sh install.sh
```

### Evaluation in one bash script
```
chmod +x ./eval.sh
./eval.sh ${RESULTS_PATH} ${EXP_NAME}
```
The final results will show in screen and will also be generated in ./results/${EXP_NAME}/final_eval_results.json.

### Break down of each evaluation step
The following provides the evaluation guideline for each step:
#### STEP-1: Convert your system output to Roberta input format
First, save your system output into key, value pairs in json format. Refer to the following example for format.

```
mkdir outputs
gsutil cp -R gs://gresearch/ASQA/outputs/t5_predictions.json outputs
RESULTS_PATH=./outputs/t5_predictions.json

EXP_NAME=t5
OUTPUT_DIR=./results/${EXP_NAME}
mkdir -p ${OUTPUT_DIR}
```

Then run the following script to convert to Roberta input format:

```
python convert_to_roberta_format.py  \
  --asqa ./dataset/ASQA.json \
  --predictions ${RESULTS_PATH}  \
  --split dev \
  --output_path $OUTPUT_DIR
```

The output question-context pairs for Roberta will be stored at `./outputs/qa.json`.

#### STEP-2: Run the Roberta Squad 2.0 inference
Next run the following command which uses an example script from the
transformers library to get QA predictions from Roberta:

```
python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path ./roberta/roberta-squad \
  --validation_file ${OUTPUT_DIR}/qa.json \
  --do_eval \
  --version_2_with_negative \
  --max_seq_length 384 \
  --output_dir $OUTPUT_DIR \
  --null_score_diff_threshold 0
```

Note: This assumes that the transformers library was cloned from github using
the install script above. If not, you can download `run_qa.py` from [here](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py).

Note: The above command takes ~30 mins to run on a CPU. This can be
significantly sped up if you have access to a GPU(s).

#### STEP-3: Take the Roberta output and run evaluation
The following command will compute the final evaluation metrics:

```
python scoring.py \
  --asqa ./dataset/ASQA.json \
  --predictions ${OUTPUT_DIR}/t5_predictions.json \
  --roberta_output ${OUTPUT_DIR}/eval_predictions.json \
  --split dev
  --out_dir $OUTPUT_DIR
```

On the `t5_predictions.json` provided above you should get the following results:

```
{
  "rougeLsum": 39.151834028242725,
  "length": 71.63185654008439,
  "str_em": 41.03902953586498,
  "QA-EM": 20.627637130801688,
  "QA-F1": 26.373514957680307,
  "QA-Hit": 5.274261603375527,
  "ovscore": 32.13365028073338
}
```

## Human Evaluation
To support human comparison, we release several scripts that streamline the process. The annotation itself happens in a Google spreadsheet, but some pre- and post-processing work needs to be done locally. See `screenshot.png` for a screenshot of the annotation interface. File `instructions.txt` provides instructions for annotators.
All of the scripts are provided under the `human_annotation` folder.

**Note**: You will need 'pandas' for running the scripts below.


### STEP-1: Preperation ###

To begin, you need to create a tab-separated setup file that has all the information necessary for pairwise comparisons. Specifically, for each pairwise comparison you want to make, the file should contain a single line that has the following information (each item corresponds to a column, see `setup.tsv` for an example):

1. Key of the corresponding ASQA instance.

2. Name of Model 1. Name of the model that generates the first (left) prediction for the pairwise comparison

3. Name of Model 2. Name of the model that generates the second (right) prediction for the pairwise comparison

4. Prediction of Model 1. Long-form answer to the ambiguous question generated by Model 1

5. Prediction of Model 2. Long-form answer to the ambiguous question generated by Model 2

After you construct this file, run the `preparation.py` script as follows:

```
cd ./human_annotation
python preparation.py --asqa ../Data/ASQA.json --setup ./setup.tsv --dst ./ready_for_drive.tsv
```
This script will create the `ready_for_drive.tsv` file that you can use in Step 2.


### Step-2: Interface creation ###

Having the `ready_for_drive.tsv` prepared, execute the following steps:

0. Create a folder in your Google Drive and go inside this folder (GF)
1. Upload the `ready_for_drive.tsv` file to Google Drive and open it as a Google Sheet (GS)
2. Back in the drive folder (GF) create a Google Apps Script (new -> More -> Google Apps Script)
3. Add Drive API to the Apps Script (open script -> Services -> Drive API -> Add)
4. Copy content of the `prepare_interface.gs` into the `Code.gs` file in the Apps Script (replace the existing placeholder code).
5. Add file id of GS (created in Step 1) to line 4 and folder id of GF (created in Step 0) to line 7 of Code.gs. See the script for instructions on how to get these ids.
6. Execute Code.gs on Google Drive (make sure you first save the project and then select the "main" function in the menu on top)

As a result of these steps, the parent folder (created in Step 0) will contain the interface for human annotation.

### Step-3: Annotation ###

Open the 'AnnotationInterface' google sheet which should now be in your drive folder (GF) and proceed to the do the annotation. We have provided instructions for the annotators for this part in 'instructions.txt'.

### Step-4: Analysis ###

After annotations are completed, download the spreadsheet as an excel (.xlsx) file and execute analysis.py script:

```
python analysis.py --setup ./setup.tsv --comparisons ./AnnotationInterface.xlsx --dst ./resuts_of_comparisons.tsv
```
The script will create the `resuts_of_comparisons.tsv` file with the results of pairwise comparisons. Each line in this file will summarize the result of a single pairwise comparison. This file can then be used for subsequent custom analysis. The resulting file will contain the following information:

* pair -- index of comparison
* numQA -- number of QA pairs associated with the corresponding ASQA instance
* accLeft -- fraction of QA pairs captured in the answer shown on the left (by leftModel)
* accRight -- fraction of QA pairs captured in the answer shown on the right (by rightModel)
* Ambiguity -- result of the disambiguation comparison
* Fluency -- result of the fluency comparison
* Overall -- result of the overall comparison
* leftModel -- name of the left model (prediction shown on the left)
* rightModel -- name of the right model (prediction shown on the right)
* key -- key of the corresponding ASQA instance
