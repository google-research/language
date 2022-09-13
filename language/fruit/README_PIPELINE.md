# Data Collection Pipeline

Our data collection pipeline can be used to produce distantly supervised annotations from pairs of Wikipedia snapshots. The pipeline may also be used to process other sources of data, however this may require modification.

In order to run the pipeline you will need [Apache Beam](https://beam.apache.org/) installed. Apache Beam allows for distributed processing of the input data, and supports a variety of execution engines (e.g., Google Cloud Dataflow, Apache Spark, etc.).

The data collection pipeline is comprised of a sequence of steps that transform the data. Each of these steps has a corresponding script in the scripts/ directory that begins with the word `run_`. Here is a basic overview of each step and the sequence they should be applied:

1. `run_convert_to_jsonl.py`: Converts a Wikipedia dump from XML format to jsonl (which is an easier format for processing with Beam). This step must be run separately on the source and target data.
2. `run_redirect_table_pipeline.py`: Extracts redirects from Wikipedia dump (to ensure accurate entity resolution from wikilinks). This step must be run separately on the source and target data.
3. `run_process_snapshot_pipeline.py`: Processes the pair of snapshots to extract updated article text and entity mentions.
4. `run_filter_for_generation_pipeline.py`: Applies operations to the output of the previous step to adjust text length and filter out unwanted data.
5. `run_to_tfrecords_pipeline.py`: Prepares the output of the previous step for training a particular model. This is where tokenization, insertion of sentinel and control tokens, and conversion to diff format happen.

Each step supports a number of flags that allow the user to customize how the data is processed (use `--help` flag for a brief explanation of each option). Please note that many of the flags were added to explore different approaches for dataset production in early phases of this research and can substantially degrade data quality. For this reason, we recommend using the default settings used to produce the FRUIT-Wiki dataset listed below.


## Example: Processing English Wikipedia Snapshots.

For concrete illustration this section lists the sequence of commands run to produce the FRUIT-Wiki dataset. Please note that these commands will be executed using Beam's DirectRunner runner, which will process the data locally using a single process. The runner and other Beam-related settings can be modified by using `--pipeline_options` flag. We direct the reader to the [Beam documentation](https://beam.apache.org/documentation/) for information on how to set this flag for their use case.

### Setup Python environment

To install the requirements:
```{bash}
conda create --name fruit python=3.7
conda activate fruit
pip install -r language/fruit/requirements.txt
```

Ensure that dependencies were installed correctly by running a unit test:
```{bash}
python -m language.fruit.beam_pipelines_test
```

### Download XML dumps

You will need to begin with two Wikipedia XML dumps. Current dumps can be downloaded from https://dumps.wikimedia.org/, while historic dumps can be downloaded from https://archive.org/. You will want to download the `enwiki-YYYYMMDD-pages-articles.xml.bz2` file.

For the rest of the steps we'll assume you have the following files:
- `enwiki-20181120-pages-articles.xml.bz2`
- `enwiki-20191120-pages-articles.xml.bz2`

### Convert XML to jsonl

Next you will need to process the XML files to a JSONL.

``` {bash}
python -m language.fruit.scripts.run_convert_to_jsonl \
 --input_xml enwiki-20181120-pages-articles.xml.bz2 \
 --output_jsonl enwiki-20181120-pages-articles.jsonl

python -m language.fruit.scripts.run_convert_to_jsonl \
 --input_xml enwiki-20191120-pages-articles.xml.bz2 \
 --output_jsonl enwiki-20191120-pages-articles.jsonl
```

### Extract redirects

``` {bash}
python -m language.fruit.scripts.run_redirect_table_pipeline \
 --input_jsonl enwiki-20181120-pages-articles.jsonl  \
 --output_tsv enwiki-20181120-redirects.tsv \

python -m language.fruit.scripts.run_redirect_table_pipeline \
 --input_jsonl enwiki-20191120-pages-articles.jsonl  \
 --output_tsv enwiki-20191120-redirects.tsv \
```

### Process snapshots

```{bash}
python -m language.fruit.scripts.run_process_snapshot_pipeline \
 --target_jsonl enwiki-20191120-pages-articles.jsonl \
 --source_jsonl enwiki-20181120-pages-articles.jsonl \
 --output_dir train \
 --notruncate \
 --keep_tables \
 --third_party \
 --use_source_mentions \
 --target_redirects enwiki-20191120-redirects.tsv* \
 --source_redirects enwiki-20181120-redirects.tsv* \
```

### Filter for generation

```{bash}
python -m language.fruit.scripts.run_filter_for_generation_pipeline \
 --input_pattern article_pairs.jsonl* \
 --output_pattern article_pairs.update.jsonl \
 --nouse_source_mentions \
 --vocab_model_file "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"
```

### Prepare for model

In this step you may want to adjust some of the following parameters to prepare the data a certain way.

Here are the ones of interest:

-   `--task`:
   -   `fullgen`: Generate all of the output text.
   -   `diff`: Generate only the edits.
   -   `controllable`: Insert control codes in the input telling the model where to make edits.

-   `--delimiter_type`:

   -   `text`: Delimit input sentences and evidence using textual delimiters (e.g., [0], [1], (2), (3), etc.)
   -   `extra_id`: Delimit input sentences and evidence using unique tokens in the model vocab.

-   `--evidence_marker_type`:
   -   `empty`: Do not add any information about which evidence was used to produce an edit.
   -   `reference`: Insert reference tokens before each edit telling the model which evidence was used.

Here are the settings used to produce the EdiT5 training data:
``` {bash}
python -m language.fruit.scripts.run_to_tfrecords_pipeline \
 --input_pattern train/article_pairs.jsonl* \
 --output_pattern train/article_pairs.update.diff.all.reference.tfrecords \
 --task diff \
 --delimiter_type text \
 --evidence_marker_type reference \
 --noinclude_source \
 --include_evidence \
 --include_distractors \
 --vocab_model_file "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model"
```

## Applying to your own data.

The pipeline above can be applied to process other English Wikipedia snapshots by simply changing the input files.
For processing other languages you will likely need to edit the `beam_pipelines.py` file to use a more appropriate tokenizer.
For processing non-Wikipedia data, you will need to prepare the data so that it resembles the format output by the `run_convert_to_jsonl.py` script, and, depending on how entities are annotated, you may also need to modify the `process_snapshot_pipeline` routine in the `beam_pipelines.py` file. For support on creating non-Wikipedia FRUIT datasets please contact Rob Logan at [rloganiv@gmail.com](mailto:rloganiv@gmail.com)
