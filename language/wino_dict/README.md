# WinoDict

This folder contains the code for constructing the WinoDict dataset, built
in order to evaluate the abilities of large language models to acquire new
language through definitions during in-context learning.

To do so we build simple examples by leveraging the adversarial pairs from
Winograd Schema Challenge
([Levesque et al., 2012](https://dl.acm.org/doi/10.5555/3031843.3031909)) and
WinoGrande ([Sakaguchi et al., 2020](https://arxiv.org/abs/1907.10641)) where
we can identify the words whose meaning is critical to resolve a coreference
resolution problem successfully. Here's an example:

> The verb to plest means to be scared of, or want to avoid an object. The city councilmen refused the demonstrators a permit because **they** plested violence.

compared to

> The verb to plest means to publicly recommend or support. The city councilmen refused the demonstrators a permit because **they** plested violence.

## Prerequisites

You will first need to install the requirements and download the necessary
resources from spacy and nltk.

```bash
pip install -r requirements.txt
python -m nltk.downloader omw-1.4
python -m nltk.downloader wordnet
python -m spacy download en_core_web_md-3.0.0a1
```

## Generating data

```bash
pip install nltk

python3 create_new_words.py --output_path=$HOME/words.tsv

python3 generate.py --output_path=$HOME/winodict
```

## Feedback and questions

Please send over any questions to
{eisenjulian , jrcole} \[at\] google \[dot\] com

## Citation

If you use this data please cite:

```
@inproceedings{eisenschlos2022winodict,
title = {WinoDict: Probing language models for in-context language acquisition},
author = {Julian Martin Eisenschlos and Jeremy R. Cole and Fangyu Liu and William Weston Cohen},
url = {https://arxiv.org/abs/2209.12153},
year = {2022},
}
```
