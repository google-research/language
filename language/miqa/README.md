# MiQA

This folder contains the dataset for the paper [MiQA: A Benchmark for Inference on Metaphorical Questions](https://arxiv.org/abs/2210.07993).

## Data Format

The dataset is available in the data/ directory in TSV (Tab-separated) format.
The columns are: literal_premise, metaphorical_premise, literal_conclusion,
metaphorical_conclusion.

## Usage

Two types of MiQA questions can be built using each row in the dataset.

1. If $M_p$, which of the following statements could that imply: $M_c$ or $L_c$?

2. $L_c$ is implied by which of the following: $M_p or $L_p$?

Here are some examples:

| "_implies_"-questions                               | "_implied-by_"-questions                         |
|-----------------------------------------------------|--------------------------------------------------|
| "I see what you mean".                              | "My eyes are working well"                       |
| Which of the following statements could that imply? | is implied by which of the following?            |
| (1) My eyes are working well **[incorrect]**        | (1) I see what you are pointing at **[correct]** |
| (2) I understand you **[correct]**                  | (2) I see what you mean **[incorrect]**          |
|                                                     |                                                  |
| "A plan is not solid".                              | "A hammer could break it"                        |
| Which of the following statements could that imply? | is implied by which of the following?            |
| (1) A hammer could break it **[incorrect]**         | (1) A table is not solid **[correct]**           |
| (2) We should not follow it **[correct]**           | (2) A plan is not solid  **[incorrect]**         |
|                                                     |                                                  |
| "My friend has a huge problem".                     | "My friend needs space"                          |
| Which of the following statements could that imply? | is implied by which of the following?            |
| (1) My friend needs space **[incorrect]**           | (1) My friend has a huge dog **[correct]** [ ]   |
| (2) My friend needs a solution **[correct]**        | (2) My friend has a huge problem **[incorrect]** |

## Citation

If you use this data please cite:

```
@inproceedings{comsa2022miqa,
    author = {Comșa, Iulia-Maria and Eisenschlos, Julian Martin and Narayanan, Srini},
    title = "{MiQA: A Benchmark for Inference on Metaphorical Questions}",
    publisher = "Association for Computational Linguistics",
    booktitle = "Asian Chapter of the Association for Computational Linguistics",
    year = {2022},
    url = {https://arxiv.org/abs/2210.07993},
}
```
