# MentionMemory model

This repository contains the code for the MentionMemory project.

## Requirements

```
git clone https://github.com/google-research/language
pip install -r language/mentionmemory/requirements.txt
```

Unit tests can be run via:

```bash
python -m language.mentionmemory.run_tests
```

Note that these tests might need to be run independently

```bash
python -m language.mentionmemory.encoders.mention_memory_encoder_test
python -m language.mentionmemory.encoders.readtwice_encoder_test
python -m language.mentionmemory.modules.kmeans_test
python -m language.mentionmemory.modules.memory_attention_layer_test
python -m language.mentionmemory.modules.memory_extraction_layer_test
python -m language.mentionmemory.modules.mention_losses_test
python -m language.mentionmemory.tasks.mention_memory_task_test
python -m language.mentionmemory.tasks.readtwice_task_test
python -m language.mentionmemory.training.trainer_test
python -m language.mentionmemory.utils.data_utils_test
```

When running the unit tests and all python commands mentioned later, the current working directory must the root the git project.
