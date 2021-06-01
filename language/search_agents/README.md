# Boosting Search Engines with Interactive Agents
Paper: <Link to the paper will be added here once it is published>

This directory contains the Search Agents presented in the paper:

* MuZero Agent trained by RL using SEED RL.
* T5 Agent fine-tuned using generated synthetic search sessions.

The API which the Search Agents use to interact with the Environment is
included; however, only a skeleton implementation of the gRPC service is
provided. In order to train the Search Agents or run inference using the
open sourced checkpoints, one must first implement this Environment Service.

## Prerequisites
We require tensorflow and other supporting libraries. Tensorflow should be
installed separately following the docs.

MuZero should be installed following instructions
[here](https://github.com/google-research/google-research/muzero).

Execute the following to download the TensorFlow Models and copy the
`official` directory into the `language` directory.

```
git clone https://github.com/tensorflow/models
cp -r models/official .
```

To install the other dependencies use

```
pip install -r requirements.txt
```

## Environment Service

To compile the gPRC interface for the Environment Service, execute:

```
python3 -m grpc_tools.protoc -I./language/search_agents \
--python_out=./language/search_agents \
--grpc_python_out=./language/search_agents \
./language/search_agents/environment.proto

sed -i \
's/import environment_pb2/import language.search_agents.environment_pb2/' \
./language/search_agents/environment_pb2_grpc.py
```

To run the service once it's implemented, execute:

```
python3 -m language.search_agents.environment_server -- --port=50055
```

## Training
To reproduce the paper results by training Search Agents, follow instructions
from the SEED RL repo to run [Distributed Training](https://github.com/google-research/seed_rl#distributed-training-using-ai-platform).
using the released code. Since the full training could take a while and consume
significant resources, we provide our checkpoints below.

## Evaluation

We release the checkpoints evaluated in the paper for both agents.

To download the MuZero checkpoints, execute:

```
mkdir -p saved_model/muzero/initial_inference/variables
gsutil cp gs://search_agents/muzero/saved_model/initial_inference/saved_model.pb saved_model/muzero/initial_inference/.
gsutil cp gs://search_agents/muzero/saved_model/initial_inference/variables/variables.data-00000-of-00001 saved_model/muzero/initial_inference/variables/.
gsutil cp gs://search_agents/muzero/saved_model/initial_inference/variables/variables.index saved_model/muzero/initial_inference/variables/.

mkdir -p saved_model/muzero/recurrent_inference/variables
gsutil cp gs://search_agents/muzero/saved_model/recurrent_inference/saved_model.pb saved_model/muzero/recurrent_inference/.
gsutil cp gs://search_agents/muzero/saved_model/recurrent_inference/variables/variables.data-00000-of-00001 saved_model/muzero/recurrent_inference/variables/.
gsutil cp gs://search_agents/muzero/saved_model/recurrent_inference/variables/variables.index saved_model/muzero/recurrent_inference/variables/.
```

To download the T5 checkpoints, execute:

```
mkdir -p saved_model/t5/variables
gsutil cp gs://search_agents/t5/saved_model/saved_model.pb saved_model/t5/.
gsutil cp gs://search_agents/t5/saved_model/variables/variables.data-00000-of-00002 saved_model/t5/variables/.
gsutil cp gs://search_agents/t5/saved_model/variables/variables.data-00001-of-00002 saved_model/t5/variables/.
gsutil cp gs://search_agents/t5/saved_model/variables/variables.index saved_model/t5/variables/.
```

[Apache Beam](https://beam.apache.org/) is required to run the evaluation
scripts. By default, Apache Beam runs in local mode but can also run in
distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/) and other
Apache Beam [runners](https://beam.apache.org/documentation/runners/capability-matrix/).

The input data for the evaluation scripts must be in jsonl with the following
format:

```
{
  'question': <question>
  'answer': <list of gold answers>
}
```

An example input file can be found
[here](https://storage.cloud.google.com/search_agents/sample_input.jsonl).
To download all other data dependencies for running the eval pipeline, execute:

```
mkdir data
gsutil cp gs://search_agents/sample_input.jsonl data/.
gsutil cp gs://search_agents/bert/bert_config.json .
gsutil cp gs://search_agents/bert/bert_model.ckpt.data-00000-of-00001 .
gsutil cp gs://search_agents/bert/bert_model.ckpt.index .
gsutil cp gs://search_agents/bert/vocab.txt .
```

The MuZero checkpoints need to be served through
[Tensorflow Serving](https://www.tensorflow.org/tfx/serving/docker).

Once the saved model for both initial_inference and recurrent_inference are
both served by tensorflow_model_server, execute the following to run the
MuZero agent eval:

```
python3 -m language.search_agents.muzero.run_inference_beam \
--input_jsonl=./data/sample_input.jsonl \
--output_path=./data/muzero_inference_output.jsonl \
--environment_server_spec=localhost:50055 \
--initial_inference_model_server_spec=localhost:8500 \
--initial_inference_model_name=initial_inference_model \
--recurrent_inference_model_server_spec=localhost:8501 \
--recurrent_inference_model_name=recurrent_inference_model
```

To run eval using the T5 checkpoint, execute:

```
python3 -m language.search_agents.t5.run_inference_beam \
--input_jsonl=./data/sample_input.jsonl \
--output_path=./data/t5_inference_output.jsonl \
--environment_server_spec=localhost:50055 \
--t5_saved_model=./saved_model/t5
```
