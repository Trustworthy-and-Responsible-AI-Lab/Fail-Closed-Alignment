# Fail-Closed Alignment for Large Language Models

This repository contains the code for training robust LLMs using the method introduced in our paper: [*Fail-Closed Alignment for Large Language Models*](https://arxiv.org/abs/2602.16977).

**Authors:** [Zachary Coalson](https://zachcoalson.com), [Sanghyun Hong](https://sanghyun-hong.com)

**Note:** We release our fine-tuned ("fail-closed") models at: [https://huggingface.co/collections/ztcoalson/fail-closed-alignment](https://huggingface.co/collections/ztcoalson/fail-closed-alignment).

----

## Supported Models

We currently support the following model families:

- [Llama-2](https://huggingface.co/docs/transformers/en/model_doc/llama2) (Chat)
- [Llama-3](https://huggingface.co/docs/transformers/en/model_doc/llama3) (Instruct)
- [Gemma-2](https://huggingface.co/docs/transformers/en/model_doc/gemma2) (Instruct)
- [Qwen2.5](https://huggingface.co/docs/transformers/en/model_doc/qwen2) (Instruct)
- [Granite-4.1](https://huggingface.co/docs/transformers/en/model_doc/granite) (Instruct)

### Adding New Models

To train other models, follow these steps:

1. Add the model's chat template to `src/feature_identification/rdo_utils/chat_templates.py`.
2. Add the model's "refusal tokens" to `src/feature_identification/rdo_utils/misc.py`. These are the tokens it begins its refusal with. To determine them, pass a few harmful prompts to the model and inspect the first token of the responses; it should generally be consistent (e.g., "I", "I'm", "As").
3. Create a model wrapper file in `src/feature_identification/dim_utils/pipeline/model_utils`. See the existing files for reference; it will also require adding the chat template and refusal tokens.
4. Register the model in `src/feature_identification/dim_utils/pipeline/model_utils/model_factory.py` by adding an `elif` branch in `construct_model_base()` that maps the model's name to your new class.

----

## Setup

Create a virtual environment (we recommend using [Conda](https://github.com/conda/conda) with `python=3.10`) and install the required packages:

```bash
conda create -n FAIL python=3.10
conda activate FAIL
pip install -r requirements.txt
```

For all experiments, navigate to the source directory:

```bash
cd src
```

----

## Running the Fail-Closed Alignment Trainer

You can run our method via the following scripts:

```bash
./scripts/generate_training_datasets.sh  # generate the model-specific training datasets (already done for models used in our paper)
./scripts/dim.sh  # generate the DIM refusal direction used by RDO

./scripts/fail_closed_alignment_trainer.sh
```

By default, the script performs full-scale fine-tuning; to do LoRA fine-tuning, uncomment the LoRA-related arguments. Models will be saved to `{MODELS_DIR}/fail-closed/{model_id}/{timestamp}`, where `{MODELS_DIR}` is an environment variable set in `.env`, `{model_id}` is a shorthand name for the target model (e.g., `google/gemma-2-2b-it` --> `gemma-2-2b-it`), and `{timestamp}` is the date and time that the script started.

----

## Evaluating Utility and Overrefusal

We provide code for evaluating the utility and over-refusal behavior of our models; you can perform the evaluation using the following scripts:

```bash
./scripts/utility_eval.sh
./scripts/overrefusal_eval.sh
```

Results will be stored in `{RESULTS_DIR}`, which you can change in the `.env` file.

**Note:** To evaluate robustness to prompt-based jailbreaks, please refer to the official [HarmBench](https://github.com/centerforaisafety/HarmBench) repository; our paper uses their exact implementations of all jailbreaks.

----

## Cite Our Work

Please cite our work if you find this source code helpful.

```
@inproceedings{coalson2026failclosed,
    title={Fail-Closed Alignment for Large Language Models},
    author={Zachary Coalson and Sanghyun Hong},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=qJN2JFwGdt}
}
```

----

Please contact Zachary Coalson ([coalsonz@oregonstate.edu](mailto:coalsonz@oregonstate.edu)) for any questions and recommendations.
