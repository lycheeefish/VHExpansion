# VHExpansion

This repository contains code for adversarial training of several multimodal language models for the paper **Automatically Generating Visual Hallucination Test Cases for Multimodal Large Language Models**.

## Environment Setup

To set up the required environment for each model, use the corresponding YAML file from the `envs` directory. Each file specifies the dependencies needed for a particular model:

- `envs/env_blip.yml`: Environment setup for InstructBLIP model.
- `envs/env_llava.yml`: Environment setup for LLaVA model.
- `envs/env_llava_next.yml`: Environment setup for LLaVA-Next model.
- `envs/env_llava_ov.yml`: Environment setup for LLaVA-OneVision model.
- `envs/env_qwen.yml`: Environment setup for Qwen model.

To create a conda environment using any of these files, run the following command (replace `<environment_file>` with the desired YAML file):

```sh
conda env create -f envs/<environment_file>
```

For example, to create an environment for the LLaVA model:

```sh
conda env create -f envs/env_llava.yml
```

## Datasets Preparations

To prepare the required datasets, download them from the following sources:

- [**MMVP Dataset**](https://huggingface.co/datasets/MMVP/MMVP)

- [**VHTest Dataset**](https://github.com/wenhuang2000/VHTest/tree/main/Benchmark)

- [**POPE Dataset**](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco)
  - Download the benchmark from the above link.
  - Download the required images from [here](https://cocodataset.org/#download).

## MLLMs Preparations

To prepare each of the multimodal large language models (MLLMs), please refer to the respective repositories and follow their instructions:

- [**InstructBLIP**](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

- [**LLaVA-1.5**](https://github.com/haotian-liu/LLaVA)
  In addition to the official instructions, make the following modifications:

  1. Add the following line below [this line](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/language_model/llava_llama.py#L44):

     ```python
     config.mm_vision_tower = "openai/clip-vit-large-patch14"
     ```

  2. Comment out the `@torch.no_grad()` decorator at [this line](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/multimodal_encoder/clip_encoder.py#L45).

- [**Qwen-VL-Chat**](https://github.com/QwenLM/Qwen-VL)

  In addition to the official instructions, modify the following:

  1. Change the `image_size` from `448` to `224` at [this line](https://huggingface.co/Qwen/Qwen-VL-Chat/blob/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/config.json#L40).

- [**LLaVA-Next**](https://github.com/LLaVA-VL/LLaVA-NeXT)

  In addition, add the function defined in [this file](https://github.com/lycheeefish/VHExpansion/blob/main/adversarial_training/additional_function.py) to the `LlavaMetaForCausalLM` class [here](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/1a7e8b2b4880f7548c1883669302a8b36bd79df6/llava/model/llava_arch.py#L162).

- [**LLaVA-OneVision**](https://github.com/LLaVA-VL/LLaVA-NeXT)

  In addition, add the function defined in [this file](https://github.com/lycheeefish/VHExpansion/blob/main/adversarial_training/additional_function.py) to the `LlavaMetaForCausalLM` class [here](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/1a7e8b2b4880f7548c1883669302a8b36bd79df6/llava/model/llava_arch.py#L162).

## Example Usage

To train a model, you can use the provided bash script `run_adversarial_training.sh` to simplify the process. This script allows you to select the model and set the required parameters for training.

To run the script, simply use:

```sh
bash run_adversarial_training.sh
```

Make sure to modify the script to suit your requirements, such as setting the appropriate input and output directories, number of steps, and training parameters.
- `--use_categories`: Set this flag if your images are organized into subdirectories based on categories, such as in the VHTest dataset. If images are not categorized (e.g., POPE or MMVP datasets), you can omit this flag.