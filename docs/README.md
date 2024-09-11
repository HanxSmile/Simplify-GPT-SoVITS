<div align="center">
<h1>Simplified GPT-SoVITS</h1>
</div>

[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

**English**]| [中文简体]((../README.md)) |


## 1. Introduction

This project streamlines [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), allowing users to perform simple model inference and training using Python code.

## 2. Installation

1. Create a virtual environment

   ```bash
   conda create -n gpt_sovits python=3.8
   conda activate gpt_sovits
   ```

2. Install torch

   ```bash
   pip install torch torchvision torchaudio
   ```

3. Install ffmpeg

   ```bash
   conda install ffmpeg
   ```

4. Clone the project and install dependencies

   ```bash
   git clone https://github.com/HanxSmile/Simplify-GPT-SoVITS.git
   cd Simplify-GPT-SoVITS
   pip install -r requirements
   ```

5. Verify installation

   ```bash
   python -c "from gpt_sovits import Factory"
   ```

   

## 3. Few-shot Model Inference

1. Download pre-trained models (refer to the original author's project [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS))

   ```python
   git lfs clone https://huggingface.co/lj1995/GPT-SoVITS
   ```

2. Download and extract the Chinese g2p model

   ```bash
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip
   unzip G2PWModel_1.1.zip -d ./
   ```

3. Modify the model configuration to include the paths of the downloaded models

   **config/example.yaml**:

   ```yaml
   model_cls: gpt_sovits
   
   hubert_model_name: GPT-SoVITS/chinese-hubert-base
   bert_model_name: GPT-SoVITS/chinese-roberta-wwm-ext-large
   t2s_model_name: GPT-SoVITS/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
   vits_model_name: /mnt/data/hanxiao/models/audio/GPT-SoVITS/gsv-v2final-pretrained/s2G2333k.pth
   text_converter:
     converter_cls: chinese_converter
     g2p_model_dir: G2PWModel_1.1
     g2p_tokenizer_dir: GPT-SoVITS/chinese-roberta-wwm-ext-large
   
   generate_cfg:
     placeholder: Null
   ```

4. Gather reference audio files and corresponding textual content

5. Perform few-shot model inference

   ```python
   from gpt_sovits import Factory
   from gpt_sovits.utils import save_audio
   import os
   import uuid
   
   cfg = Factory.read_config("/mnt/data/hanxiao/MyCode/Simiply-GPT-SoVITS/config/example.yaml")
   model = Factory.build_model(cfg)
   
   inputs = {
       "prompt_audio": "examples/linghua_90.wav",
       "prompt_text": "藏明刀的刀工,也被算作是本領通神的神士相關人員,歸屬統籌文化、藝術、祭祀的射鳳形意派管理。",
       "text": "明月几时有，把酒问青天"
   }
   model = model.cuda()
   sr, audio_data = model.generate(inputs)
   
   name = uuid.uuid4().hex
   output_dir = os.getcwd()
   output_file = os.path.join(output_dir, name + '.wav')
   
   output_file = save_audio(audio_data, sr, output_file)
   print(output_file)
   ```

## Todo List

- [x] **Model Inference:**
  - [x] Chinese
  - [ ] English
  - [ ] Japanese

- [ ] **Model Training:**
  - [ ] Fine-tuning AR model
  - [ ] Fine-tuning Vits model

## Referenced Projects

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)