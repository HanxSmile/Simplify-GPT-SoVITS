<div align="center">
<h1>Simplified GPT-SoVITS</h1>
</div>

[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

**English**| [中文简体](../README.md) |


## 1. Introduction

This project streamlines [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)、[FishSpeech](https://github.com/fishaudio/fish-speech)、[ChatTTS](https://github.com/2noise/ChatTTS), allowing users to perform simple model inference and training using Python code.

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
   pip install .
   ```

5. Verify installation

   ```bash
   python -c "from gpt_sovits import Factory"
   ```

   

## 3. Few-shot Model Inference

### 3.1 GPT-SoVITS

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

   **config/gpt_sovits.yaml**:

   ```yaml
   model_cls: gpt_sovits
   
   hubert_model_name: GPT-SoVITS/chinese-hubert-base
   bert_model_name: GPT-SoVITS/chinese-roberta-wwm-ext-large
   t2s_model_name: GPT-SoVITS/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
   vits_model_name: /mnt/data/hanxiao/models/audio/GPT-SoVITS/gsv-v2final-pretrained/s2G2333k.pth
   cut_method: cut5
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
   
   cfg = Factory.read_config("/mnt/data/hanxiao/MyCode/Simiply-GPT-SoVITS/config/gpt_sovits.yaml")
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

### 3.2 FishSpeech

1. Download pre-trained models (refer to the original author's project [FishSpeech](https://github.com/fishaudio/fish-speech))

   ```bash
   git lfs clone https://huggingface.co/fishaudio/fish-speech-1.4
   ```

2. Modify the model configuration to include the paths of the downloaded models

   **config/fishspeech.yaml**:

   ```yaml
   model_cls: fish_speech
   cut_method: cut5
   vqgan:
     model_cls: filefly_vqgan
     ckpt: fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth
     spec_transform:
       sample_rate: 44100
       n_mels: 160
       n_fft: 2048
       hop_length: 512
       win_length: 2048
     backbone:
       input_channels: 160
       depths: [ 3, 3, 9, 3 ]
       dims: [ 128, 256, 384, 512 ]
       drop_path_rate: 0.2
       kernel_size: 7
     head:
       hop_length: 512
       upsample_rates: [ 8, 8, 2, 2, 2 ]
       upsample_kernel_sizes: [ 16, 16, 4, 4, 4 ]
       resblock_kernel_sizes: [ 3, 7, 11 ]
       resblock_dilation_sizes: [ [ 1, 3, 5 ], [ 1, 3, 5 ], [ 1, 3, 5 ] ]
       num_mels: 512
       upsample_initial_channel: 512
       pre_conv_kernel_size: 13
       post_conv_kernel_size: 13
     quantizer:
       input_dim: 512
       n_groups: 8
       n_codebooks: 1
       levels: [ 8, 5, 5, 5 ]
       downsample_factor: [ 2, 2 ]
   
   text2semantic:
     model_cls: dual_ar_transformer
     tokenizer_name: fish-speech-1.4/
     ckpt: fish-speech-1.4/model.pth
     model:
       attention_qkv_bias: False
       codebook_size: 1024
       dim: 1024
       dropout: 0.1
       head_dim: 64
       initializer_range: 0.02
       intermediate_size: 4096
       max_seq_len: 4096
       n_fast_layer: 4
       n_head: 16
       n_layer: 24
       n_local_heads: 2
       norm_eps: 1e-6
       num_codebooks: 8
       rope_base: 1e6
       tie_word_embeddings: False
       use_gradient_checkpointing: True
       vocab_size: 32000
   
   text_converter:
     converter_cls: chinese_fs_converter
   ```

3. Gather reference audio files and corresponding textual content

4. Perform few-shot model inference

   ```python
   from gpt_sovits import Factory
   from gpt_sovits.utils import save_audio
   import os
   import uuid
   
   cfg = Factory.read_config("/mnt/data/hanxiao/MyCode/Simiply-GPT-SoVITS/config/fishspeech.yaml")
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
  - [x] GPT-SoVITS
  - [x] FishSpeech
  - [ ] ChatTTS

- [ ] **Model Training**

## Referenced Projects

* [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* [FishSpeech](https://github.com/fishaudio/fish-speech)