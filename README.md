<div align="center">
<h1>Simplified GPT-SoVITS</h1>
</div>

[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

[**English**](./docs/README.md)| **中文简体** |

## 1. 简介

本项目对 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 进行精简，允许用户使用python代码进行简单地模型推理、训练

## 2. 安装

1. 创建虚拟环境

   ```bash
   conda create -n gpt_sovits python=3.8
   conda activate gpt_sovits
   ```

2. 安装torch

   ```bash
   pip install torch torchvision torchaudio
   ```

3. 安装ffmpeg

   ```bash
   conda install ffmpeg
   ```

4. 拉取项目并安装依赖

   ```bash
   git clone https://github.com/HanxSmile/Simplify-GPT-SoVITS.git
   cd Simplify-GPT-SoVITS
   pip install .
   ```

5. 验证是否安装成功

   ```bash
   python -c "from gpt_sovits import Factory"
   ```

   

## 3. few-shot 模型推理

1. 下载预训练模型（可以参考原作者项目 [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS)）

   ```python
   git lfs clone https://huggingface.co/lj1995/GPT-SoVITS
   ```

2. 下载中文g2p模型并解压

   ```bash
   wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip
   unzip G2PWModel_1.1.zip -d ./
   ```

3. 修改模型配置，将上面下载的模型的路径填写到模型配置的相应位置

   **config/example.yaml**:

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

4. 收集参考音频文件与相应的文本内容

5. 模型few-shot推理

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

- [x] **模型推理:**
  - [x] 中文
  - [x] 英文
  - [x] 日文
  
- [ ] **模型训练**
  - [ ] AR模型微调
  - [ ] Vits模型微调
  

## 参考项目

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)