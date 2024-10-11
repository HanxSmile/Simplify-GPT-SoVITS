import torchaudio
import numpy as np
import random
import torch
import uuid
import os
from gpt_sovits.utils import save_audio


def save_audio_temp(prompt_audio):
    audio_data, sr = torchaudio.load(prompt_audio)
    audio_data = audio_data.mean(dim=0, keepdim=False)
    audio_data = audio_data.cpu().numpy()

    name = uuid.uuid4().hex
    output_dir = os.path.join(os.getcwd(), ".temp")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, name + '.wav')

    output_file = save_audio(audio_data, sr, output_file)
    return output_file


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
