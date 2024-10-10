import os
from .utils import save_audio_temp


def generate_audio(model, prompt_audio, prompt_text, text, ref_audio=None):
    prompt_audio = save_audio_temp(prompt_audio)
    if ref_audio is not None:
        ref_audio = save_audio_temp(ref_audio)
    inputs = {
        "prompt_audio": prompt_audio,
        "prompt_text": prompt_text,
        "text": text,
    }
    if ref_audio is not None:
        inputs["ref_audio"] = [ref_audio]
    else:
        inputs["ref_audio"] = [prompt_audio]

    model.register_prompt(inputs)
    sr, audio_data = model.generate(
        {"text": text},
        top_p=0.7,
        temperature=0.7,
        max_new_tokens=0,
        repetition_penalty=1.2,
    )
    try:
        os.remove(prompt_audio)
        if ref_audio is not None:
            os.remove(ref_audio)
    except FileNotFoundError as e:
        pass
    return sr, audio_data
