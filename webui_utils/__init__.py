from .gpt_sovits import generate_audio as gpt_sovits_generate
from .fish_speech import generate_audio as fish_speech_generate

model_gen_funcs = {
    "gpt_sovits": gpt_sovits_generate,
    "fish_speech": fish_speech_generate,
}
