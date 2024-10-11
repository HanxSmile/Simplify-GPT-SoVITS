from .gpt_sovits import generate_audio as gpt_sovits_generate
from .fish_speech import generate_audio as fish_speech_generate
from .utils import set_all_random_seed

model_gen_funcs = {
    "gpt_sovits": gpt_sovits_generate,
    "fish_speech": fish_speech_generate,
}
