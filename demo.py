from gpt_sovits import Factory
from gpt_sovits.utils import save_audio
import os
import uuid

cfg = Factory.read_config("config/fishspeech.yaml")
model = Factory.build_model(cfg)

inputs = {
    "prompt_audio": "examples/linghua_90.wav",
    "prompt_text": "藏明刀的刀工,也被算作是本領通神的神士相關人員,歸屬統籌文化、藝術、祭祀的射鳳形意派管理。",
    "text": "在桃花坞深处，有一位剑术超群的先生，传说中无人见过他挥剑，因为见过的人都已不在人世。这是武林中常见的夸赞高手的方式，也是描述我的惯用语。人们口中的那位先生，正是慕星尘，也就是我。"
}
sr, audio_data = model.generate(inputs)

name = uuid.uuid4().hex
output_dir = os.getcwd()
output_file = os.path.join(output_dir, name + '.wav')

output_file = save_audio(audio_data, sr, output_file)
print(output_file)
