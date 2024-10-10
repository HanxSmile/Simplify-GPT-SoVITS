import gradio as gr
import torch
import os.path as osp
from gpt_sovits import Factory
from webui_utils import model_gen_funcs

CONFIG_PATH = osp.join(osp.dirname(__file__), 'config')
ARTICLE = r"""
If you find this repository is helpful, please help to ⭐ the <a href='https://github.com/HanxSmile/Simplify-GPT-SoVITS' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/opendatalab/VIGC?style=social)](https://github.com/HanxSmile/Simplify-GPT-SoVITS)

---

"""

model_lst = {
    "fish_speech": osp.join(CONFIG_PATH, "fishspeech.yaml"),
    "gpt_sovits": osp.join(CONFIG_PATH, "gpt_sovits.yaml"),
}

model_choices = [
    ("Fish Speech", "fish_speech"),
    ("GPT Sovits", "gpt_sovits"),
]

default_model = "gpt_sovits"

model = None


def init_model(model_type):
    global model
    cfg_path = model_lst[model_type]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Factory.read_config(cfg_path)
    model = Factory.build_model(cfg).to(device)


def generate_audio(model_type, prompt_audio, prompt_text, text):
    global model
    return model_gen_funcs[model_type](model, prompt_audio, prompt_text, text)


with gr.Blocks() as demo:
    with gr.Row():
        model_radio = gr.Radio(
            choices=model_choices,
            value=default_model,
            label="Select Model Type"
        )
        init_model_button = gr.Button("Init Model", variant="primary", size="lg")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=1, min_width=400):
                    with gr.Group():
                        reference_audio = gr.Audio(
                            label="Reference Audio",
                            type="filepath",
                        )
                        gr.Text(
                            "Please select a reference audio file and input the corresponding transcription on the right.",
                            max_lines=1,
                            container=False,
                            interactive=False
                        )
                with gr.Column(scale=10):
                    prompt_input_textbox = gr.Textbox(
                        label="Input Text for Reference Audio"
                    )
        with gr.Column():
            synthetic_input_textbox = gr.Textbox(
                label="Please Input the Text to be Converted to Audio",
            )
        with gr.Column():
            generate_button = gr.Button("Generate Audio", variant="primary", size="lg")

        with gr.Column():
            output_audio = gr.Audio(label="Synthesized Audio")

    generate_button.click(
        fn=generate_audio,
        inputs=[model_radio, reference_audio, prompt_input_textbox, synthetic_input_textbox],
        outputs=[output_audio])

    init_model_button.click(
        fn=init_model,
        inputs=[model_radio]
    )

demo.queue().launch(server_port=7860)
