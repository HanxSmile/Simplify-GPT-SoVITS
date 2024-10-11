import gradio as gr
import torch
import os
import os.path as osp
from gpt_sovits import Factory
from webui_utils import model_gen_funcs, set_all_random_seed

REPO_PATH = osp.dirname(osp.abspath(__file__))
CONFIG_PATH = osp.join(REPO_PATH, 'config')
ARTICLE = r"""
If you find this repository is helpful, please help to ⭐ the <a href='https://github.com/HanxSmile/Simplify-GPT-SoVITS' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/HanxSmile/Simplify-GPT-SoVITS?style=social)](https://github.com/HanxSmile/Simplify-GPT-SoVITS)

---

"""
model_lst = sorted([_.replace(".yaml", "") for _ in os.listdir(CONFIG_PATH) if _.endswith(".yaml")])
model = None
MODEL_TYPE = model_lst[0]


def change_choices():
    all_models = sorted([_.replace(".yaml", "") for _ in os.listdir(CONFIG_PATH) if _.endswith(".yaml")])

    return {"choices": all_models, "__type__": "update"}


def init_model(model_type):
    global model
    global MODEL_TYPE
    MODEL_TYPE = model_type
    cfg_path = osp.join(CONFIG_PATH, f"{model_type}.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Factory.read_config(cfg_path)
    gr.Warning(f"Waiting for model '{model_type}' to load...")
    model = Factory.build_model(cfg).to(device)
    gr.Info(f"Model '{model_type}' Loaded!")


def generate_audio(model_type, upload_prompt_audio, recording_prompt_audio, prompt_text, text, seed=42):
    global model
    global MODEL_TYPE
    prompt_audio = upload_prompt_audio or recording_prompt_audio
    if prompt_audio is None:
        gr.Warning(
            f"Please select a reference audio file or recording your voice and input the corresponding transcription on the right.")
    elif upload_prompt_audio is None:
        gr.Info("You are using Recording Reference Audio!")
    else:
        gr.Info("You are using Upload Reference Audio!")
    if model is None or model_type != MODEL_TYPE:
        gr.Warning(f"Model '{model_type}' not loaded! Click 'Init Model' to load.'")
        init_model(model_type)
    cfg_path = osp.join(CONFIG_PATH, f"{model_type}.yaml")
    cfg = Factory.read_config(cfg_path)
    gr.Warning("Waiting for inference ...")
    set_all_random_seed(seed)
    result = model_gen_funcs[cfg.model_cls](model, prompt_audio, prompt_text, text)
    gr.Info("Inference Complete!")
    return result


def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


with gr.Blocks() as demo:
    with gr.Column():
        with gr.Group():
            models_dropdown = gr.Dropdown(label="Model List", choices=model_lst, value=model_lst[0], interactive=True,
                                          scale=14)
            with gr.Row():
                refresh_button = gr.Button("Refresh Model", variant="primary", scale=14)
                init_model_button = gr.Button("Init Model", variant="primary", scale=14)

    with gr.Column():
        with gr.Row(equal_height=True):
            with gr.Group():
                reference_audio = gr.Audio(
                    label="Upload Reference Audio",
                    type="filepath",
                    show_download_button=True,
                    sources=["microphone", "upload"],
                )
                recording_audio = gr.Microphone(
                    label="Recording Reference Audio",
                    type="filepath",
                    show_download_button=True,
                )
                gr.Text(
                    "Please upload a reference audio file or recording your voice and input the corresponding transcription on the right.\n"
                    "If both are provided, the uploaded reference audio will be used.",
                    lines=3,
                    container=False,
                    interactive=False,
                )
            prompt_input_textbox = gr.Textbox(
                label="Input Text for the Uploaded or Recording Reference Audio",
                lines=10
            )
    with gr.Column():
        synthetic_input_textbox = gr.Textbox(
            label="Please Input the Text to be Converted to Audio",
            lines=10,
        )
        generate_button = gr.Button("Generate Audio", variant="primary", size="lg")

        output_audio = gr.Audio(
            label="Synthesized Audio",
            show_download_button=True
        )

        gr.Markdown(html_center(ARTICLE), 'h3')

    refresh_button.click(fn=change_choices, inputs=[], outputs=[models_dropdown])
    generate_button.click(
        fn=generate_audio,
        inputs=[models_dropdown, reference_audio, recording_audio, prompt_input_textbox, synthetic_input_textbox],
        outputs=[output_audio])

    init_model_button.click(
        fn=init_model,
        inputs=[models_dropdown]
    )

demo.queue().launch(server_port=7860, share=True, server_name="0.0.0.0", debug=True)
