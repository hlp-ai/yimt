
import gradio as gr


def ocr(img):
    return "OCR"


def asr(audio):
    return "ASR"


def tts(txt):
    return "TTS"


ocr_face = gr.Interface(fn=ocr,
                     inputs=gr.Image(type='pil'),
                     outputs=gr.Textbox(),
                     title="请选择图片")

asr_face = gr.Interface(fn=asr,
                     inputs=gr.Audio(),
                     outputs=gr.Textbox(),
                     title="请输入声音")

tts_face = gr.Interface(fn=tts,
                     inputs=gr.Textbox(),
                     outputs=gr.Audio(),
                     title="请输入文本")

tabbed_interface = gr.TabbedInterface([ocr_face, asr_face, tts_face], ["OCR", "ASR", "TTS"])
tabbed_interface.launch()