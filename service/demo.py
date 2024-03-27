
import gradio as gr

from service.asr import AudioRecognizers
from service.ocr import TextRecognizers
from service.tts import AudioGenerators

audio_gen = AudioGenerators()
img_recognizer = TextRecognizers()
audio_reconginzer = AudioRecognizers()


def mt_func(text, sl, tl):
    return "translation"


def ocr_func(img, lang):
    r = img_recognizer.recognize(img, lang)
    return r


def asr_func(audio_fn):
    r = audio_reconginzer.recognize_file(audio_fn)
    return r[0]["text"]


def tts_func(txt, lang):
    r = audio_gen.generate(txt, lang)
    return r[0]["sr"], r[0]["audio"]


mt_face = gr.Interface(fn=mt_func,
                     inputs=[gr.Textbox(label="源文本"), gr.Textbox(label="源语言"), gr.Textbox(label="目标语言")],
                     outputs=gr.Textbox(),)


ocr_face = gr.Interface(fn=ocr_func,
                     inputs=[gr.Image(type='filepath'), gr.Textbox(label="图像语言")],
                     outputs=gr.Textbox(),
                     title="请选择图片")

asr_face = gr.Interface(fn=asr_func,
                     inputs=gr.Audio(type="filepath"),
                     outputs=gr.Textbox(),
                     title="请输入声音")

tts_face = gr.Interface(fn=tts_func,
                     inputs=[gr.Textbox(label="合成文本"), gr.Textbox(label="合成语言")],
                     outputs=gr.Audio(),
                     title="请输入文本")

tabbed_interface = gr.TabbedInterface([mt_face, ocr_face, asr_face, tts_face], ["MT", "OCR", "ASR", "TTS"])
tabbed_interface.launch()