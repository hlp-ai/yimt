import time

import whisper


class ASR:

    def __init__(self, model_path=r"D:\kidden\github\yimt\pretrained\asr\whisper\medium.pt"):
        self._model = whisper.load_model(model_path)

    def recognize_file(self, wave_fn):
        audio = whisper.load_audio_librosa(wave_fn)

        return self.recognize(audio)

    def recognize(self, audio):
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self._model.device)
        _, probs = self._model.detect_language(mel)
        lang = max(probs, key=probs.get)

        options = whisper.DecodingOptions(language=lang, fp16=True)
        result = whisper.decode(self._model, mel, options)

        # s1 = zhconv.convert(result.text, 'zh-cn')
        txt = result.text

        return lang, txt


if __name__ == '__main__':
    asr = ASR()

    audio_file = input("输入WAV文件路径：")
    start = time.time()
    lang, txt = asr.recognize_file(audio_file)
    print(time.time() - start)
    print(lang)
    print(txt)
