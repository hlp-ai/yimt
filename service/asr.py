from zhconv import zhconv

from whisper.api import ASR

from pydub import AudioSegment


def amr2wav(amr_file, wav_file):
    amr = AudioSegment.from_file(amr_file, format="amr")
    wav = amr.set_frame_rate(16000).set_channels(1)
    wav.export(wav_file, format="wav")


class AudioRecognizers:

    def __init__(self, model_path=r"D:\kidden\github\yimt\pretrained\asr\whisper\medium.pt"):
        self.model_path = model_path
        self._model = None

    def recognize_file(self, wave_fn, lang=None):
        if lang is not None:
            if lang not in self.supported_languages():
                return None

        if self._model  is None:
            print("Loading ASR...")
            self._model = ASR(self.model_path)

        output = self._model.recognize_file(wave_fn)

        lang, text = output
        if lang == "zh":
            text = zhconv.convert(text, 'zh-cn')
        return [{"lang": lang,
                 "text": text}]

    def recognize(self, audio, lang=None):
        if lang is not None:
            if lang not in self.supported_languages():
                return None

        if self._model  is None:
            print("Loading ASR...")
            self._model = ASR(self.model_path)

        output = self._model.recognize(audio)
        lang, text = output
        if lang == "zh":
            text = zhconv.convert(text, 'zh-cn')
        return [{"lang": lang,
                 "text": text}]

    @staticmethod
    def supported_languages():
        """支持语言列表"""
        return ["en", "zh", "ko", "ja", "ru", "th", "ta", "nl", "es", "it", "de", "pt", "pl", "id", "sv",
                "cs", "fr", "ro", "tr", "hu", "uk", "el", "bg", "ar", "sr", "hi", "da", "ur", "he"]


if __name__ == '__main__':
    s = AudioRecognizers()
    print(s.recognize_file(r"D:\dataset\LJSpeech-1.1\wavs\LJ001-0001.wav"))
