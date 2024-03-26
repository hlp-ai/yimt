from whisper.api import ASR


class AudioRecognizers:

    def __init__(self, model_path=r"D:\kidden\github\yimt\pretrained\asr\whisper\medium.pt"):
        self._model = ASR(model_path)

    def recognize_file(self, wave_fn):
        output = self._model.recognize_file(wave_fn)
        return [{"lang": output[0],
                 "text": output[1]}]

    def recognize(self, audio):
        output = self._model.recognize(audio)
        return [{"lang": output[0],
                 "text": output[1]}]

    @staticmethod
    def supported_languages():
        return ["en", "zh"]