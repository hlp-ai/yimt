import numpy as np

import torch
from vits import utils

from scipy.io import wavfile

from vits.models import SynthesizerTrn
from vits.text.zh_symbols import symbols
from vits.text.zh_symbols import cleaned_text_to_sequence
from vits.vits_pinyin import VITS_PinYin


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))



class TTS_ZH:

    def __init__(self, config_path, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pinyin
        self.tts_front = VITS_PinYin(r"D:\kidden\github\yimt\tts\vits\text\pinyin-local.txt")

        # config
        self.hps = utils.get_hparams_from_file(config_path)

        # # model
        self.net_g = utils.load_class(self.hps.train.eval_class)(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)

        utils.load_model(model_path, self.net_g)

        self.net_g.eval()
        self.net_g.to(self.device)

    def synthesize(self, txt):
        phonemes, _ = self.tts_front.chinese_to_phonemes(txt)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, bert=None, noise_scale=0.5,
                                length_scale=1)[0][0, 0].data.cpu().float().numpy()

        return audio, self.hps.data.sampling_rate


if __name__ == "__main__":
    tts = TTS_ZH(config_path=r"D:\kidden\github\yimt\pretrained\tts\zho\config.json",
                 model_path=r"D:\kidden\github\yimt\pretrained\tts\zho\G_100000.pth")
    item = "不知道这个测试能否行，这里有停顿吗？"
    audio, sr = tts.synthesize(item)

    save_wav(audio, f"./zh.wav", sr)
