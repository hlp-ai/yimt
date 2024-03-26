import re
import tempfile

import torch
from scipy.io.wavfile import write

from vits import commons

import os
import locale

from vits import utils
from vits.models import SynthesizerTrn

locale.getpreferredencoding = lambda: "UTF-8"


def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        print(self._symbol_to_id)
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl, perl_path=f"c:/Strawberry/perl/bin/perl.exe"):
        iso = "xxx"
        txt_fn = tempfile.NamedTemporaryFile().name  # "D:/kidden/github/yimt/vits/src.txt"
        roman_fn = tempfile.NamedTemporaryFile().name  # "D:/kidden/github/yimt/vits/roman.txt"
        with open(txt_fn, "w", encoding="utf-8") as f:
            f.write("\n".join([text]))
        cmd = perl_path + " " + uroman_pl
        cmd += f" -l {iso} "
        cmd += f" < {txt_fn} > {roman_fn}"
        print(cmd)
        os.system(cmd)
        outtexts = []
        with open(roman_fn, encoding="utf-8") as f:
            for line in f:
                line = re.sub(r"\s+", " ", line).strip()
                outtexts.append(line)
        outtext = outtexts[0]

        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        print(text_norm)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None, perl_path=f"c:/Strawberry/perl/bin/perl.exe"):
    txt = preprocess_char(txt, lang=lang)
    print(hps.data.training_files)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
        print(f"uromanize")
        txt = text_mapper.uromanize(txt, uroman_pl, perl_path=perl_path)
        print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


class TTS:

    def __init__(self, lang, models_dir="./models"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Run inference with {self.device}")

        self.lang = lang
        self.ckpt_dir = os.path.join(models_dir, lang)

        vocab_file = f"{self.ckpt_dir}/vocab.txt"
        config_file = f"{self.ckpt_dir}/config.json"
        assert os.path.isfile(config_file), f"{config_file} doesn't exist"

        self.hps = utils.get_hparams_from_file(config_file)
        self.text_mapper = TextMapper(vocab_file)

        self._load_model()

        # self.net_g = SynthesizerTrn(
        #     len(self.text_mapper.symbols),
        #     self.hps.data.filter_length // 2 + 1,
        #     self.hps.train.segment_size // self.hps.data.hop_length,
        #     **self.hps.model)
        # self.net_g.to(self.device)
        # _ = self.net_g.eval()
        #
        # g_pth = f"{self.ckpt_dir}/G_100000.pth"
        # print(f"load {g_pth}")
        #
        # _ = utils.load_checkpoint(g_pth, self.net_g, None)

    def _load_model(self):
        self.net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        self.net_g.to(self.device)
        _ = self.net_g.eval()

        g_pth = f"{self.ckpt_dir}/G_100000.pth"
        print(f"load {g_pth}")

        _ = utils.load_checkpoint(g_pth, self.net_g, None)


    def preprocess_text(self, txt, text_mapper, hps, uroman_dir=None, lang=None,
                        perl_path=f"c:/Strawberry/perl/bin/perl.exe"):
        txt = preprocess_char(txt, lang=lang)
        print(hps.data.training_files)
        is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
        if is_uroman:
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl, perl_path=perl_path)
            print(f"uroman text: {txt}")
        txt = txt.lower()
        txt = text_mapper.filter_oov(txt)
        return txt

    def synthesize(self, txt, uroman_dir="D:/kidden/github/yimt/tts/vits/uroman", perl_path=f"c:/Strawberry/perl/bin/perl.exe"):
        print(f"text: {txt}")
        txt = self.preprocess_text(txt, self.text_mapper, self.hps, lang=self.lang, uroman_dir=uroman_dir, perl_path=perl_path)
        stn_tst = self.text_mapper.get_text(txt, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            hyp = self.net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()

        print(f"Generated audio")

        return hyp, self.hps.data.sampling_rate


if __name__ == "__main__":
    tts = TTS("fra")

    audio, sr = tts.synthesize("Bonjour, je m'appelle Sophie. J'ai 25 ans et j'adore voyager.")
    write("./fra1.wav", sr, audio)

    audio, sr = tts.synthesize("Salut à tous! Aujourd'hui, c'est une journée ensoleillée et pleine de promesses.")
    write("./fra2.wav", sr, audio)

    tts_kor = TTS("kor")

    audio, sr = tts_kor.synthesize("안녕하세요, 저는 지금 한국에 살고 있는 청년입니다. 여러분들과 소통하며 즐거운 시간을 보내고 싶어요.")
    write("./kor1.wav", sr, audio)

    tts_eng = TTS("eng")

    audio, sr = tts_eng.synthesize("He made the remarks while presiding over a group study session of the Political Bureau of the CPC Central Committee on Thursday.")
    write("./eng1.wav", sr, audio)
