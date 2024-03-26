import re
import torch
from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from scipy.io.wavfile import write
import argparse

import os
import locale

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

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        txt_fn = "D:/kidden/github/yimt/vits/src.txt"
        roman_fn = "D:/kidden/github/yimt/vits/roman.txt"
        with open(txt_fn, "w", encoding="utf-8") as f:
            f.write("\n".join([text]))
        cmd = f"c:/Strawberry/perl/bin/perl.exe " + uroman_pl
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


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    print(hps.data.training_files)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
        print(f"uromanize")
        txt = text_mapper.uromanize(txt, uroman_pl)
        print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


def process_transcript(args):
    line, shared_args = args
    text_mapper, hps, device, ckpt_dir, lang, output_dir = shared_args
    global net_g  # Use the global net_g variable in the subprocess
    file_name, txt = line.strip().split("|")
    print(f"text: {txt}")
    txt = preprocess_text(txt, text_mapper, hps, lang=lang, uroman_dir="./uroman")
    stn_tst = text_mapper.get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()

    print(f"Generated audio")

    # Save audio
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"MMSTTS_{lang}_{file_name}.wav")
    write(output_file, hps.data.sampling_rate, hyp)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Run inference with {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language",
        default="eng",
    )
    # Transcript 
    parser.add_argument(
        "--transcript_file",
        type=str,
        required=True,
        help="Transcript file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()
    ckpt_dir = f"./models/{args.lang}"

    print(f"Run inference with {device}")
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)

    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_g.to(device)
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"load {g_pth}")

    _ = utils.load_checkpoint(g_pth, net_g, None)

    for line in open(args.transcript_file, encoding="utf-8"):
        file_name, txt = line.strip().split("|")
        print(f"text: {txt}")
        txt = preprocess_text(txt, text_mapper, hps, lang=args.lang, uroman_dir="D:/kidden/github/yimt/vits/uroman")
        stn_tst = text_mapper.get_text(txt, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0,0].cpu().float().numpy()

        print(f"Generated audio")

        # Save audio
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{file_name}.wav")
        write(output_file, hps.data.sampling_rate, hyp)


if __name__ == "__main__":
    main()