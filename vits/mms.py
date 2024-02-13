import re
import tempfile
import torch
import commons
import utils
from models import SynthesizerTrn
from scipy.io.wavfile import write
import argparse
from torch.multiprocessing import Pool, get_start_method, set_start_method

import os
import subprocess
import locale

locale.getpreferredencoding = lambda: "UTF-8"


def download(lang, tgt_extract_dir="./models", tgt_dir="./"):
    lang_fn, lang_dir = os.path.join(tgt_dir, lang + '.tar.gz'), os.path.join(tgt_extract_dir, lang)
    if not os.path.exists(tgt_extract_dir):
        os.makedirs(tgt_extract_dir)
    if os.path.exists(lang_fn):
        print(f"Model for language {lang} exists in {lang_fn}")
        if os.path.exists(lang_dir):
            print(f"Model for language {lang} exists in {lang_dir}")
            print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
            return lang_dir
        else:
            print(f"Extracting model for language {lang} from {lang_fn} to {tgt_extract_dir}")

            cmd = f"tar zxvf {lang_fn} -C {tgt_extract_dir}"
            subprocess.check_output(cmd, shell=True)
            print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
            return lang_dir
    cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn} -C {tgt_extract_dir}"
    ])
    print(f"Download model for language: {lang}")
    subprocess.check_output(cmd, shell=True)
    print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
    return lang_dir


# #	English (eng), Korean (kor), Russian (rus), Vietnamese (vie), Thai (nod), Hindi (hin), Arabic (ara), French (fra), German standard (deu)
# LANGS = ["eng", "kor", "rus", "vie", "nod", "hin", "ara", "fra", "deu"]
#
# for LANG in LANGS:
#     ckpt_dir = download(LANG)


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
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w", encoding="utf-8") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name, encoding="utf-8") as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


def init(shared_args):
    global net_g  # Make net_g a global variable in the subprocess
    text_mapper, hps, device, ckpt_dir, _, _ = shared_args

    # Reinitialize net_g in the subprocess
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


def process_transcript(args):
    line, shared_args = args
    text_mapper, hps, device, ckpt_dir, lang, output_dir = shared_args
    global net_g  # Use the global net_g variable in the subprocess
    file_name, txt = line.strip().split("|")
    print(f"text: {txt}")
    txt = preprocess_text(txt, text_mapper, hps, lang=lang, uroman_dir="../uroman")
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

    LANGS = ["eng", "kor", "rus", "vie", "nod", "hin", "ara", "fra", "deu"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language",
        choices=LANGS,
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

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Max workers",
    )

    args = parser.parse_args()
    ckpt_dir = f"./models/{args.lang}"

    print(f"Run inference with {device}")
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    # net_g = SynthesizerTrn(
    #     len(text_mapper.symbols),
    #     hps.data.filter_length // 2 + 1,
    #     hps.train.segment_size // hps.data.hop_length,
    #     **hps.model)
    # net_g.to(device)
    # _ = net_g.eval()

    # g_pth = f"{ckpt_dir}/G_100000.pth"
    # print(f"load {g_pth}")

    # _ = utils.load_checkpoint(g_pth, net_g, None)

    # for line in open(args.transcript_file):
    #     file_name, txt = line.strip().split("|")
    #     print(f"text: {txt}")
    #     txt = preprocess_text(txt, text_mapper, hps, lang=args.lang, uroman_dir="../uroman")
    #     stn_tst = text_mapper.get_text(txt, hps)
    #     with torch.no_grad():
    #         x_tst = stn_tst.unsqueeze(0).to(device)
    #         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    #         hyp = net_g.infer(
    #             x_tst, x_tst_lengths, noise_scale=.667,
    #             noise_scale_w=0.8, length_scale=1.0
    #         )[0][0,0].cpu().float().numpy()

    #     print(f"Generated audio") 

    #     # Save audio
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     output_file = os.path.join(args.output_dir, f"{file_name}.wav")
    #     write(output_file, hps.data.sampling_rate, hyp)
    # Read transcript lines
    if get_start_method() == 'fork':
        set_start_method('spawn', force=True)

    transcript_lines = [line.strip() for line in open(args.transcript_file, encoding="utf-8")]
    # Initialize shared arguments
    shared_args = (text_mapper, hps, device, ckpt_dir, args.lang, args.output_dir)

    # Initialize multiprocessing pool
    with Pool(processes=args.max_workers, initializer=init, initargs=(shared_args,)) as pool:
        # Use pool.map to process transcript lines
        pool.map(process_transcript, [(line, shared_args) for line in transcript_lines])


if __name__ == "__main__":
    main()