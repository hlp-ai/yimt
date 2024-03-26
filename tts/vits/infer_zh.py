import numpy as np

import torch
from vits import utils
import argparse

from scipy.io import wavfile

from vits.models import SynthesizerTrn
from vits.text.zh_symbols import symbols
from vits.text.zh_symbols import cleaned_text_to_sequence
from vits.vits_pinyin import VITS_PinYin

parser = argparse.ArgumentParser(description='Inference code for bert vits models')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))

# device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# pinyin
tts_front = VITS_PinYin("./bert", device, hasBert=False)

# config
hps = utils.get_hparams_from_file(args.config)


# # model
net_g = utils.load_class(hps.train.eval_class)(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)

# model_path = "logs/bert_vits/G_200000.pth"
# utils.save_model(net_g, "vits_bert_model.pth")
# model_path = "vits_bert_model.pth"
utils.load_model(args.model, net_g)

# utils.load_checkpoint(args.model, net_g, None)

net_g.eval()
net_g.to(device)


item = "不知道这个测试能否行，这里有停顿吗？"
phonemes, _ = tts_front.chinese_to_phonemes(item)
input_ids = cleaned_text_to_sequence(phonemes)
with torch.no_grad():
    x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, bert=None, noise_scale=0.5,
                            length_scale=1)[0][0, 0].data.cpu().float().numpy()

save_wav(audio, f"./bert_vits_no_bert.wav", hps.data.sampling_rate)
