import torch

from vits import utils
from vits.models import SynthesizerTrn

file_model = r"D:\kidden\github\yimt\pretrained\tts\vits\eng\G_100000.pth"
hps_ms = utils.get_hparams_from_file(r"D:\kidden\github\yimt\pretrained\tts\vits\eng\config.json")

symbols = [x.replace("\n", "") for x in
           open(r"D:\kidden\github\yimt\pretrained\tts\vits\eng\vocab.txt", encoding="utf-8").readlines()]

device = torch.device("cpu")

torch_model = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)

_ = torch_model.eval().to(device)

_ = utils.load_checkpoint(file_model, torch_model, None)

torch_model.forward = torch_model.infer

torch_model.eval()

x = torch.tensor([[2, 4, 11, 1, 2, 3, 4, 4]], dtype=torch.int64)
x_lengths = torch.tensor([1], dtype=torch.int64)
sid = torch.tensor([162], dtype=torch.int64)

inputs = (x, x_lengths, sid, 0.6, 1, 0.668)

export_onnx_file = "./vit-eng.onnx"
torch.onnx.export(torch_model,
                  inputs,
                  export_onnx_file,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=["x", "x_lengths", "sid", "noise_scale", "length_scale", "noise_scale_w"],
                  output_names=["audio"],
                  dynamic_axes={"x": {1: "text_length"},
                                "audio": {2: "audio_info"}})
