import whisper
import torch

model = whisper.load_model(r"D:\kidden\github\yimt\pretrained\asr\whisper\tiny.en.pt")
model.cpu()
model.eval()
mels = torch.randn(1, 80, 3000)
tokens = torch.randint(0, 51865, (1, 448))
torch.onnx.export(model, (mels, tokens), 'tiny.en.onnx', verbose=True)
