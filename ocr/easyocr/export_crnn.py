import onnx
import torch
from onnx import checker

from easyocr.model.vgg_model import Model

if __name__ == '__main__':
    model = Model(input_channel=1, output_channel=512, hidden_size=256, num_class=10)

    # model.eval()
    # with torch.no_grad():
    #     model(torch.randn(1, 1, 32, 32))
    #     model = torch.quantization.quantize_dynamic(
    #         model, {nn.Linear}, dtype=torch.qint8
    #     )
    # torch.quantization.convert(model, inplace=True)

    dummy_input = torch.randn(1, 1, 32, 32)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)

    onnx_model = onnx.load("crnn.onnx")
    checker.check_model(onnx_model)
