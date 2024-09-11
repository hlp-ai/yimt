import numpy as np
import onnx
import torch
from onnx import checker
import onnxruntime as ort

from easyocr.model.vgg_model import Model

if __name__ == '__main__':
    model = Model(input_channel=1, output_channel=512, hidden_size=256, num_class=10)

    onnx_model_path = "crnn.onnx"

    dummy_input = torch.randn(1, 1, 32, 32)
    torch.onnx.export(model, dummy_input, onnx_model_path,
                      input_names=["images"],
                      output_names=["logits"]
                      )

    onnx_model = onnx.load(onnx_model_path)
    checker.check_model(onnx_model)

    # 加载模型
    sess = ort.InferenceSession(onnx_model_path)

    print(len(sess.get_inputs()))

    # 进行推理
    inp = np.random.randn(1, 1, 32, 32).astype(np.float32)
    result = sess.run(None, {'images': inp})

    # 输出结果
    print(result[0].shape)
