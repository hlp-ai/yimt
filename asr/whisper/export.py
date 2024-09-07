import os

import onnx
import torch
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

import whisper

model = whisper.load_model(r"D:\kidden\github\yimt\pretrained\asr\whisper\tiny.en.pt")
model.cpu()
model.eval()
mels = torch.randn(1, 80, 3000)
tokens = torch.randint(0, 51865, (1, 448))
model_fp32 = 'tiny-en-fp32.onnx'
torch.onnx.export(model, (mels, tokens), model_fp32,
                  input_names=["mels", "tokens"],
                  output_names=["logits"],
                  dynamic_axes={"tokens":[1],
                                "logits":[1]})

model = onnx.load(model_fp32)
onnx.checker.check_model(model)

model_quant = 'tiny-en-quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

model = onnx.load(model_quant)
onnx.checker.check_model(model)

print('ONNX full precision model size (MB):', os.path.getsize(model_fp32)/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize(model_quant)/(1024*1024))

# 加载模型
sess = ort.InferenceSession(model_quant)

# 创建输入数据
mels = np.random.randn(1, 80, 3000).astype(np.float32)
tokens = np.random.randint(0, 51865, (1, 320)).astype(np.int64)

print(ort.get_available_providers())

# 进行推理
input_name_1 = sess.get_inputs()[0].name
input_name_2 = sess.get_inputs()[1].name
result = sess.run(None, {input_name_1: mels, input_name_2: tokens})

# 输出结果
print(result[0].shape)
