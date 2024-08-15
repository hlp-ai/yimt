#!/usr/bin/env python
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="发布用于推理的翻译模型")
    parser.add_argument("--model", "-m", help="模型路径", required=True)
    parser.add_argument("--output", "-o", help="输出路径", required=True)
    parser.add_argument(
        "--format",
        choices=["pytorch", "ctranslate2"],
        default="pytorch",
        help="发布模型格式",
    )
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["int8", "int16", "float16", "int8_float16"],
        default=None,
        help="CT2模型量化类型",
    )
    opt = parser.parse_args()

    model = torch.load(opt.model, map_location=torch.device("cpu"))
    if opt.format == "pytorch":
        model["optim"] = None  # 推理不保存优化器状态
        torch.save(model, opt.output)
    elif opt.format == "ctranslate2":
        import ctranslate2

        converter = ctranslate2.converters.OpenNMTPyConverter(opt.model)
        converter.convert(opt.output, force=True, quantization=opt.quantization)


if __name__ == "__main__":
    main()
