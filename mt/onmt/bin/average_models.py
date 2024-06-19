#!/usr/bin/env python
import argparse
import os

import torch


def average_models(model_files, fp32=False):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        print(model_file)
        m = torch.load(model_file, map_location="cpu")
        model_weights = m["model"]
        generator_weights = m["generator"]

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()
            for k, v in generator_weights.items():
                generator_weights[k] = v.float()

        if i == 0:
            vocab, opt = m["vocab"], m["opt"]
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for k, v in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for k, v in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)

    final = {
        "vocab": vocab,
        "opt": opt,
        "optim": None,
        "generator": avg_generator,
        "model": avg_model,
    }
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-model_dir", "-d", required=True, help="model directory"
    )
    parser.add_argument("-output", "-o", required=True, help="Output file")
    parser.add_argument(
        "-fp32", "-f", action="store_true", help="Cast params to float32"
    )
    opt = parser.parse_args()

    files = os.listdir(opt.model_dir)
    models = []
    for f in files:
        if f.endswith(".pt"):
            models.append(os.path.join(opt.model_dir, f))

    if len(models) > 2:
        final = average_models(models, opt.fp32)
        torch.save(final, opt.output)
    else:
        print("待平均模型文件小于2，退出")


if __name__ == "__main__":
    main()
