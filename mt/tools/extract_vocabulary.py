#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse


def main():
    parser = argparse.ArgumentParser("导出模型词典")
    parser.add_argument("-model", type=str, required=True, help="模型路径")
    parser.add_argument("-out_file", type=str, required=True, help="保存词典路径")
    parser.add_argument("-side", choices=["src", "tgt"], help="导出哪个方向的词典")

    opt = parser.parse_args()

    if opt.side not in ["src", "tgt"]:
        raise ValueError("词典方向为src或tgt")

    import torch

    print("读入模型文件...")
    model = torch.load(opt.model, map_location=torch.device("cpu"))
    voc = model["vocab"][opt.side]

    print("写出词典到文件...")
    with open(opt.out_file, "wb") as f:
        for w in voc:
            f.write("{0}\n".format(w).encode("utf-8"))


if __name__ == "__main__":
    main()
