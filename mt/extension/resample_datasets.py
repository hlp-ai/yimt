import argparse
import io
import os
import random
from math import ceil


def count_lines(fn):
    from tqdm import tqdm
    lines = 0
    with open(fn, encoding="utf-8") as f:
        for _ in tqdm(f):
            lines += 1

    return lines


def sample(file, n):
    """"Sample sentences from bitext or source and target file"""
    in_file = io.open(file, encoding="utf-8")
    out_file = io.open("{}-{}".format(file, n), "w", encoding="utf-8")

    total = count_lines(files[0])
    print(total)

    sampled = 0
    scanned = 0
    sample_prob = (1.1*n) / total
    for p in in_file:
        scanned += 1
        prob = random.uniform(0, 1)
        if prob < sample_prob:
            out_file.write(p.strip() + "\n")

            sampled += 1
            if sampled % 10000 == 0:
                print(scanned, sampled)
            if sampled >= n:
                break
    print(scanned, sampled)


def upsample(file, n):
    """"UpSample sentences from bitext or source and target file"""
    total = count_lines(file)
    assert n>=total

    out_fn = "{}-{}".format(file, n)
    out_file = io.open(out_fn, "w", encoding="utf-8")

    times = n // total
    sampled = 0
    for _ in range(times):
        in_file = io.open(file, encoding="utf-8")
        for p in in_file:
            out_file.write(p.strip() + "\n")
            sampled += 1
            if sampled % 10000 == 0:
                print(sampled)
        in_file.close()

    scanned = 0
    sampled = 0
    n = n - total*times
    if n == 0:
        out_file.close()
        return out_fn

    sample_prob = (1.1*n) / total
    in_file = io.open(file, encoding="utf-8")
    for p in in_file:
        scanned += 1
        prob = random.uniform(0, 1)
        if prob < sample_prob:
            out_file.write(p.strip() + "\n")
            sampled += 1
            if sampled % 10000 == 0:
                print(scanned, sampled)
            if sampled >= n:
                break
    print(scanned, sampled)

    out_file.close()

    return out_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="tsv files directory")
    parser.add_argument("--t", type=float, default=3.0, help="sampling temperature")
    args = parser.parse_args()

    root = args.root

    files = os.listdir(root)
    files = [os.path.join(root, f) for f in files]

    counts_raw = []  # 原始数量
    for f in files:
        print("Counting lines:", f)
        n = count_lines(f)
        print("  ", f, ":", n)
        counts_raw.append(n)

    total = sum(counts_raw)
    print("Total lines:", total)
    max_count = max(counts_raw)  # 最大语料数量
    print("max", max_count)

    probs_raw = []  # 原始概率
    print("Raw counts and probabilities")
    for i, f in enumerate(files):
        p = counts_raw[i] / float(total)
        probs_raw.append(p)
        print("  ", f, counts_raw[i], p)

    T = 1 / args.t
    probs_tmp = [p ** T for p in probs_raw]
    prob_total = sum(probs_tmp)
    probs_normalized = [p / prob_total for p in probs_tmp]  # 温度采样概率
    print("采样概率")
    for i, f in enumerate(files):
        print("  ", f, probs_normalized[i])
    max_prob = max(probs_normalized)
    print("max sample prob", max_prob)

    sample_total = int(max_count / max_prob)
    print("sample total", sample_total)

    sample_counts = [ceil(p * sample_total) for p in probs_normalized]  # 训练集中各语料采样数量
    print("Sample counts and probabilities")
    for i, f in enumerate(files):
        print("  ", f, sample_counts[i], probs_normalized[i])

    for i in range(len(files)):
        print("采样", files[i])
        upsample(files[i], sample_counts[i])

