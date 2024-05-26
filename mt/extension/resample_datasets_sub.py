"""对各种语言单语语料进行重采样以产生SentencePiece训练语料"""
import argparse
import os

from extension.resample_datasets import count_lines, sample, upsample


def resample_prob(root, T=3.0):
    files = os.listdir(root)
    files = [os.path.join(root, f) for f in files]

    counts_raw = []
    for f in files:
        print("Counting lines:", f)
        n = count_lines(f)
        print("  ", f, ":", n)
        counts_raw.append(n)

    total = sum(counts_raw)
    print("Total lines:", total)

    probs_raw = []
    print("Raw counts and probabilities")
    for i, f in enumerate(files):
        p = counts_raw[i] / float(total)
        probs_raw.append(p)
        print("  ", f, counts_raw[i], p)

    T = 1 / T

    probs_raw_temp = [p ** T for p in probs_raw]
    prob_total = sum(probs_raw_temp)
    probs_normalized = [p / prob_total for p in probs_raw_temp]

    return zip(files, counts_raw, probs_raw, probs_normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="monolingual files directory")
    parser.add_argument("--total", type=int, default=15000000, help="total number of sentences for training SP")
    parser.add_argument("--t", type=float, default=3.0, help="sampling temperature")
    args = parser.parse_args()

    root = args.root

    files = os.listdir(root)
    files = [os.path.join(root, f) for f in files]

    counts_raw = []
    for f in files:
        print("Counting lines:", f)
        n = count_lines(f)
        print("  ", f, ":", n)
        counts_raw.append(n)

    total = sum(counts_raw)
    print("Total lines:", total)

    probs_raw = []
    print("Raw counts and probabilities")
    for i, f in enumerate(files):
        p = counts_raw[i] / float(total)
        probs_raw.append(p)
        print("  ", f, counts_raw[i], p)

    T = 1 / args.t
    probs_raw = [p ** T for p in probs_raw]
    prob_total = sum(probs_raw)
    probs_normalized = [p / prob_total for p in probs_raw]

    if args.total < total:
        total = args.total

    print("Total number of sentences after sampling:", total)

    counts_normalized = [int(p * total) for p in probs_normalized]
    print("Normalized counts and probabilities")
    for i, f in enumerate(files):
        print("  ", f, counts_normalized[i], probs_normalized[i])

    for i in range(len(counts_raw)):
        if counts_normalized[i] < counts_raw[i]:
            print("Sampling", files[i], counts_normalized[i])
            sample(files[i], counts_normalized[i])
        else:
            print("Upsampling", files[i], counts_normalized[i])
            upsample(files[i], counts_normalized[i])

