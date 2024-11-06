"""将一个大文件平分成固定大小的多个文件"""
import argparse


def split(file, num_per_file=1000000):
    """Split corpus into multiple files with the same lines"""
    in_file = open(file, encoding="utf-8")

    cnt = 0
    n_f = 0

    print("Split {}: File {}".format(file, n_f))
    out_file = open("{}-{}".format(file, n_f), "w", encoding="utf-8")

    for p in in_file:
        cnt += 1

        out_file.write(p.strip() + "\n")

        if cnt % 100000 == 0:
            print("Split {}: {}".format(file, cnt))

        if cnt % num_per_file == 0:
            out_file.close()

            n_f += 1
            out_file = open("{}-{}".format(file, n_f), "w", encoding="utf-8")
            print("Split {}: File {}".format(file, n_f))

    out_file.close()

    print("Split {}: {}".format(file, cnt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input file")
    parser.add_argument("-n", "--num", type=int, default=500000, help="the number of samples in each file")
    args = parser.parse_args()

    input = args.input
    sample_num = args.num

    split(input, num_per_file=sample_num)
