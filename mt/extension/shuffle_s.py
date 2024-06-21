"""对单个文件进行混洗"""
import argparse
import random


def shuffle_block(block, out):
    random.shuffle(block)
    for s in block:
        out.write(s + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input file")
    parser.add_argument("-o", "--output", required=True, help="output file")
    parser.add_argument("-b", "--block", type=int, default=9600000, help="shuffle size")
    args = parser.parse_args()

    infn = args.input
    outfn = args.output
    buffer_size = args.block

    with open(outfn, "w", encoding="utf-8") as out:
        with open(infn, encoding="utf-8") as f:
            block = []
            for line in f:
                block.append(line.strip())
                if len(block) == buffer_size:
                    print("Shuffling {}".format(len(block)))
                    shuffle_block(block, out)
                    block = []

            if len(block) > 0:
                print("Shuffling {}".format(len(block)))
                shuffle_block(block, out)
