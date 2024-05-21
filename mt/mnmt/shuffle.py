"""对多个平行语料文件按块进行混洗"""
import argparse
import os


def write(f, outf, block=2):
    n = 0
    for i in range(block):
        line = f.readline()
        if not line:
            return False, n

        line = line.strip()
        if len(line)>0:
            outf.write(line + "\n")
            n += 1
    return True, n


def shuffle(source, out_fn, block=2):
    files = os.listdir(source)
    files = [os.path.join(source, f) for f in files]
    files = [open(f, encoding="utf-8") for f in files]

    out = open(out_fn, "w", encoding="utf-8")
    total = 0
    n_files = len(files)
    i = 0
    print(n_files, "files left")
    while i < n_files:
        more, n = write(files[i], out, block)
        total += n
        if not more:
            print(files[i], "done")
            files[i].close()
            files.remove(files[i])
            print(len(files), "files left")
            n_files -= 1

        i += 1
        if i >= n_files:
            i = 0

        print(total, "lines")

    out.close()
    print(total, "lines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input directory")
    parser.add_argument("-o", "--output", required=True, help="output file")
    parser.add_argument("-b", "--block", type=int, default=8192, help="output file")
    args = parser.parse_args()

    shuffle(args.input, args.output, block=args.block)