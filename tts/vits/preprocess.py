"""对文件中文本进行clean"""
import argparse
from vits import text
from vits.utils import load_filepaths_and_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt",
                                                           "filelists/ljs_audio_text_test_filelist.txt"])
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)  # 每行格式：音频文件路径|文本
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)  # 清理文本
            filepaths_and_text[i][args.text_index] = cleaned_text

        new_filelist = filelist + "." + args.out_extension  # 清理后的文件
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
