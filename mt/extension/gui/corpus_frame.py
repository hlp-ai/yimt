import tkinter as tk
import tkinter.messagebox
from functools import partial

from extension.gui.win_utils import ask_open_file, ask_save_file
from extension.utils import pair_to_single, single_to_pair, sample


def create_tsv2mono_corpus(parent):
    tk.Label(parent, text="TSV平行语料文件").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_pair = tk.Entry(parent, width=50)
    entry_corpus_pair.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_corpus_pair)).grid(row=0, column=2,
                                                                                                padx=10, pady=5)

    tk.Label(parent, text="输出源语言文件").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_src = tk.Entry(parent, width=50)
    entry_corpus_src.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_save_file, entry=entry_corpus_src)).grid(row=1, column=2, padx=10,
                                                                                               pady=5)

    tk.Label(parent, text="输出目标语言文件").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_tgt = tk.Entry(parent, width=50)
    entry_corpus_tgt.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_save_file, entry=entry_corpus_tgt)).grid(row=2, column=2, padx=10,
                                                                                               pady=5)

    def go():
        corpus_pair = entry_corpus_pair.get().strip()
        corpus_src = entry_corpus_src.get().strip()
        corpus_tgt = entry_corpus_tgt.get().strip()

        if len(corpus_pair) == 0 or len(corpus_src) == 0 or len(corpus_tgt) == 0:
            tk.messagebox.showinfo(title="Info", message="语料文件路径为空。")
            return

        pair_to_single(corpus_pair, corpus_src, corpus_tgt)

        tk.messagebox.showinfo(title="Info", message="转换完成")

    tk.Button(parent, text="将TSV双语文件分成单语文件", command=go).grid(row=5, column=1,
                                                              padx=10, pady=5)


def create_mono2tsv_corpus(parent):
    tk.Label(parent, text="path of source file").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_src = tk.Entry(parent, width=50)
    entry_corpus_src.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_corpus_src)).grid(row=0, column=2,
                                                                                               padx=10, pady=5)

    tk.Label(parent, text="path of target file").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_tgt = tk.Entry(parent, width=50)
    entry_corpus_tgt.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_corpus_tgt)).grid(row=1, column=2, padx=10,
                                                                                               pady=5)

    tk.Label(parent, text="path of parallel file").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_corpus_pair = tk.Entry(parent, width=50)
    entry_corpus_pair.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_save_file, entry=entry_corpus_pair)).grid(row=2, column=2,
                                                                                                padx=10,
                                                                                                pady=5)

    def go():
        corpus_pair = entry_corpus_pair.get().strip()
        corpus_src = entry_corpus_src.get().strip()
        corpus_tgt = entry_corpus_tgt.get().strip()

        if len(corpus_pair) == 0 or len(corpus_src) == 0 or len(corpus_tgt) == 0:
            tk.messagebox.showinfo(title="Info", message="Some parameter empty.")
            return
        single_to_pair(corpus_src, corpus_tgt, corpus_pair)
        tk.messagebox.showinfo(title="Info", message="done")

    tk.Button(parent, text="Combine source and target file into a parallel one", command=go).grid(row=5, column=1,
                                                                                                  padx=10, pady=5)


def create_sample_corpus(parent):
    tk.Label(parent, text="TSV File/Source File").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_sample_in1 = tk.Entry(parent, width=50)
    entry_sample_in1.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_sample_in1)).grid(row=0, column=2,
                                                                                               padx=10, pady=5)

    tk.Label(parent, text="Target File (Optional)").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_sample_in2 = tk.Entry(parent, width=50)
    entry_sample_in2.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_sample_in2)).grid(row=1, column=2,
                                                                                               padx=10, pady=5)

    tk.Label(parent, text="number of samples").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_sample_number = tk.Entry(parent, width=50)
    entry_sample_number.grid(row=2, column=1, padx=10, pady=5)

    def go():
        corpus_sample_in1 = entry_sample_in1.get().strip()
        corpus_sample_in2 = entry_sample_in2.get().strip()
        corpus_sample_number = entry_sample_number.get().strip()
        if len(corpus_sample_in1) != 0 and len(corpus_sample_in2) != 0:
            files = [corpus_sample_in1, corpus_sample_in2]
        elif len(corpus_sample_in1) != 0 and len(corpus_sample_in2) == 0:
            files = [corpus_sample_in1]
        elif len(corpus_sample_in1) == 0 and len(corpus_sample_in2) != 0:
            files = [corpus_sample_in2]
        else:
            tk.messagebox.showinfo(title="Info", message="Some parameter empty.")
            return
        if len(corpus_sample_number) == 0:
            tk.messagebox.showinfo(title="Info", message="Some parameter empty.")
            return
        sample(files, int(corpus_sample_number))
        tk.messagebox.showinfo(title="Info", message="done")

    tk.Button(parent, text="Sample sentences into a new corpus", command=go).grid( \
        row=5, column=1, padx=10, pady=5)
