import os
import tkinter as tk
from tkinter import *
from functools import partial

from extension.gui.win_utils import ask_open_file


def create_translate_file(parent):
    tk.Label(parent, text="待翻译单语文件").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_source = tk.Entry(parent, width=50)
    entry_source.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_source)).grid(row=0, column=2,
                                                                                          padx=10, pady=5)

    tk.Label(parent, text="模型推理配置文件").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_infer_conf = tk.Entry(parent, width=50)
    entry_infer_conf.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_infer_conf)).grid(row=1, column=2, padx=10,
                                                                                          pady=5)

    tk.Label(parent, text="输出翻译结果文件").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_target = tk.Entry(parent, width=50)
    entry_target.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_target)).grid(row=2, column=2,
                                                                                           padx=10, pady=5)

    def go():
        source_path = entry_source.get().strip()
        conf_path = entry_infer_conf.get().strip()
        output_path = entry_target.get().strip()

        if len(source_path) == 0 or len(conf_path) == 0:
            tk.messagebox.showwarning(title="Info", message="输入文件和推理配置文件不能为空.")
            return

        if len(output_path) == 0:
            output_path = source_path + ".pred"

        cmd = "python -m onmt.bin.translate -config {} -src {} -output {}"

        cf = os.popen(cmd.format(conf_path, source_path,output_path))
        lines = cf.readlines()
        for line in lines:
            print(line.strip())

        tk.messagebox.showinfo(title="Info", message="done")

    button_start = tk.Button(parent, text="开始翻译", command=go)
    button_start.grid(padx=3, pady=10, row=5, column=1)


def create_sarcebleu_trans(parent):
    tk.Label(parent, text="Reference File").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_ref = tk.Entry(parent, width=50)
    entry_ref.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_ref)).grid(row=0, column=2,
                                                                                          padx=10, pady=5)

    tk.Label(parent, text="Hyp File").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_sys = tk.Entry(parent, width=50)
    entry_sys.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_sys)).grid(row=1, column=2, padx=10,
                                                                                          pady=5)

    tk.Label(parent, text="Language Pair").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_lang = tk.Entry(parent, width=50)
    entry_lang.grid(row=2, column=1, padx=10, pady=5)
    entry_lang.insert(0, "en-zh")

    tk.Label(parent, text="Additional Metrics").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    var_ter = IntVar()
    check_ter = Checkbutton(parent, text="TER", variable=var_ter, onvalue=1, offvalue=0)
    check_ter.grid(row=3, column=1, padx=10, pady=5)

    var_chrf = IntVar()
    check_chrf = Checkbutton(parent, text="ChrF", variable=var_chrf, onvalue=1, offvalue=0)
    check_chrf.grid(row=4, column=1, padx=10, pady=5)

    def go():
        ref_path = entry_ref.get().strip()
        sys_path = entry_sys.get().strip()

        if len(ref_path) == 0 or len(sys_path) == 0:
            tk.messagebox.showwarning(title="Info", message="Some parameter empty.")
            return

        cal_cmd = "sacrebleu {} -i {} -l {} -f text"
        if var_ter.get() == 1 or var_chrf.get() == 1:
            cal_cmd += " -m bleu"
            if var_ter.get() == 1:
                cal_cmd += " ter"
            if var_chrf.get() == 1:
                cal_cmd += " chrf"
        cf = os.popen(cal_cmd.format(ref_path, sys_path, entry_lang.get().strip()))
        lines = cf.readlines()
        for line in lines:
            print(line.strip())

        tk.messagebox.showinfo(title="Info", message="done")

    button_start = tk.Button(parent, text="Calculate Metric", command=go)
    button_start.grid(padx=5, pady=10, row=5, column=1)
