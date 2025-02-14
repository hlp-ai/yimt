import os
import tkinter.font as tkFont
import tkinter as tk
from tkinter import *
from functools import partial

from extension.gui.win_utils import ask_open_file, ask_save_file
from web.tm_utils import TMList


def create_translate_pdf(parent):
    from extension.files.translate_pdf import main

    tk.Label(parent, text="待翻译PDF文件").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_source = tk.Entry(parent, width=50)
    entry_source.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_source)).grid(row=0, column=2,
                                                                                          padx=10, pady=5)

    tk.Label(parent, text="源语言").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_sl = tk.Entry(parent, width=50)
    entry_sl.grid(row=1, column=1, padx=10, pady=5)
    entry_sl.insert(0, "en")

    tk.Label(parent, text="目标语言").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_tl = tk.Entry(parent, width=50)
    entry_tl.grid(row=2, column=1, padx=10, pady=5)
    entry_tl.insert(0, "zh")

    var_box = IntVar()
    check_ter = Checkbutton(parent, text="输出翻译框", variable=var_box, onvalue=1, offvalue=0)
    check_ter.grid(row=3, column=0, padx=10, pady=5)

    def go():
        source_path = entry_source.get().strip()
        debug_box = (var_box.get() == 1)

        sl = entry_sl.get().strip()
        tl = entry_tl.get().strip()

        if len(source_path) == 0:
            tk.messagebox.showwarning(title="Info", message="输入文件和不能为空.")
            return

        main(source_path, debug=debug_box, source_lang=sl, target_lang=tl)

        tk.messagebox.showinfo(title="Info", message="翻译完成")

    button_start = tk.Button(parent, text="开始翻译", command=go)
    button_start.grid(padx=3, pady=10, row=4, column=1)


tms = None
index = 0

def create_tm(parent):
    tk.Label(parent, text="语言对").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_lang = tk.Entry(parent, width=50)
    entry_lang.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(parent, text="源文本").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    text_src = Text(parent, width=50, height=6, undo=True, autoseparators=False,
                    spacing1=10, spacing2=10, spacing3=10)

    fontExample = tkFont.Font(family="Arial", size=14)
    text_src.configure(font=fontExample)

    text_src.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(parent, text="目标文本").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    text_tgt = Text(parent, width=50, height=6, undo=True, autoseparators=False,
                    spacing1=10, spacing2=10, spacing3=10)

    text_tgt.configure(font=fontExample)

    text_tgt.grid(row=2, column=1, padx=10, pady=5)

    label_stat = tk.Entry(parent, width=50)
    label_stat.grid(row=3, column=1, padx=10, pady=5)

    def update_stat():
        label_stat.delete(0, tk.END)
        label_stat.insert(0, "{}/{}".format(index, len(tms.records)))

    def open_tm():
        global tms
        global index
        filename = tk.filedialog.askopenfilename()
        tms = TMList(filename)
        index = 0
        if len(tms.records) > 0:
            tm = tms.records[index]
            entry_lang.insert(0, tm["direction"])
            text_src.delete("1.0", "end")
            text_src.insert(INSERT, tm["source"])
            text_tgt.delete("1.0", "end")
            text_tgt.insert(INSERT, tm["target"])

            update_stat()

    def next_tm():
        global tms
        global index
        if len(tms.records) > index+1:
            index += 1
            tm = tms.records[index]
            entry_lang.delete(0, tk.END)
            entry_lang.insert(0, tm["direction"])
            text_src.delete("1.0", "end")
            text_src.insert(INSERT, tm["source"])
            text_tgt.delete("1.0", "end")
            text_tgt.insert(INSERT, tm["target"])

            update_stat()

    def prev_tm():
        global tms
        global index
        if index > 0:
            index -= 1
            tm = tms.records[index]
            entry_lang.delete(0, tk.END)
            entry_lang.insert(0, tm["direction"])
            text_src.delete("1.0", "end")
            text_src.insert(INSERT, tm["source"])
            text_tgt.delete("1.0", "end")
            text_tgt.insert(INSERT, tm["target"])

            update_stat()

    button_open = tk.Button(parent, text="打开翻译记忆文件", command=open_tm)
    button_open.grid(padx=3, pady=10, row=4, column=1)

    button_next = tk.Button(parent, text="下一条", command=next_tm)
    button_next.grid(padx=3, pady=10, row=5, column=1)

    button_prev = tk.Button(parent, text="上一条", command=prev_tm)
    button_prev.grid(padx=3, pady=10, row=5, column=2)


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
    tk.Button(parent, text="...", command=partial(ask_save_file, entry=entry_target)).grid(row=2, column=2,
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

        tk.messagebox.showinfo(title="Info", message="翻译完成")

    button_start = tk.Button(parent, text="开始翻译", command=go)
    button_start.grid(padx=3, pady=10, row=5, column=1)


def create_sarcebleu_trans(parent):
    tk.Label(parent, text="参考翻译文件").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_ref = tk.Entry(parent, width=50)
    entry_ref.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_ref)).grid(row=0, column=2,
                                                                                          padx=10, pady=5)

    tk.Label(parent, text="系统翻译文件").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_sys = tk.Entry(parent, width=50)
    entry_sys.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_sys)).grid(row=1, column=2, padx=10,
                                                                                          pady=5)

    tk.Label(parent, text="语言对").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    entry_lang = tk.Entry(parent, width=50)
    entry_lang.grid(row=2, column=1, padx=10, pady=5)
    entry_lang.insert(0, "en-zh")

    tk.Label(parent, text="额外指标").grid(row=3, column=0, padx=10, pady=5, sticky="e")
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
            tk.messagebox.showwarning(title="Info", message="语料文件路径为空.")
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

        tk.messagebox.showinfo(title="Info", message="计算完成")

    button_start = tk.Button(parent, text="计算评价指标", command=go)
    button_start.grid(padx=5, pady=10, row=5, column=1)
