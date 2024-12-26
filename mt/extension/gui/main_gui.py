"""Admin GUI entry"""
from functools import partial
from tkinter import *
import tkinter as tk

from extension.gui.app_frame import create_sarcebleu_trans, create_translate_file, create_translate_pdf
from extension.gui.corpus_frame import create_sample_corpus, create_mono2tsv_corpus, \
    create_tsv2mono_corpus
from extension.gui.train_frame import create_sp_tokenize, create_sp_train


def on_menu(frame):
    for f in frames:
        if f == frame:
            f.pack()
        else:
            f.pack_forget()


if __name__ == "__main__":
    win_main = tk.Tk()
    win_main.title("YiMT GUI")
    win_main.geometry("800x700")

    ##########################################################

    frames = []

    tsv2mono_frame=tk.Frame(win_main)
    tsv2mono_frame.pack()
    create_tsv2mono_corpus(tsv2mono_frame)
    frames.append(tsv2mono_frame)

    mono2tsv_frame = tk.Frame(win_main)
    mono2tsv_frame.pack()
    create_mono2tsv_corpus(mono2tsv_frame)
    frames.append(mono2tsv_frame)

    sample_frame = tk.Frame(win_main)
    sample_frame.pack()
    create_sample_corpus(sample_frame)
    frames.append(sample_frame)

    sp_train_frame = tk.Frame(win_main)
    sp_train_frame.pack()
    create_sp_train(sp_train_frame)
    frames.append(sp_train_frame)

    sp_tokenize_frame = tk.Frame(win_main)
    sp_tokenize_frame.pack()
    create_sp_tokenize(sp_tokenize_frame)
    frames.append(sp_tokenize_frame)

    bleu_frame = tk.Frame(win_main)
    bleu_frame.pack()
    create_sarcebleu_trans(bleu_frame)
    frames.append(bleu_frame)

    trans_file_frame = tk.Frame(win_main)
    trans_file_frame.pack()
    create_translate_file(trans_file_frame)
    frames.append(trans_file_frame)

    trans_pdf_frame = tk.Frame(win_main)
    trans_pdf_frame.pack()
    create_translate_pdf(trans_pdf_frame)
    frames.append(trans_pdf_frame)


    ####################################################################

    mainmenu = Menu(win_main)

    corpus_menu = Menu(mainmenu, tearoff=False)
    corpus_menu.add_command(label="双语到单语",command=partial(on_menu, tsv2mono_frame))
    corpus_menu.add_command(label="单语到双语",command=partial(on_menu,mono2tsv_frame))
    corpus_menu.add_separator()
    corpus_menu.add_command(label="采样", command=partial(on_menu, sample_frame))
    corpus_menu.add_separator()
    corpus_menu.add_command(label="退出", command=win_main.quit)

    mainmenu.add_cascade(label="语料", menu=corpus_menu)

    train_menu = Menu(mainmenu, tearoff=False)
    train_menu.add_command(label="训练SP分词模型", command=partial(on_menu, sp_train_frame))
    train_menu.add_command(label="基于SP模型切分", command=partial(on_menu, sp_tokenize_frame))

    mainmenu.add_cascade(label="训练", menu=train_menu)

    app_menu = Menu(mainmenu, tearoff=False)
    app_menu.add_command(label="PDF翻译", command=partial(on_menu, trans_pdf_frame))
    app_menu.add_command(label="翻译文件", command=partial(on_menu, trans_file_frame))
    app_menu.add_command(label="计算BLEU", command=partial(on_menu, bleu_frame))

    mainmenu.add_cascade(label="应用", menu=app_menu)

    win_main.config(menu=mainmenu)

    for f in frames:
        f.pack_forget()

    win_main.mainloop()
