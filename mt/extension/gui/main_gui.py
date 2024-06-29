"""Admin GUI entry"""
from functools import partial
from tkinter import *
import tkinter as tk

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
    win_main.title("MT Pipeline")
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


    ####################################################################

    mainmenu = Menu(win_main)

    corpus_menu = Menu(mainmenu, tearoff=False)
    corpus_menu.add_command(label="TSV2Mono",command=partial(on_menu, tsv2mono_frame))
    corpus_menu.add_command(label="Mono2TSV",command=partial(on_menu,mono2tsv_frame))
    corpus_menu.add_separator()
    corpus_menu.add_command(label="Sample", command=partial(on_menu, sample_frame))
    corpus_menu.add_separator()
    corpus_menu.add_command(label="Exit", command=win_main.quit)

    mainmenu.add_cascade(label="Corpus", menu=corpus_menu)

    train_menu = Menu(mainmenu, tearoff=False)
    train_menu.add_command(label="Train SP", command=partial(on_menu, sp_train_frame))
    train_menu.add_command(label="Tokenize with SP", command=partial(on_menu, sp_tokenize_frame))

    mainmenu.add_cascade(label="Train", menu=train_menu)

    win_main.config(menu=mainmenu)

    for f in frames:
        f.pack_forget()

    win_main.mainloop()
