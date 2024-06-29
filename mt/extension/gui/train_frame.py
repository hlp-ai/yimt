""""Frame UI for train menu"""
import os
import tkinter as tk
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
from functools import partial


from extension.gui.win_utils import ask_open_file, ask_dir
from extension.sp import get_tok_file, get_sp_prefix, load_spm
from extension.sp_tokenize import tokenize_file_sp
from extension.sp_train import train_spm


def create_sp_train(parent):
    tk.Label(parent, text="Raw Corpus path").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_corpus = tk.Entry(parent, width=50)
    entry_corpus.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_corpus)).grid(row=0, column=2, padx=10,
                                                                                           pady=5)

    tk.Label(parent, text="Size of vocab").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_vocab_size = tk.Entry(parent)
    entry_vocab_size.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    entry_vocab_size.insert(0, "4800")

    tk.Label(parent, text="SP model path").grid(row=2, column=0, sticky="e")
    entry_model = tk.Entry(parent, width=50)
    entry_model.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_dir, entry_model)).grid(row=2, column=2, padx=10, pady=5)

    tk.Label(parent, text="Max num of sentences (M)").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    entry_max_sentences = tk.Entry(parent)
    entry_max_sentences.grid(row=3, column=1, padx=10, pady=5, sticky="w")
    entry_max_sentences.insert(0, "5")

    tk.Label(parent, text="Character coverage").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    entry_coverage = tk.Entry(parent)
    entry_coverage.grid(row=4, column=1, padx=10, pady=5, sticky="w")
    entry_coverage.insert(0, "0.9999")

    tk.Label(parent, text="User Defined Symbols File").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    entry_symbols_file = tk.Entry(parent, width=50)
    entry_symbols_file.grid(row=5, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry=entry_symbols_file)).grid(row=5, column=2,
                                                                                                 padx=10,
                                                                                                 pady=5)

    tk.Label(parent, text="remove_extra_whitespaces").grid(row=6, column=0, padx=10, pady=5, sticky="e")
    entry_extra_space = tk.Entry(parent)
    entry_extra_space.grid(row=6, column=1, padx=10, pady=5, sticky="w")
    entry_extra_space.insert(0, "true")

    tk.Label(parent, text="normalization_rule_name").grid(row=7, column=0, padx=10, pady=5, sticky="e")
    entry_norm = tk.Entry(parent)
    entry_norm.grid(row=7, column=1, padx=10, pady=5, sticky="w")
    entry_norm.insert(0, "nmt_nfkc")

    tk.Label(parent, text="split_digits").grid(row=8, column=0, padx=10, pady=5, sticky="e")
    entry_split_digits = tk.Entry(parent)
    entry_split_digits.grid(row=8, column=1, padx=10, pady=5, sticky="w")
    entry_split_digits.insert(0, "false")

    def go():
        corpus_file = entry_corpus.get()
        if len(corpus_file.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="Corpus path empty.")
            return

        vocab_size = entry_vocab_size.get()
        if len(vocab_size.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="Vocab size empty.")
            return

        sp_model = entry_model.get()
        if len(sp_model.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="Model path empty.")
            return

        sp_model = os.path.join(sp_model, get_sp_prefix(corpus_file, vocab_size))

        print(corpus_file, vocab_size, sp_model)

        max_sents = int(float(entry_max_sentences.get()) * 1000000)

        symbols_file = entry_symbols_file.get()
        if len(symbols_file.strip()) == 0:
            symbols_file = None

        remove_space = entry_extra_space.get().strip().lower()
        if remove_space == "true":
            remove_space = True
        else:
            remove_space = False

        split_digits = entry_split_digits.get().strip().lower()
        if split_digits == "true":
            split_digits = True
        else:
            split_digits = False

        normalization = entry_norm.get().strip().lower()

        train_spm(corpus_file, sp_model, vocab_size,
                  num_sentences=max_sents,
                  coverage=entry_coverage.get(),
                  remove_extra_whitespaces=remove_space,
                  normalization_rule_name=normalization,
                  split_digits=split_digits,
                  user_defined_symbols_file=symbols_file)

        tk.messagebox.showinfo(title="Info", message="SentencePiece model created.")

    tk.Button(parent, text="Train SentencePiece Model", command=go).grid(row=9, column=1, padx=10, pady=5)


def create_sp_tokenize(parent):
    tk.Label(parent, text="Raw Corpus path").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    entry_corpus = tk.Entry(parent, width=50)
    entry_corpus.grid(row=0, column=1, padx=10, pady=5)

    tk.Button(parent, text="...", command=partial(ask_open_file, entry_corpus)).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(parent, text="SP model path").grid(row=1, column=0, sticky="e")
    entry_model = tk.Entry(parent, width=50)
    entry_model.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_open_file, entry_model)).grid(row=1, column=2, padx=10, pady=5)

    tk.Label(parent, text="Output path").grid(row=2, column=0, sticky="e")
    entry_output = tk.Entry(parent, width=50)
    entry_output.grid(row=2, column=1, padx=10, pady=5)
    tk.Button(parent, text="...", command=partial(ask_dir, entry_output)).grid(row=2, column=2, padx=10, pady=5)


    def go():
        corpus_file = entry_corpus.get()
        if len(corpus_file.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="Corpus path empty.")
            return

        sp_model = entry_model.get()
        if len(sp_model.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="SP model empty.")
            return

        tok_output = entry_output.get()
        if len(tok_output.strip()) == 0:
            tk.messagebox.showinfo(title="Info", message="Output path empty.")
            return

        tok_output = os.path.join(tok_output, get_tok_file(corpus_file))

        print(corpus_file, sp_model, tok_output)

        sp = load_spm(sp_model)
        tokenize_file_sp(sp, corpus_file, tok_output)

        tk.messagebox.showinfo(title="Info", message="Raw corpus tokenized.")

    tk.Button(parent, text="Tokenize Corpus with SP", command=go).grid(row=3, column=1, padx=10, pady=5)


