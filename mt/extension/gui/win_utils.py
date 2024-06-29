import tkinter as tk
import tkinter.filedialog


def ask_open_file(entry):
    filename = tk.filedialog.askopenfilename()
    if filename != '':
        entry.delete(0, tk.END)
        entry.insert(0, filename)


def ask_save_file(entry):
    filename = tk.filedialog.asksaveasfilename()
    if filename != '':
        entry.delete(0, tk.END)
        entry.insert(0, filename)


def ask_dir(entry):
    filename = tk.filedialog.askdirectory()
    if filename != '':
        entry.delete(0, tk.END)
        entry.insert(0, filename)