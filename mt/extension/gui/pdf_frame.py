from functools import partial
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.font import Font

from extension.files.translate_pdf import main
from extension.gui.win_utils import ask_open_file


class PdfFrame(Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        Label(self, text="待翻译PDF文件").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.entry_source = Entry(self, width=50)
        self.entry_source.grid(row=0, column=1, padx=10, pady=5)

        Button(self, text="...", command=partial(ask_open_file, entry=self.entry_source)).grid(row=0, column=2,
                                                                                               padx=10, pady=5)

        Label(self, text="源语言").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.entry_sl = Entry(self, width=50)
        self.entry_sl.grid(row=1, column=1, padx=10, pady=5)
        self.entry_sl.insert(0, "en")

        Label(self, text="目标语言").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.entry_tl = Entry(self, width=50)
        self.entry_tl.grid(row=2, column=1, padx=10, pady=5)
        self.entry_tl.insert(0, "zh")

        self.var_box = IntVar()
        check_ter = Checkbutton(self, text="输出翻译框", variable=self.var_box, onvalue=1, offvalue=0)
        check_ter.grid(row=3, column=1, padx=10, pady=5)

        Button(self, text="开始翻译", command=self.go).grid(padx=3, pady=10, row=4, column=1)

    def go(self):
        source_path = self.entry_source.get().strip()
        debug_box = (self.var_box.get() == 1)

        sl = self.entry_sl.get().strip()
        tl = self.entry_tl.get().strip()

        if len(source_path) == 0:
            messagebox.showwarning(title="Info", message="输入文件和不能为空.")
            return

        main(source_path, debug=debug_box, source_lang=sl, target_lang=tl)

        messagebox.showinfo(title="Info", message="翻译完成")


if __name__ == "__main__":
    main_win = Tk()
    tm_frame = PdfFrame(main_win)
    tm_frame.pack()

    main_win.mainloop()