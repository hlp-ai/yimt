from tkinter import *
from tkinter import filedialog

from web.tm_utils import TMList


class TMFrame(Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.file_label = Label(self, text="TM文件(/)")
        self.langs_label = Label(self, text="")
        self.src_text = Text(self)
        self.tgt_text = Text(self)

        self.file_label.pack()
        self.langs_label.pack()
        self.src_text.pack()
        self.tgt_text.pack()

        self.nav_frame = Frame(self)
        self.next_button = Button(self.nav_frame, text="下一个", command=self.next)
        self.prev_button = Button(self.nav_frame, text="上一个", command=self.prev)
        self.first_button = Button(self.nav_frame, text="第一个", command=self.first)
        self.last_button = Button(self.nav_frame, text="最后 一个", command=self.last)

        self.next_button.grid(row=0, column=0)
        self.prev_button.grid(row=0, column=1)
        self.first_button.grid(row=0, column=2)
        self.last_button.grid(row=0, column=3)
        self.nav_frame.pack()

        self.func_frame = Frame(self)
        self.open_button = Button(self.func_frame, text="打开文件", command=self.open_tm)
        self.save_button = Button(self.func_frame, text="保存文件")
        self.saveas_button = Button(self.func_frame, text="另存文件")

        self.open_button.grid(row=0, column=0)
        self.save_button.grid(row=0, column=1)
        self.saveas_button.grid(row=0, column=2)
        self.func_frame.pack()

    def open_tm(self):
        filename = filedialog.askopenfilename()
        if filename is None or len(filename)==0:
            return

        self.filename = filename

        self.tms = TMList(self.filename)
        self.index = 0
        if len(self.tms.records) > 0:
            self.display()

    def display(self):
        tm = self.tms.records[self.index]
        self.langs_label["text"] = tm["direction"]
        info = "{}({}/{})".format(self.filename, self.index+1, len(self.tms.records))
        self.file_label["text"] = info
        self.src_text.delete("1.0", "end")
        self.src_text.insert(INSERT, tm["source"])
        self.tgt_text.delete("1.0", "end")
        self.tgt_text.insert(INSERT, tm["target"])

    def next(self):
        if self.index >= len(self.tms.records)-1:
            return

        self.index += 1
        self.display()

    def prev(self):
        if self.index == 0:
            return

        self.index -= 1
        self.display()

    def first(self):
        self.index = 0
        self.display()

    def last(self):
        self.index = len(self.tms.records)-1
        self.display()


if __name__ == "__main__":
    main_win = Tk()
    tm_frame = TMFrame(main_win)
    tm_frame.pack()

    main_win.mainloop()
