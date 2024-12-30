from tkinter import *
from tkinter import filedialog
from tkinter.font import Font

from web.tm_utils import TMList


class TMFrame(Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.file_label = Label(self, text="TM文件(/)")
        self.langs_entry = Entry(self)

        self.src_text = Text(self, height=6, spacing1=10, spacing2=10, spacing3=10)
        self.tgt_text = Text(self, height=6, spacing1=10, spacing2=10, spacing3=10)
        fontExample = Font(family="Arial", size=14)
        self.src_text.configure(font=fontExample)
        self.tgt_text.configure(font=fontExample)

        self.file_label.pack(pady=5)
        self.langs_entry.pack(pady=5)
        self.src_text.pack()
        self.tgt_text.pack()

        self.nav_frame = Frame(self)
        self.next_button = Button(self.nav_frame, text="下一个", command=self.next)
        self.prev_button = Button(self.nav_frame, text="上一个", command=self.prev)
        self.first_button = Button(self.nav_frame, text="第一个", command=self.first)
        self.last_button = Button(self.nav_frame, text="最后 一个", command=self.last)

        self.delete_button = Button(self.nav_frame, text="删除", command=self.delete_item)
        self.save_item_button = Button(self.nav_frame, text="保存修改", command=self.save_item)

        self.next_button.grid(row=0, column=0, padx=10, pady=5)
        self.prev_button.grid(row=0, column=1, padx=10, pady=5)
        self.first_button.grid(row=0, column=2, padx=10, pady=5)
        self.last_button.grid(row=0, column=3, padx=10, pady=5)
        self.delete_button.grid(row=0, column=4, padx=10, pady=5)
        self.save_item_button.grid(row=0, column=5, padx=10, pady=5)
        self.nav_frame.pack()

        self.func_frame = Frame(self)
        self.open_button = Button(self.func_frame, text="打开文件", command=self.open_tm)
        self.save_button = Button(self.func_frame, text="保存文件", command=self.save_tm)
        self.saveas_button = Button(self.func_frame, text="另存文件")

        self.open_button.grid(row=0, column=0, padx=10, pady=5)
        self.save_button.grid(row=0, column=1, padx=10, pady=5)
        self.saveas_button.grid(row=0, column=2, padx=10, pady=5)
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

    def save_tm(self):
        self.tms.save()

    def delete_item(self):
        self.tms.records.pop(self.index)
        self.index -= 1
        if self.index < 0:
            self.index = 0
        self.display()

    def save_item(self):
        tm = self.tms.records[self.index]
        tm["source"] = self.src_text.get('0.0','end')
        tm["target"] = self.tgt_text.get('0.0','end')

    def display(self):
        tm = self.tms.records[self.index]

        self.langs_entry.delete(0, END)
        self.langs_entry.insert(0, tm["direction"])

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
