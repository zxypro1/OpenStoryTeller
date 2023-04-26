import copy
from tkinter import *
from tkinter.ttk import Progressbar
import tkinter.messagebox
import tkinter.simpledialog
from typing import Callable

from utils import StoryTeller
import threading
from concurrent import futures


BG_GRAY = "#ABB2B9"
BG_COLOR = "#ffffff"
TEXT_COLOR = "#0b0b0b"

FONT = "Arial"
FONT_BOLD = "Rockwell"

story_background = "The continent of Syndicate is divided into the provinces of Torrey, Nile and Mura, in which live three races of dwarves, elves and humans as well as countless monsters. You are a human male wizard from Torrey, 21 years old. With a flaming staff in your left hand, a spellbook in your right, and a backpack with rations to last a week, you enter the Lykens Rainforest for an adventure."


def thread_it(func, *args):
    # 创建
    t = threading.Thread(target=func, args=args)
    # 守护进程
    t.daemon = True
    t.setDaemon(True)
    # 启动
    t.start()
    # t.join()


def format_form(form, width, height):
    """设置居中显示"""
    # 得到屏幕宽度
    win_width = form.winfo_screenwidth()
    # 得到屏幕高度
    win_height = form.winfo_screenheight()

    # 计算偏移量
    width_adjust = (win_width - width) / 2
    height_adjust = (win_height - height) / 2

    form.geometry("%dx%d+%d+%d" % (width, height, width_adjust, height_adjust))


class ChatApplication:

    def __init__(self, background):
        self.type = None  # 0 for api token (official api) or 1 for openai account (free api)
        self.api_key = None  # api_key
        self.api_entry = None  # api输入框
        self.api_window = None  # api输入弹窗
        self.login = None
        self.bar = None
        self.wait_window = None
        self.story_teller = None
        self.background = background
        self.window = Tk()
        self._setup_main_window()
        self.expired_creds = None

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Open-StoryTeller")
        self.window.resizable(width=False, height=False)
        format_form(self.window, 470, 550)
        self.window.configure(bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Open-StoryTeller", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=0.98, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.window)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # ">你"
        self.pre_text = Label(bottom_label, font=FONT, bg=BG_COLOR, text=">You")
        self.pre_text.config(fg='black', anchor="center", cursor="arrow")
        self.pre_text.place(relwidth=0.10, relheight=0.06, rely=0.008)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR,
                               fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06,
                             rely=0.008, relx=0.111)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_return)

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=self._on_enter_pressed)
        send_button.place(relx=0.87, rely=0.008, relheight=0.06, relwidth=0.12)
        self.before_start()

    def _on_return(self, event):
        """
        After enter <Return> in text entry.
        """
        self._on_enter_pressed()

    def show_api_window(self):
        # 创建一个窗口
        self.api_window = Toplevel()
        self.api_window.resizable(width=False, height=False)
        format_form(self.api_window, 300, 100)

        # 关闭则弹出登录窗口
        # self.api_window.protocol("WM_DELETE_WINDOW", self.show_login_window)

        # 创建一个标签
        label = Label(self.api_window, text="Enter your api key")

        # 创建一个输入框
        self.api_entry = Entry(self.api_window)

        button = Button(self.api_window, text="Confirm", command=self.on_api_confirm)

        # 将标签和输入框放置在窗口上
        label.pack()
        self.api_entry.pack()
        button.pack()

        # 启动窗口的主循环
        self.api_window.mainloop()

    def on_api_confirm(self):
        self.api_key = self.api_entry.get()
        self.close_api_window()
        self.register_storyteller(use_default=True)

    def close_api_window(self):
        if self.api_window is not None:
            self.api_window.destroy()

        self.api_window = None

    def close_login_window(self):
        """
        Close login window
        """
        if self.login is not None:
            self.login.destroy()

        self.login = None

    def show_background_window(self):
        """
        Show background story window.
        """
        result = tkinter.simpledialog.askstring(
            title='Enter background story', prompt='Please enter background story (or use the default one)', initialvalue=self.background,
            parent=self.window)
        if result:
            self.background = result
        self._init_background(self.background)

    def before_start(self):
        """
        Check if access token is expired.
        """
        self.show_api_window()

    def register_storyteller(self, use_default):
        """
        Init background and register a storyteller
        :param use_default: Whether user use default session token in config.py. If not, use access token.
        """
        self.show_background_window()
        self.story_teller = StoryTeller(self.api_key, self.background)

    def start_toplevel_window(self, msg):
        """
        Start a toplevel window when waiting for response.
        :param msg: Window title
        """

        # toplevel
        self.wait_window = Toplevel()
        self.wait_window.resizable(width=False, height=False)
        format_form(self.wait_window, 300, 50)
        self.wait_window.title(msg)
        self.wait_window.wm_attributes("-topmost", True)

        # progressbar
        self.bar = Progressbar(self.wait_window, length=250, mode="indeterminate",
                               orient=HORIZONTAL)
        self.bar.pack(expand=True)
        self.bar.start(10)

    def close_toplevel_window(self):
        """
        Close toplevel window.
        """
        if self.wait_window is not None:
            self.wait_window.destroy()

        self.wait_window = None
        self.bar = None

    def _init_background(self, background):
        """
        Put background story into dialog window.
        """
        if not background:
            return
        self.msg_entry.delete(0, END)
        msg1 = f">{background}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

    def _on_enter_pressed(self):
        """
        After enter <Send>.
        """
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        self.close_toplevel_window()

    def _insert_message(self, msg, sender):
        """
        Try to get message from ChatGPT and insert into dialog window.
        """
        if not msg:
            return
        self.msg_entry.delete(0, END)
        msg1 = f">{sender} {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        res = self.story_teller.action(msg)
        msg2 = f"{res}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication(story_background)
    app.run()
