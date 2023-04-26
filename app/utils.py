from revChatGPT.V3 import Chatbot

class StoryTeller():
    def __init__(self, api_key: str, background: str) -> None:
        self.chatbot = Chatbot(api_key=api_key)
        self.background = background
        self.firstReq = True

    def action(self, msg: str):
        if self.firstReq:
            res = self.chatbot.ask("Now you come to act as an adventure word game, description of the time to pay attention to the pace, not too fast, carefully describe the mood of each character and the surrounding environment. The story background is:\n" + self.background + '\nI ' + msg)
            self.firstReq = False
        else:
            next = "Continue the story. Don't be too long, just about 3 paragraph."
            res = self.chatbot.ask(next + '\nI ' + msg)
        return res
