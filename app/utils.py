from revChatGPT.V3 import Chatbot
from pygpt4all.models.gpt4all_j import GPT4All_J

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out

# Define hyperparameters
max_word_count = 200 # maximum possible word count of user input
input_size = max_word_count # size of input layer
hidden_size = 128 # size of hidden layer
output_size = 5 # size of output layer
learning_rate = 0.01 # learning rate for optimizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=max_word_count)
with open('../data/intent classification dataset/intent_classification_data.json', 'r') as f:
    raw_data = json.load(f)

data = []
for i in raw_data.keys():
    data.extend(raw_data[i])
X = np.array([d['value'] for d in data])
X = vectorizer.fit_transform(X)


def new_text_callback(text):
    res = ''
    res = res + text
    print(text, end="")

class StoryTeller():
    def __init__(self) -> None:
        self.chatbot = GPT4All_J('./ggml-gpt4all-j-v1.3-groovy.bin')
        self.firstReq = True
        self.class_model = DNN(input_size, hidden_size, output_size)
        self.class_model.load_state_dict(torch.load('../models/intent_model_DNN.ckpt'))
        self._vectorizer = vectorizer

    def action(self, msg: str):
        print(msg)
        input = [msg]
        input = self._vectorizer.transform(input)
        input = torch.from_numpy(input.toarray()).float()
        res = ''

        # Make predictions
        with torch.no_grad():
            outputs = self.class_model(input)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()[0]

        print(predicted)
        if predicted == 1 and self.firstReq:
            res ="\n".join(self.chatbot.generate("Now you come to act as an adventure word game, description of the time to pay attention to the pace, not too fast, carefully describe the mood of each character and the surrounding environment. \n" + msg, new_text_callback=new_text_callback).split('\n')[1:])
            self.firstReq = False
        elif predicted == 2:
            res = "GoodBye"
        elif predicted == 3: 
            pass
        elif predicted == 4:
            next = "Continue the story. Don't be too long, just about 3 paragraph. \n"
            res = "\n".join(self.chatbot.generate(next + '\nI ' + msg, new_text_callback=new_text_callback).split('\n')[1:])
        else:
            res = self.chatbot.generate(msg, new_text_callback=new_text_callback)
        return res


