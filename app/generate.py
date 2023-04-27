import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


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
class_model = DNN(input_size, hidden_size, output_size)
class_model.load_state_dict(torch.load('../models/intent_model_DNN.ckpt'))
isFirstReq = True
history = []

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
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
    class_model = DNN(input_size, hidden_size, output_size)
    class_model.load_state_dict(torch.load('../models/intent_model_DNN.ckpt'))
    isFirstReq = True
    history = []

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=300,
        stream_output=False,
        **kwargs,
    ):
        msg = [instruction]
        msg = vectorizer.transform(msg)
        msg = torch.from_numpy(msg.toarray()).float()

        history = history + instruction + '\n'
        history = history + input + '\n'

        # Make predictions
        with torch.no_grad():
            class_result = class_model(msg)
            _, predicted = torch.max(class_result.data, 1)
            predicted = predicted.numpy()[0]

        if predicted == 1 and isFirstReq:  
            instruction = "Now you come to act as an adventure word game, description of the time to pay attention to the pace, not too fast, carefully describe the mood of each character and the surrounding environment. \n" + instruction
            isFirstReq = False
        elif predicted == 2:
            yield prompter.get_response("### Response: Okey, Goodbye!")
            history = []
            return
        elif predicted == 3: 
            yield prompter.get_response("### Response: Do summary")
            return
        elif predicted == 4:
            instruction = history + instruction + "\n Continue the story. Don't be too long, just about 3 paragraph and less than 300 words."
        else:
            pass

        prompt = prompter.generate_prompt(instruction, input)
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)
                    history = history + (prompter.get_response(decoded_output) + '\n')

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        history = history + (prompter.get_response(decoded_output) + '\n')
        yield prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Start a story. The background is in the world of Harry Potter, and I am a 13 year old new wizard.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=300, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            ),
            gr.components.Textbox(
                lines=5,
                label="History",
            )
        ],
        title="Open Story Teller ✒️",
        description="Open Story Teller is an AI assistant designed to assist writers to create more interesting and imaginative work.",
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
