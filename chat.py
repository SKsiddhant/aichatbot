import json
import os
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = 0.75
FALLBACK = "I do not understand..."

bot_name = "Sam"


class ModelNotTrained(RuntimeError):
    """Raised when data.pth is missing, i.e. ``python train.py`` has not been run."""


class Chatbot:
    def __init__(self, intents_path=None, model_path=None):
        intents_path = intents_path or os.path.join(HERE, 'intents.json')
        model_path = model_path or os.path.join(HERE, 'data.pth')
        if not os.path.exists(model_path):
            raise ModelNotTrained(f"{os.path.basename(model_path)} not found. Run `python train.py` first.")

        with open(intents_path, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(model_path, map_location=self.device, weights_only=True)
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

    def get_response(self, msg):
        X = bag_of_words(tokenize(msg), self.all_words)
        X = torch.from_numpy(X.reshape(1, X.shape[0])).to(self.device)

        with torch.no_grad():
            output = self.model(X)
            probs = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probs, dim=1)

        if prob.item() > CONFIDENCE_THRESHOLD:
            tag = self.tags[predicted.item()]
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        return FALLBACK


_bot = None


def get_response(msg):
    """Module-level helper; loads the model on first use so importing this file has no side effects."""
    global _bot
    if _bot is None:
        _bot = Chatbot()
    return _bot.get_response(msg)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    try:
        while True:
            sentence = input("You: ")
            if sentence.strip().lower() == "quit":
                break
            print(get_response(sentence))
    except ModelNotTrained as exc:
        raise SystemExit(str(exc))
    except (EOFError, KeyboardInterrupt):
        print()  # Ctrl-D / Ctrl-C ends the chat without a traceback
