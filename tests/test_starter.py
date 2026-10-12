import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("torch")

import chat  # noqa: E402
import nltk_utils  # noqa: E402
import train as train_mod  # noqa: E402


def test_tokenize_and_bag_of_words():
    assert nltk_utils.tokenize("Hello, you?")[0] == "Hello"
    assert nltk_utils.bag_of_words(["Organizing"], ["organ", "x"]).tolist() == [1, 0]


def test_missing_model_error(tmp_path):
    with pytest.raises(chat.ModelNotTrained):
        chat.Chatbot(model_path=str(tmp_path / "x.pth"))


def test_train_and_chat(tmp_path):
    out = tmp_path / "data.pth"
    train_mod.train(os.path.join(ROOT, "intents.json"), str(out), num_epochs=400, verbose=False)
    bot = chat.Chatbot(model_path=str(out))
    greeting = next(i for i in json.load(open(os.path.join(ROOT, "intents.json"), encoding="utf-8"))["intents"]
                    if i["tag"] == "greeting")
    assert bot.get_response("Hello") in greeting["responses"]
    assert bot.get_response("zxqv wlkj") == chat.FALLBACK
