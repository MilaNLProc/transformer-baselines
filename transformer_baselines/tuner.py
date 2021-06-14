from transformers import AutoConfig, AutoTokenizer


class Tuner:
    def __init__(self, base_encoder: str, base_tokenizer: str) -> None:
        self.base_encoder = base_encoder
        self.base_tokenizer = base_tokenizer

    def fit(texts: list, tasks: list, optimize: str = "memory"):
        raise NotImplementedError()

    def predict(texts):
        raise NotImplementedError()

    def score(texts):
        raise NotImplementedError()

    def cross_validate(texts):
        raise NotImplementedError()
