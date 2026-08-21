from .. import Small_LLM_Model
import numpy as np


class Processor():

    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm

    def encode_tensor(self, prompt: dict):
        tensor = self.llm.encode(prompt.get('prompt'))
        print(tensor)
        return tensor[0].tolist()

    def get_logits(self, tensor: list[int]):
        return self.llm.get_logits_from_input_ids(tensor)

    @staticmethod
    def apply_softmax(logits: list[int]):
        return np.exp(logits) / np.sum(np.exp(logits), 0)

    def decode(self, tensor: list[int]):
        return self.llm.decode(tensor)

    def process_prompt(self, prompt: dict):
        tensor = self.encode_tensor(prompt)
        logits = self.get_logits(tensor)
        logits = self.apply_softmax(logits)
        tensor.append(np.argmax(logits))
        result = self.decode(tensor)
        print(f"result: {result}")

    def calculate_next_word(self, tensor: list[int]):
        eos_ids = [151645, 151643]
        actual_word = None
        while (actual_word not in eos_ids):
            logits = self.get_logits(tensor)
            logits = self.apply_softmax(logits)
            actual_word = np.argmax(logits)
            tensor.append(actual_word)
            print(f"actual word: {actual_word}")
        result = self.decode(tensor)
        print(f"result: {result}")
