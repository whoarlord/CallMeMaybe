from .. import Small_LLM_Model
import numpy as np
import json


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

    def improve_prompt(self, prompt: str, functions: list[dict]):
        functions = json.dumps(functions)
        return f"""You have access to these functions:
        {functions}

        You have this prompt inputs to answer: {prompt}

        You have to give me a valid JSON output without any other answer with this exact format:
        {{"function": "<function_name>", "arguments": {{...}}}}
        """


    def process_prompt(self, prompt: dict, functions: list[dict]):
        prompt.update({'prompt': self.improve_prompt(prompt.get('prompt'), functions)})
        tensor = self.encode_tensor(prompt)
        eos_ids = [151645, 151643]
        actual_word = None
        iter: int = 0
        tensor_result = []
        while (actual_word not in eos_ids and iter < 1000):
            logits = self.get_logits(tensor)
            logits = self.apply_softmax(logits)
            actual_word = np.argmax(logits)
            tensor.append(actual_word)
            tensor_result.append(actual_word)
            print(f"actual word: {actual_word}")
            iter += 1
        print(f"last word: {actual_word}")
        result = self.decode(tensor_result)
        return result
