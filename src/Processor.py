from . import Small_LLM_Model
from . import JsonTokenizer
import numpy as np
import json


class Processor():

    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[int, str] = self.get_vocab()
        self.json_tokenizer: JsonTokenizer = JsonTokenizer()
        self._json_mask_cache: dict[tuple, list[int]] = {}

    def encode_tensor(self, prompt: dict):
        tensor = self.llm.encode(prompt.get('prompt'))
        return tensor[0].tolist()

    def get_logits(self, tensor: list[int]):
        return self.llm.get_logits_from_input_ids(tensor)

    @staticmethod
    def apply_softmax(logits: list[int]):
        return np.exp(logits) / np.sum(np.exp(logits), 0)

    def decode(self, tensor: list[int]):
        return self.llm.decode(tensor)

    def get_vocab(self):
        result: dict
        with open(self.llm.get_path_to_vocab_file(),'r', encoding='utf-8') as file:
            result = json.load(file)
        result = {v: k for k, v in result.items()}
        return result

    def improve_prompt(self, prompt: str, functions: list[dict]):
        functions = json.dumps(functions)
        return f"""You are a function-calling engine. You do not explain, you do not think out loud, you only output JSON.

        Available functions:
        {functions}

        Rules:
        - Output ONLY a single valid JSON object. Nothing before it, nothing after it.
        - No markdown code fences (no ```).
        - No explanations, no reasoning, no <think> tags.
        - The JSON must match exactly this schema: 
        {{"prompt": "<request_prompt>", "name": "<function_name>", "arguments": {{ < param_name > : <value>, ...}}   }} 
        - Pick the single function that matches the request. Use only parameter names defined for that function.

        Examples:
        Request: "What is the sum of 10 and 5?"
        {{"prompt": "What is the sum of 10 and 5?", "name": "fn_add_numbers", "arguments": {{"a": 10, "b": 5}} }}

        Request: "Greet maria"
        {{"prompt": "Greet maria", "name": "fn_greet", "arguments": {{"name": "maria"}} }}

        Now respond to this request:
        Request: "{prompt}"

        <think>

        </think>
        """

    def token_is_valid(self, token: str):
        temp_json_tokenizer = self.json_tokenizer.clone()
        return temp_json_tokenizer.check_token(token)

    def calculate_valid_logits(self):
        key = (self.json_tokenizer.state, tuple(self.json_tokenizer.stack))
        if (key not in self._json_mask_cache):
            self._json_mask_cache[key] = [tki for tki, tkv in self.vocab.items()
                    if self.token_is_valid(tkv)]

        return self._json_mask_cache[key]
    
    def process_valid_logits(self, logits: list[float]) -> list[float]:
        valid_logits = set(self.calculate_valid_logits())

        for i in range(len(logits)):
            if i not in valid_logits:
                logits[i] = float('-inf')

        return logits

    def process_step(self, tki: int) -> None:
        return self.vocab.get(tki)


    def print_text(self, tensor: list[int]):
        result = ""
        for i in tensor:
            result += self.vocab.get(i)
        print(f"result: {result}")
                

    def process_prompt(self, prompt: dict, functions: list[dict]):
        prompt.update({'prompt': self.improve_prompt(
            prompt.get('prompt'), functions)})
        tensor = self.encode_tensor(prompt)
        eos_ids = [151645, 151643]
        actual_word = None
        iter: int = 0
        tensor_result = []
        while (actual_word not in eos_ids and iter < 1000):
            logits = self.get_logits(tensor)
            logits = self.process_valid_logits(logits)
            logits = self.apply_softmax(logits)
            actual_word = np.argmax(logits)
            print(f"actual word: {actual_word}")
            tensor.append(actual_word)
            tensor_result.append(actual_word)
            print(f"before adding token: {self.vocab.get(actual_word)}")
            self.json_tokenizer.check_token(self.vocab.get(actual_word))
            self.json_tokenizer.print_tokenizer()
            self.print_text(tensor_result)
            iter += 1
        result = self.decode(tensor_result)
        return result.strip()
