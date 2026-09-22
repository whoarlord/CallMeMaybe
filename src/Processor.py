from . import Small_LLM_Model
from .Constraineds import (FunctionCallGrammar, FunctionSchema)
import numpy as np
import json
import textwrap


class Processor():

    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[int, str] = self.get_vocab()
        self.eos_ids = [151645, 151643]
        self._json_mask_cache: dict[tuple, list[int]] = {}

    def encode_tensor(self, prompt: str):
        tensor = self.llm.encode(prompt)
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
        vocab_file: str = self.llm.get_path_to_vocab_file()
        with open(vocab_file, 'r', encoding='utf-8') as file:
            result = json.load(file)
        result = {v: k for k, v in result.items()}
        return result

    def improve_prompt(self, prompt: str, functions: list[dict]):
        functions = json.dumps(functions)
        return textwrap.dedent(f"""\
        Available functions:
        {functions}

        Rules:
        - The JSON must match exactly this schema:
        {{"prompt": "<request_prompt>", "name": "<function_name>",
        "parameters": {{"<param_name>": "<value>", ...}}}}
        - "source_string" must be exactly the raw string or sentence to modify,
        without any changes. Do NOT perform any substitutions
        or edits yourself inside the JSON values.

        Now respond to this request:
        Request: "{prompt}"

        """)

    def calculate_valid_logits(self, func_tokenizer: FunctionCallGrammar):
        key = (func_tokenizer.json.state, tuple(func_tokenizer.json.stack))
        if (key not in self._json_mask_cache):
            self._json_mask_cache[key] = [
                tki for tki, tkv in self.vocab.items()
                if func_tokenizer.check_step(tkv)]

        return self._json_mask_cache[key]

    def process_valid_logits(self, logits: list[float],
                             func_tokenizer: FunctionCallGrammar
                             ) -> list[float]:
        valid_logits = set(self.calculate_valid_logits(func_tokenizer))
        original_logits = logits.copy()

        for i in range(len(logits)):
            if i not in valid_logits:
                logits[i] = float('-inf')

        if func_tokenizer.json.state == func_tokenizer.json.DONE:
            for eos_id in self.eos_ids:
                logits[eos_id] = original_logits[eos_id]

        return logits

    def process_step(self, tki: int) -> None:
        return self.vocab.get(tki)

    def print_text(self, tensor: list[int]):
        result = ""
        for i in tensor:
            result += self.vocab.get(i)
        print(f"result: {result}")

    def build_func_tokenizer(self, functions: list[dict]) -> FunctionCallGrammar:
        schemas = {
            fn["name"]: FunctionSchema.from_dict(fn)
            for fn in functions
        }
        return FunctionCallGrammar(schemas)

    @staticmethod
    def get_start_prompt(prompt: dict):
        value = prompt.get('prompt')
        data = {"prompt": value}
        return json.dumps(data)

    def process_prompt(self, prompt: dict, functions: list[dict]):
        start: str = self.get_start_prompt(prompt)[:-1] + ', "name":'
        prompt_str: str = self.improve_prompt(prompt.get('prompt'), functions)
        print(f"start: {start}")
        tensor: list[int] = self.encode_tensor(prompt_str + start)
        tensor_result: list[int] = self.encode_tensor(start)
        actual_word = None
        iter: int = 0
        func_tokenizer: FunctionCallGrammar = self.build_func_tokenizer(functions)
        while (func_tokenizer.json.state != func_tokenizer.json.DONE and iter < 500):
            logits = self.get_logits(tensor)
            logits = self.process_valid_logits(logits, func_tokenizer)
            logits = self.apply_softmax(logits)
            actual_word = np.argmax(logits)
            print(f"actual word: {actual_word}")
            tensor.append(actual_word)
            tensor_result.append(actual_word)
            if actual_word in self.eos_ids:
                break
            print(f"before adding token: {self.vocab.get(actual_word)}")
            func_tokenizer.apply_token(self.vocab.get(actual_word))
            self.print_text(tensor_result)
            iter += 1
        result = self.decode(tensor_result)
        return result.strip()
