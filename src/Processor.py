import json
import textwrap
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from . import Small_LLM_Model
from .Constraineds import FunctionCallGrammar, FunctionSchema


class Processor:
    """Generate function calls from prompts using a constrained LLM.

    At each step, the tokens that would break the expected JSON
    structure are masked out, so the model can only produce a valid
    function call.
    """

    def __init__(self, llm: Small_LLM_Model) -> None:
        """Initialize the processor.

        Args:
            llm: The language model used to compute the logits.
        """
        self.llm: Small_LLM_Model = llm
        self.vocab: dict[int, str] = self.get_vocab()
        self.eos_ids: list[int] = [151645, 151643]
        self._json_mask_cache: dict[tuple[Any, ...], list[int]] = {}

    def encode_tensor(self, prompt: str) -> list[int]:
        """Encode a text into a list of token ids."""
        tensor = self.llm.encode(prompt)
        result: list[int] = tensor[0].tolist()
        return result

    def get_logits(self, tensor: list[int]) -> list[float]:
        """Return the next-token logits for the given token ids."""
        return self.llm.get_logits_from_input_ids(tensor)

    @staticmethod
    def apply_softmax(logits: list[float]) -> npt.NDArray[np.float64]:
        """Convert logits into probabilities.

        The maximum is subtracted first for numerical stability.
        Logits equal to -inf get probability 0.
        """
        values = np.asarray(logits, dtype=np.float64)
        exps = np.exp(values - np.max(values))
        result: npt.NDArray[np.float64] = exps / np.sum(exps)
        return result

    def decode(self, tensor: list[int]) -> str:
        """Decode a list of token ids into text."""
        return self.llm.decode(tensor)

    def get_vocab(self) -> dict[int, str]:
        """Load the vocabulary file and invert it.

        Returns:
            A mapping from token id to token text.
        """
        vocab_file: str = self.llm.get_path_to_vocab_file()
        with open(vocab_file, 'r', encoding='utf-8') as file:
            raw: dict[str, int] = json.load(file)
        return {v: k for k, v in raw.items()}

    def improve_prompt(self, prompt: str,
                       functions: list[dict[str, Any]]) -> str:
        """Build the full prompt sent to the model.

        Args:
            prompt: The user request.
            functions: The available function definitions.

        Returns:
            The prompt with the functions, the rules and the request.
        """
        functions_json = json.dumps(functions)
        return textwrap.dedent(f"""\
        Available functions:
        {functions_json}

        Rules:
        - The JSON must match exactly this schema:
        {{"prompt": "<request_prompt>", "name": "<function_name>",
        "parameters": {{"<param_name>": "<value>", ...}}}}
        - The regex values must prioritize \\\\ over other escape chars
        - choose the function whose description best matches the user's
        overall intent, not just individual words in the prompt

        Now respond to this request:
        Request: "{prompt}"

        """)

    def calculate_valid_logits(
            self, func_tokenizer: FunctionCallGrammar) -> list[int]:
        """Return the ids of the tokens allowed in the current state.

        Results are cached by grammar state, since scanning the whole
        vocabulary is expensive.
        """
        key = (
            func_tokenizer.json.state,
            tuple(func_tokenizer.json.stack),
            func_tokenizer.json.escaped,
            func_tokenizer.json.no_more_parameters,
            func_tokenizer.json.ws_run,
            func_tokenizer.json._key_buffer,
            func_tokenizer.phase,
            func_tokenizer.current_param,
            getattr(func_tokenizer.active_value_constraint, "buffer", None),
            getattr(func_tokenizer.active_schema, "name", None),
        )
        if key not in self._json_mask_cache:
            self._json_mask_cache[key] = [
                tki for tki, tkv in self.vocab.items()
                if func_tokenizer.check_step(tkv)]

        return self._json_mask_cache[key]

    def process_valid_logits(self, logits: list[float],
                             func_tokenizer: FunctionCallGrammar
                             ) -> list[float]:
        """Mask the logits of the tokens that are not allowed.

        Args:
            logits: Raw logits from the model (not modified).
            func_tokenizer: Grammar holding the current state.

        Returns:
            A new list where invalid tokens have -inf. End-of-sequence
            tokens keep their original logit once the JSON is complete.
        """
        valid_logits = set(self.calculate_valid_logits(func_tokenizer))
        masked = [x if i in valid_logits else float('-inf')
                  for i, x in enumerate(logits)]

        if func_tokenizer.json.state == func_tokenizer.json.DONE:
            for eos_id in self.eos_ids:
                masked[eos_id] = logits[eos_id]

        return masked

    def process_step(self, tki: int) -> str | None:
        """Return the text of a token id, or None if it is unknown."""
        return self.vocab.get(tki)

    def print_text(self, tensor: list[int]) -> None:
        """Print the text corresponding to a list of token ids."""
        result = "".join(self.vocab.get(i, "") for i in tensor)
        print(f"result: {result}")

    def build_func_tokenizer(
            self, functions: list[dict[str, Any]]) -> FunctionCallGrammar:
        """Create a grammar from the raw function definitions."""
        schemas = {
            fn["name"]: FunctionSchema.from_dict(fn)
            for fn in functions
        }
        return FunctionCallGrammar(schemas)

    @staticmethod
    def get_start_prompt(prompt: dict[str, Any]) -> str:
        """Return the JSON text that starts the model's answer."""
        value = prompt.get('prompt')
        data = {"prompt": value}
        return json.dumps(data)

    def process_prompt(self, prompt: dict[str, Any],
                       functions: list[dict[str, Any]],
                       timeout_total: float = 10) -> str:
        """Generate the function call for a prompt.

        Tokens are generated greedily until the JSON is complete.

        Args:
            prompt: Dictionary with the user request under 'prompt'.
            functions: The available function definitions.
            timeout_total: Maximum time in seconds for the generation.

        Returns:
            The generated JSON text.

        Raises:
            ValueError: If the time limit is exceeded.
        """
        deadline = time.monotonic() + timeout_total
        start: str = self.get_start_prompt(prompt)[:-1] + ', "name":'
        prompt_str: str = self.improve_prompt(prompt.get('prompt'), functions)
        tensor: list[int] = self.encode_tensor(prompt_str + start)
        tensor_result: list[int] = self.encode_tensor(start)
        func_tokenizer = self.build_func_tokenizer(functions)
        iteration = 0
        while (func_tokenizer.json.state != func_tokenizer.json.DONE
                and iteration < 500):
            if deadline - time.monotonic() <= 0:
                raise ValueError("timeout thrown")
            logits = self.get_logits(tensor)
            masked = self.process_valid_logits(logits, func_tokenizer)
            probs = self.apply_softmax(masked)
            next_token = int(np.argmax(probs))
            tensor.append(next_token)
            tensor_result.append(next_token)
            if next_token in self.eos_ids:
                break
            func_tokenizer.apply_token(self.vocab.get(next_token, ""))
            self.print_text(tensor_result)
            iteration += 1
        result = self.decode(tensor_result)
        return result.strip()