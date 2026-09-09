
from pydantic import BaseModel
import json

class TensorProcessor(BaseModel):
    vocabulary: dict[str, int]
    tensor: list[int]
    last_word: str

    def __init__(self, vocab_file: str, prompt: str):
        data: dict = {}
        with open(vocab_file, 'r', encoding='utf-8') as file:
            data.update({'vocabulary': json.load(file)})
        tensor, last_word = self.str_encoder(prompt)
        data.update({'tensor': self.str_encoder(prompt)})
        super().__init__(**data)

    def logits_decoder(self, tensor: list[int]):
        result: str = ''
        for index in tensor:
            result += self.vocabulary.get(index)
        print(f"result: {result}")
        return result

    def split_not_alnum_word(self, word: str, i: int) -> list[int]:
        result: list[int] = []
        current: str = ''
        is_first_token: bool = True

        for char in word:
            if char.isalnum() == (current != '' and current[-1].isalnum()) and current:
                current += char
                continue
            if current:
                if is_first_token and i != 0:
                    current = 'Ġ' + current
                result.append(self.vocabulary.get(current, 0))
                is_first_token = False
            current = char

        if current:
            if is_first_token and i != 0:
                current = 'Ġ' + current
            result.append(self.vocabulary.get(current, 0))

        return result

    def str_encoder(self, prompt: str) -> list[int]:
        tokens: list[int] = []
        for i, splitted in enumerate(prompt.split(' ')):
            if not splitted:
                continue
            tokens.extend(self.split_not_alnum_word(splitted, i))
        return tokens