from pydantic import BaseModel, model_validator
from typing import Any
import json


class Parser(BaseModel):
    functions_definition: str
    input: str
    output: str

    @model_validator(mode="after")
    def validation_rules(self) -> Any:
        if (self.functions_definition.startswith("/")
                or self.input.startswith("/")
                or self.output.startswith("/")):
            raise ValueError("input parameterss must not start with /")
        if (self.functions_definition.startswith("..")
                or self.input.startswith("..")
                or self.output.startswith("..")):
            raise ValueError("input parameterss must not start with ..")
        return self

    def get_functions_definition_json(self):
        result = None
        with open(self.functions_definition, 'r', encoding='utf-8') as file:
            result = json.load(file)
        return result

    def get_input_json(self):
        result = None
        with open(self.input, 'r', encoding='utf-8') as file:
            result = json.load(file)
        return result

    def load_in_output(self, output: list[dict]):
        with open(self.output, 'w', encoding='utf-8') as file:
            json.dump(output, file, indent=2)

    def check_files_correctness(self):
        try:
            result = self.get_functions_definition_json()
            if (result):
                print("function definition file is correct")
        except Exception as e:
            print(f"there was an error with the function definition file: {e}")
            return 1
        try:
            result = self.get_input_json()
            if (result):
                print("function calling file is correct")
        except Exception as e:
            print(f"there was an error with the function callings file: {e}")
            return 1
        try:
            open(self.input, 'r', encoding='utf-8')
        except Exception as e:
            print(f"there was an error with the output file: {e}")
            return 1
