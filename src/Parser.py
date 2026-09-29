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
            raise ValueError("input parameters must not start with /")
        if (self.functions_definition.startswith("..")
                or self.input.startswith("..")
                or self.output.startswith("..")):
            raise ValueError("input parameters must not start with ..")
        if (not self.output.startswith('data/output')):
            raise ValueError(
                "output parameter must start with data/output/ ..")
        if (not self.functions_definition.startswith('data/input')
                and not self.input.startswith('data/input')):
            raise ValueError(
                "input parameters must start with data/input/ ..")
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
        parameter_values = ['number', 'num', 'integer', 'int',
                            'boolean', 'bool', 'str', 'string']
        try:
            result = self.get_functions_definition_json()
            if (result):
                print("function definition file is correct")
            for function in result:
                if (not function.get('name')
                    or not function.get('parameters')
                    or not function.get('description')
                        or not function.get('return')):
                    raise Exception('invalid keys')

                if (not all('type' in v and v['type'] in parameter_values
                            for v in function.get('parameters').values())):
                    raise Exception('invalid parameters')
        except Exception as e:
            print(f"there was an error with the function definition file: {e}")
            return 1
        try:
            result = self.get_input_json()
            if (result):
                print("function calling file is correct")
            for prompt in result:
                if (not in prompt.get('prompt')):
                    raise Exception('invalid prompt key')
        except Exception as e:
            print(f"there was an error with the function callings file: {e}")
            return 1
        try:
            open(self.input, 'r', encoding='utf-8')
        except Exception as e:
            print(f"there was an error with the output file: {e}")
            return 1
