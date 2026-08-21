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
        with open(self.input) as file:
            result = json.load(file)
        return result

    def get_input_json(self):
        result = None
        with open(self.input) as file:
            result = json.load(file)
        return result
