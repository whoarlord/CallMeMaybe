import json
import os
from typing import Any

from pydantic import BaseModel, model_validator


class Parser(BaseModel):
    """Validate and access the input/output files of the program.

    All paths must be relative and stay inside `data/input/` (for the
    files that are read) or `data/output/` (for the file that is written).
    """

    functions_definition: str
    input: str
    output: str

    @model_validator(mode="after")
    def validation_rules(self) -> "Parser":
        """Check that the paths are safe, exist and have permissions.

        Returns:
            The validated instance.

        Raises:
            ValueError: If any path or permission rule is violated.
        """
        paths = (self.functions_definition, self.input, self.output)
        if any(p.startswith("/") for p in paths):
            raise ValueError("input parameters must not start with /")
        if any(p.startswith("..") for p in paths):
            raise ValueError("input parameters must not start with ..")
        if not self.output.startswith("data/output"):
            raise ValueError(
                "output parameter must start with data/output/ ..")
        if not (self.functions_definition.startswith("data/input")
                and self.input.startswith("data/input")):
            raise ValueError(
                "input parameters must start with data/input/ ..")

        if not (os.path.isfile(self.input)
                and os.path.isfile(self.functions_definition)):
            raise ValueError("input files must be files")

        if not (os.access(self.input, os.R_OK)
                and os.access(self.functions_definition, os.R_OK)
                and os.access(self.output, os.W_OK)):
            raise ValueError("files must have the correct permisions")
        return self

    def get_functions_definition_json(self) -> list[dict[str, Any]]:
        """Load the functions definition file.

        Returns:
            The parsed JSON content of the functions definition file.
        """
        result: list[dict[str, Any]]
        with open(self.functions_definition, 'r', encoding='utf-8') as file:
            result = json.load(file)
        return result

    def get_input_json(self) -> list[dict[str, Any]]:
        """Load the input (prompts) file.

        Returns:
            The parsed JSON content of the input file.
        """
        result: list[dict[str, Any]]
        with open(self.input, 'r', encoding='utf-8') as file:
            result = json.load(file)
        return result

    def load_in_output(self, output: list[dict[str, Any]]) -> None:
        """Write the results to the output file as indented JSON.

        Args:
            output: The list of results to serialize.
        """
        with open(self.output, 'w', encoding='utf-8') as file:
            json.dump(output, file, indent=2)

    def check_files_correctness(self) -> int:
        """Check the structure of the functions and input files.

        Prints a message describing the first problem found.

        Returns:
            0 if both files are valid, 1 otherwise.
        """
        parameter_values = ['number', 'num', 'integer', 'int',
                            'boolean', 'bool', 'str', 'string']
        try:
            functions = self.get_functions_definition_json()
            for function in functions:
                if (not function.get('name')
                        or not function.get('parameters')
                        or not function.get('description')
                        or not function.get('returns')):
                    raise ValueError('invalid keys')

                if not all('type' in v and v['type'] in parameter_values
                           for v in function['parameters'].values()):
                    raise ValueError('invalid parameters')
        except Exception as e:
            print("there was an error with the function definition "
                  f"file: {e}")
            return 1
        try:
            prompts = self.get_input_json()
            for prompt in prompts:
                if not prompt.get('prompt'):
                    raise ValueError('invalid prompt key')
        except Exception as e:
            print(f"there was an error with the function callings file: {e}")
            return 1
        return 0