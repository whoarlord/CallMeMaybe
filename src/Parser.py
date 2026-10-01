import json
import os

from pydantic import BaseModel, model_validator
from .PydanticModels import (FunctionDef, FunctionsFile, PromptEntry,
                             PromptsFile, FunctionCall)


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

    def get_functions_definition(self) -> list[FunctionDef]:
        """Load the input (functions) file.

        Returns:
            The parsed JSON content of the input file.
        """
        with open(self.functions_definition, 'r', encoding='utf-8') as file:
            return FunctionsFile.model_validate_json(file.read()).root

    def get_input(self) -> list[PromptEntry]:
        """Load the input (prompts) file.

        Returns:
            The parsed JSON content of the input file.
        """
        with open(self.input, 'r', encoding='utf-8') as file:
            return PromptsFile.model_validate_json(file.read()).root

    def load_in_output(self, output: list[FunctionCall]) -> None:
        """Load the input (prompts) file.

        Returns:
            The parsed JSON content of the input file.
        """
        with open(self.output, 'w', encoding='utf-8') as file:
            json.dump([c.model_dump() for c in output], file, indent=2)
