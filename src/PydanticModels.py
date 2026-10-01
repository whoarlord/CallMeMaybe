"""Pydantic models for the input and output files of the project."""

from pydantic import (BaseModel, Field, RootModel,
                      field_validator, model_validator)

ParamValue = str | int | float | bool

# Accepted spellings, mapped to their canonical name.
_CANONICAL_TYPES: dict[str, str] = {
    "string": "string", "str": "string",
    "number": "number", "num": "number",
    "integer": "integer", "int": "integer",
    "boolean": "boolean", "bool": "boolean",
}


def _matches(type_name: str, value: ParamValue) -> bool:
    """Return True if a value fits a canonical parameter type."""
    if type_name == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):  # bool is a subclass of int
        return False
    if type_name == "integer":
        return isinstance(value, int)
    if type_name == "number":
        return isinstance(value, (int, float))
    return isinstance(value, str)


class PromptEntry(BaseModel):
    """One item of the input file: a user request."""

    prompt: str = Field(min_length=1, max_length=200)

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject prompts made only of whitespace."""
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class ParamDef(BaseModel):
    """Definition of a single parameter (only its type)."""

    type: str = Field(min_length=1, max_length=40)

    @field_validator("type")
    @classmethod
    def known_type(cls, value: str) -> str:
        """Check the type is supported and normalize it.

        Aliases ('int', 'bool', ...) become their canonical name.
        """
        canonical = _CANONICAL_TYPES.get(value.lower())
        if canonical is None:
            raise ValueError(f"unsupported parameter type '{value}'")
        return canonical


class ReturnDef(BaseModel):
    """Definition of the value returned by a function."""

    type: str = Field(min_length=1, max_length=200)


class FunctionDef(BaseModel):
    """One item of the functions definition file."""

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=200)
    parameters: dict[str, ParamDef] = Field(max_length=10)
    returns: ReturnDef

    @field_validator("parameters")
    @classmethod
    def names_not_empty(
            cls, value: dict[str, ParamDef]) -> dict[str, ParamDef]:
        """Reject empty parameter names."""
        if any(not name.strip() or len(name) > 20 for name in value):
            raise ValueError("parameter names must not be empty")
        return value


class FunctionsFile(RootModel[list[FunctionDef]]):
    """The whole functions definition file."""

    @model_validator(mode="after")
    def unique_names(self) -> "FunctionsFile":
        """Require a non-empty list with unique function names."""
        names = [fn.name for fn in self.root]
        if not names:
            raise ValueError("at least one function is required")
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"duplicated function names: {duplicated}")
        return self


class PromptsFile(RootModel[list[PromptEntry]]):
    """The whole input (prompts) file."""


class FunctionCall(BaseModel):
    """One item of the output file: a generated function call."""

    prompt: str
    name: str
    parameters: dict[str, ParamValue]

    def check_against(self, functions: dict[str, FunctionDef]) -> None:
        """Check the call against the function definitions.

        Args:
            functions: Function definitions indexed by name.

        Raises:
            ValueError: If the function is unknown, the parameter names
                do not match, or a value has the wrong type.
        """
        fn = functions.get(self.name)
        if fn is None:
            raise ValueError(f"unknown function '{self.name}'")

        expected = set(fn.parameters)
        got = set(self.parameters)
        if expected != got:
            raise ValueError(
                f"parameters mismatch: missing {sorted(expected - got)}, "
                f"unexpected {sorted(got - expected)}")

        for pname, value in self.parameters.items():
            type_name = fn.parameters[pname].type
            if not _matches(type_name, value):
                raise ValueError(
                    f"parameter '{pname}' must be {type_name}, "
                    f"got {value!r}")


def index_functions(
        functions: list[FunctionDef]) -> dict[str, FunctionDef]:
    """Index function definitions by name."""
    return {fn.name: fn for fn in functions}
