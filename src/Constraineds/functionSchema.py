from enum import Enum
from typing import Any

from pydantic import BaseModel


class ParamType(str, Enum):
    """Supported parameter types in a function schema."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INT = "integer"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> "ParamType":
        """Resolve aliases (e.g. 'int', 'bool') for unknown values.

        Args:
            value: The raw value that did not match any member.

        Returns:
            The member matching the alias, or OTHER if there is none.
        """
        aliases = {
            "str": cls.STRING,
            "num": cls.NUMBER,
            "bool": cls.BOOLEAN,
            "int": cls.INT,
        }
        if isinstance(value, str):
            return aliases.get(value, cls.OTHER)
        return cls.OTHER


class ParamSchema(BaseModel):
    """Schema of a single function parameter."""

    type: ParamType


class FunctionSchema(BaseModel):
    """Schema of a callable function: its name and its parameters."""

    name: str
    parameters: dict[str, ParamSchema]

    @classmethod
    def from_dict(cls, fn: dict[str, Any]) -> "FunctionSchema":
        """Build a FunctionSchema from a raw dictionary.

        Args:
            fn: Dictionary with a 'name' key and a 'parameters' mapping
                from parameter name to a definition containing 'type'.

        Returns:
            The validated FunctionSchema.
        """
        props: dict[str, Any] = fn.get("parameters", {})

        params: dict[str, ParamSchema] = {}
        for pname, pdef in props.items():
            raw_type: str = pdef.get("type", "string").lower()
            params[pname] = ParamSchema(type=ParamType(raw_type))
        return cls(name=fn.get("name", ""), parameters=params)