from pydantic import BaseModel, field_validator
from enum import Enum


class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INT = "integer"

    @classmethod
    def _missing_(cls, value):
        aliases = {
            "str": cls.STRING,
            "num": cls.NUMBER,
            "bool": cls.BOOLEAN,
            "int": cls.INT,
        }
        return aliases.get(value)


class ParamSchema(BaseModel):
    type: ParamType


class FunctionSchema(BaseModel):
    name: str
    parameters: dict[str, ParamSchema]

    @classmethod
    def from_dict(cls, fn: dict) -> "FunctionSchema":
        props = fn.get("parameters", {})

        params: dict[str, ParamSchema] = {}
        for pname, pdef in props.items():
            raw_type: str = pdef.get("type", "string").lower()

            params[pname] = ParamSchema(type=ParamType(raw_type))
        return cls(name=fn.get('name', ''), parameters=params)
