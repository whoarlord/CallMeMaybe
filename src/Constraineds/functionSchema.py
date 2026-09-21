from pydantic import BaseModel, field_validator
from enum import Enum


class ParamType(Enum):
    STRING = "string", "str"
    NUMBER = "number", "num"
    BOOLEAN = "boolean", "bool"
    INT = "int", "integer"


class ParamSchema(BaseModel):
    type: ParamType
    name: str


class FunctionSchema(BaseModel):
    name: str
    parameters: dict[str, ParamSchema]

    @classmethod
    def from_dict(cls, fn: dict) -> "FunctionSchema":
        props = fn.get("parameters", {})

        params: dict[str, ParamSchema] = {}
        for pname, pdef in props.values():
            raw_type: str = pdef.get("type", "string").lower()

            params[pname] = ParamSchema(type=ParamType(raw_type))
        return cls(name=fn.get('name', ''), parameters=params)
