from . import JsonTokenizer, FunctionSchema, PrefixTrieConstraint
from enum import Enum


class Phase(Enum):
    EXPECT_NAME = 1
    IN_ARGUMENTS = 2
    OTHER = 3


class FunctionCallGrammar:
    def __init__(self, function_schemas: dict[str, FunctionSchema]):
        self.function_schemas = function_schemas
        self.json = JsonTokenizer(
            on_key_closed=self._on_key_closed,
            on_value_enter=self._on_value_enter,
            on_string_char=self._on_string_char,
            on_string_closed=self._on_string_closed,
        )
        self.phase = Phase.EXPECT_NAME
        self.active_schema: FunctionSchema | None = None
        self.current_param: str | None = None
        self.name_constraint = PrefixTrieConstraint(list(function_schemas.keys()))
        self.active_string_constraint: PrefixTrieConstraint | None = None

    def _on_key_closed(self, key: str) -> None:
        if key == "arguments":
            self.phase = Phase.IN_ARGUMENTS
        elif self.phase == Phase.IN_ARGUMENTS:
            self.current_param = key

    def _on_value_enter(self, kind: str) -> bool:
        if self.phase == Phase.EXPECT_NAME:
            self.name_constraint.reset()
            self.active_string_constraint = self.name_constraint
            return kind == 'string'

        if self.phase == Phase.IN_ARGUMENTS and self.active_schema:
            schema = self.active_schema.parameters.get(self.current_param)
            if schema is None:
                return False  # clave inventada, no está en el schema
            if schema.type in ("number", "int") and kind == 'string':
                return False
            # if schema.type == "enum":
            #     self.active_string_constraint = PrefixTrieConstraint(schema.enum_values)
            #     self.active_string_constraint.reset()
            else:
                self.active_string_constraint = None
        return True

    def _on_string_char(self, char: str) -> bool:
        if self.active_string_constraint:
            return self.active_string_constraint.feed(char)
        return True

    def _on_string_closed(self) -> bool:
        if self.phase == Phase.EXPECT_NAME:
            if not self.name_constraint.close():
                return False
            self.active_schema = self.function_schemas.get(
                self.name_constraint.buffer)
            self.phase = Phase.OTHER
            return self.active_schema is not None
        if self.active_string_constraint:
            return self.active_string_constraint.close()
        return True

    def check_step(self, string: str):
        return self.json.check_token(str)
