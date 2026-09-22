from enum import Enum
from .JsonTokenizer import JsonTokenizer
from .functionSchema import FunctionSchema
from .PrefixTrieConstraint import (
    PrefixTrieConstraint, NumberConstraint,
    IntConstraint, FreeStringConstraint,
)


class Phase(Enum):
    EXPECT_NAME = 1
    IN_ARGUMENTS = 2
    OTHER = 3


class FunctionCallGrammar:
    BOOL_CANDIDATES = ["true", "false"]

    def __init__(self, function_schemas: dict[str, FunctionSchema]):
        self.function_schemas = function_schemas
        self.json = JsonTokenizer(
            on_key_closed=self._on_key_closed,
            on_value_enter=self._on_value_enter,
            on_value_char=self._on_value_char,
            on_value_closed=self._on_value_closed,
        )
        self.phase = Phase.EXPECT_NAME
        self.active_schema: FunctionSchema | None = None
        self.current_param: str | None = None
        self.name_constraint = PrefixTrieConstraint(
            list(function_schemas.keys()))
        self.active_value_constraint = None

    def _on_key_closed(self, key: str) -> None:
        if key == "arguments":
            self.phase = Phase.IN_ARGUMENTS
        elif self.phase == Phase.IN_ARGUMENTS:
            self.current_param = key

    def _constraint_for_param(self, schema_type: str, kind: str):
        """Elige el constraint correcto según el
        tipo declarado en el schema."""
        if schema_type == "enum":
            param_schema = self.active_schema.parameters[self.current_param]
            return PrefixTrieConstraint(param_schema.enum_values)
        if schema_type == "number":
            return NumberConstraint()
        if schema_type == "int":
            return IntConstraint()
        if schema_type == "boolean":
            return PrefixTrieConstraint(self.BOOL_CANDIDATES)
        return FreeStringConstraint()

    def _on_value_enter(self, kind: str) -> bool:
        if self.phase == Phase.EXPECT_NAME:
            self.name_constraint.reset()
            self.active_value_constraint = self.name_constraint
            return kind == 'string'

        if self.phase == Phase.IN_ARGUMENTS:
            if self.active_schema is None:
                return False
            schema = self.active_schema.parameters.get(self.current_param)
            if schema is None:
                return False

            expects_string_syntax = schema.type in ("string", "enum")
            if expects_string_syntax != (kind == 'string'):
                return False

            self.active_value_constraint = self._constraint_for_param(
                schema.type, kind)
            self.active_value_constraint.reset()
            return True

        self.active_value_constraint = None
        return True

    def _on_value_char(self, char: str) -> bool:
        if self.active_value_constraint:
            return self.active_value_constraint.feed(char)
        return True

    def _on_value_closed(self) -> bool:
        if self.phase == Phase.EXPECT_NAME:
            if not self.name_constraint.close():
                return False
            self.active_schema = self.function_schemas.get(
                self.name_constraint.buffer)
            self.phase = Phase.OTHER
            return self.active_schema is not None
        if self.active_value_constraint:
            return self.active_value_constraint.close()
        return True

    def check_step(self, token: str) -> bool:
        temp_json_tokenizer = self.json.clone()
        return temp_json_tokenizer.check_token(token)

    def apply_token(self, token: str):
        self.json.check_token(token)
