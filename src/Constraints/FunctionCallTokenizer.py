from enum import Enum
from typing import Protocol

from .JsonTokenizer import JsonTokenizer
from ..PydanticModels import FunctionDef
from .PrefixTrieConstraint import (
    PrefixTrieConstraint, NumberConstraint,
    IntConstraint, FreeStringConstraint,
)


class ValueConstraint(Protocol):
    """Structural interface shared by all value constraints."""

    def reset(self) -> None:
        """Reset the constraint to its initial state."""

    def feed(self, char: str) -> bool:
        """Feed one character; return False if it is not allowed."""

    def close(self) -> bool:
        """Finish the value; return True if it is valid."""

    def clone(self) -> "ValueConstraint":
        """Return an independent copy of the constraint."""


class Phase(Enum):
    """Parsing phase of the function call being generated."""

    EXPECT_NAME = 1
    IN_PARAMETERS = 2
    OTHER = 3


class FunctionCallGrammar:
    """Grammar that constrains token generation to a valid function call.

    It drives a JSON tokenizer and applies per-value constraints based
    on the function schemas (function name, parameter names and types).
    """

    BOOL_CANDIDATES = ["true", "false"]

    def __init__(self, function_schemas: dict[str, FunctionDef]) -> None:
        """Initialize the grammar.

        Args:
            function_schemas: Mapping from function name to its schema.
        """
        self.function_schemas = function_schemas
        self.json = JsonTokenizer(
            on_key_closed=self._on_key_closed,
            on_value_enter=self._on_value_enter,
            on_value_char=self._on_value_char,
            on_value_closed=self._on_value_closed,
        )
        self.phase = Phase.EXPECT_NAME
        self.active_schema: FunctionDef | None = None
        self.current_param: str | None = None
        self.name_constraint = PrefixTrieConstraint(
            list(function_schemas.keys()))
        self.active_value_constraint: ValueConstraint | None = None
        self.seen_arg_values: list[str] = []

    def _on_key_closed(self, key: str) -> None:
        """Update the phase and current parameter when a key is closed."""
        if key == "parameters":
            self.phase = Phase.IN_PARAMETERS
            self.current_param = None
        elif self.phase == Phase.IN_PARAMETERS:
            self.current_param = key

    def _constraint_for_param(self, schema_type: str) -> ValueConstraint:
        """Pick the right constraint for the type declared in the schema."""
        schema_type = schema_type.lower()
        if schema_type == "number":
            return NumberConstraint()
        if schema_type == "integer":
            return IntConstraint()
        if schema_type == "boolean":
            return PrefixTrieConstraint(self.BOOL_CANDIDATES)
        return FreeStringConstraint()

    def _on_value_enter(self, kind: str) -> bool:
        """Decide whether a value of the given kind may start here.

        Args:
            kind: JSON kind of the value (object, array, string, ...).

        Returns:
            True if the value is allowed in the current phase.
        """
        if kind in ('object', 'array'):
            if self.phase == Phase.IN_PARAMETERS:
                return kind == 'object' and self.current_param is None
            return True

        if self.phase == Phase.EXPECT_NAME:
            self.name_constraint.reset()
            self.active_value_constraint = self.name_constraint
            return kind == 'string'

        if self.phase == Phase.IN_PARAMETERS:
            if self.active_schema is None or self.current_param is None:
                return False

            schema = self.active_schema.parameters.get(self.current_param)
            if schema is None:
                return False

            t = schema.type.lower()
            if t == 'integer' and kind == 'number':
                kind = 'integer'
            expected = {
                "string": "string", "enum": "string",
                "number": "number", "num": "number",
                "integer": "integer", "int": "integer",
                "boolean": "boolean", "bool": "boolean",
            }.get(t, "string")

            if kind != expected:
                return False
            constraint = self._constraint_for_param(expected)
            constraint.reset()
            self.active_value_constraint = constraint
            return True

        self.active_value_constraint = None
        return True

    def _on_value_char(self, char: str) -> bool:
        """Validate one character of the current value."""
        if self.active_value_constraint:
            return self.active_value_constraint.feed(char)
        return True

    def _on_value_closed(self) -> bool:
        """Validate and finalize the value that has just been closed."""
        if self.phase == Phase.EXPECT_NAME:
            if not self.name_constraint.close():
                return False
            self.active_schema = self.function_schemas.get(
                self.name_constraint.buffer)
            self.phase = Phase.OTHER
            return self.active_schema is not None
        if self.phase == Phase.IN_PARAMETERS:
            if self.active_schema is None or self.current_param is None:
                return False
            self.seen_arg_values.append(self.current_param)
            expected_count = len(self.active_schema.parameters)
            if len(self.seen_arg_values) >= expected_count:
                self.json.no_more_parameters = True
            if self.active_value_constraint:
                return self.active_value_constraint.close()
        return True

    def clone(self) -> "FunctionCallGrammar":
        """Return an independent copy of the current grammar state.

        The function schemas are shared (read-only), everything else
        is copied.
        """
        result = FunctionCallGrammar(self.function_schemas)

        result.json.stack = self.json.stack.copy()
        result.json.state = self.json.state
        result.json.escaped = self.json.escaped
        result.json.no_more_parameters = self.json.no_more_parameters
        result.json.ws_run = self.json.ws_run
        result.json._key_buffer = self.json._key_buffer

        result.phase = self.phase
        result.active_schema = self.active_schema
        result.current_param = self.current_param
        result.seen_arg_values = self.seen_arg_values.copy()

        result.name_constraint.buffer = self.name_constraint.buffer
        result.active_value_constraint = (
            self.active_value_constraint.clone()
            if self.active_value_constraint is not None
            else None
        )
        return result

    def check_step(self, token: str) -> bool:
        """Check whether a token is valid without modifying the state."""
        temp = self.clone()
        return temp.apply_token(token)

    def apply_token(self, token: str) -> bool:
        """Apply a token to the grammar; return False if it is invalid."""
        return self.json.check_token(token)
