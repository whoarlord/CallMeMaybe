from typing import Callable


class JsonTokenizer:
    """Character-level JSON tokenizer driven by a small state machine.

    It validates a JSON object one character at a time and notifies the
    caller through callbacks, so that an external grammar can constrain
    keys and values while text is being generated.
    """

    START, OBJ_OPEN, KEY_STRING, AFTER_KEY, AFTER_COLON = range(5)
    STRING_VALUE, AFTER_VALUE, NUMBER, ARR_OPEN, DONE = range(5, 10)
    BOOL = 10

    ESCAPE_CHARS = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    def __init__(self,
                 on_key_closed: Callable[[str], None],
                 on_value_enter: Callable[[str], bool],
                 on_value_char: Callable[[str], bool],
                 on_value_closed: Callable[[], bool]) -> None:
        """Initialize the tokenizer.

        Args:
            on_key_closed: Called with the key when a key string ends.
            on_value_enter: Called with the value kind ('string',
                'number', 'object', 'array' or 'boolean') when a value
                starts. Return False to reject it.
            on_value_char: Called for each character of a value.
                Return False to reject the character.
            on_value_closed: Called when a value ends. Return False to
                reject the value.
        """
        self._on_key_closed = on_key_closed
        self._on_value_enter = on_value_enter
        self._on_value_char = on_value_char
        self._on_value_closed = on_value_closed
        self._key_buffer = ""
        self.state = self.AFTER_COLON
        self.stack: list[str] = ['{']
        self.blacklist: set[str] = {'\n', '\t'}
        self.escaped: bool = False
        self.no_more_parameters: bool = False
        self.ws_run = 0

    def check_token(self, token: str) -> bool:
        """Feed a whole token to the tokenizer.

        The 'Ġ' marker used by BPE tokenizers is treated as a space.
        On failure, only `escaped` and `ws_run` are restored; the rest
        of the state may have advanced, so use a clone to test tokens.

        Args:
            token: Text of the token to apply.

        Returns:
            True if every character was accepted, False otherwise.
        """
        escaped = self.escaped
        ws_run = self.ws_run
        token = token.replace('Ġ', ' ')
        for ch in token:
            if not self.step(ch, escaped):
                self.escaped = escaped
                self.ws_run = ws_run
                return False
            escaped = (ch == '\\') and not escaped
        self.escaped = escaped
        return True

    def step(self, char: str, escaped: bool = False) -> bool:
        """Advance the state machine by one character.

        Args:
            char: The character to process.
            escaped: Whether the previous character was an unescaped
                backslash.

        Returns:
            True if the character is valid in the current state.
        """
        if char in (' ', '\t', '\n'):
            if self.state not in (self.KEY_STRING, self.STRING_VALUE):
                if self.ws_run >= 1:
                    return False
                self.ws_run += 1
                return True
        self.ws_run = 0

        s = self.state

        if s == self.START:
            if char == '{':
                self.stack.append('{')
                self.state = self.OBJ_OPEN
                return True
            return False

        if s == self.OBJ_OPEN:
            if char == '"':
                self.state = self.KEY_STRING
                return True
            if char == '}' and self.stack and self.stack[-1] == '{':
                self.stack.pop()
                self.state = self.AFTER_VALUE if self.stack else self.DONE
                return True
            return False

        if s == self.KEY_STRING:
            if char == '"':
                self.state = self.AFTER_KEY
                self._on_key_closed(self._key_buffer)
                self._key_buffer = ""
                return True
            if char in self.blacklist:
                return False
            self._key_buffer += char
            return True

        if s == self.AFTER_KEY:
            if char == ':':
                self.state = self.AFTER_COLON
                return True
            return False

        if s == self.AFTER_COLON:
            if char == '"':
                if not self._on_value_enter('string'):
                    return False
                self.state = self.STRING_VALUE
                return True
            if char.isdigit() or char == '-':
                if not self._on_value_enter('number'):
                    return False
                if not self._on_value_char(char):
                    return False
                self.state = self.NUMBER
                return True
            if char == '{':
                if not self._on_value_enter('object'):
                    return False
                self.stack.append('{')
                self.state = self.OBJ_OPEN
                return True
            if char == '[':
                if not self._on_value_enter('array'):
                    return False
                self.stack.append('[')
                self.state = self.ARR_OPEN
                return True
            if char in ('t', 'f'):
                if not self._on_value_enter('boolean'):
                    return False
                if not self._on_value_char(char):
                    return False
                self.state = self.BOOL
                return True
            return False

        if s == self.STRING_VALUE:
            if escaped:
                if char not in self.ESCAPE_CHARS:
                    return False
                return True
            if char == '"':
                if not self._on_value_closed():
                    return False
                self.state = self.AFTER_VALUE
                return True
            if not self._on_value_char(char):
                return False
            if char in self.blacklist:
                return False
            return True

        if s == self.ARR_OPEN:
            if char == ']' and self.stack and self.stack[-1] == '[':
                self.stack.pop()
                self.state = self.AFTER_VALUE if self.stack else self.DONE
                return True
            if char == '{':
                self.stack.append('{')
                self.state = self.OBJ_OPEN
                return True
            if char == '[':
                self.stack.append('[')
                self.state = self.ARR_OPEN
                return True
            if char == '"':
                self.state = self.STRING_VALUE
                return True
            if char.isdigit() or char == '-':
                self.state = self.NUMBER
                return True
            return False

        if s == self.NUMBER:
            if char.isdigit() or char == '.':
                if not self._on_value_char(char):
                    return False
                return True
            if not self._on_value_closed():
                return False
            return self._close(char)

        if s == self.BOOL:
            if char.isalpha():
                if not self._on_value_char(char):
                    return False
                return True
            if not self._on_value_closed():
                return False
            return self._close(char)

        if s == self.AFTER_VALUE:
            return self._close(char)

        return False

    def _close(self, char: str) -> bool:
        """Handle the character that follows a completed value.

        Accepts a comma, or a closing brace/bracket that matches the
        top of the stack.
        """
        if char == ',' and not self.no_more_parameters:
            if self.stack and self.stack[-1] == '{':
                self.state = self.OBJ_OPEN
                return True
            if self.stack and self.stack[-1] == '[':
                self.state = self.ARR_OPEN
                return True
            return False
        if char == '}' and self.stack and self.stack[-1] == '{':
            self.stack.pop()
            self.state = self.AFTER_VALUE if self.stack else self.DONE
            return True
        if (char == ']' and self.stack and self.stack[-1] == '['
                and not self.no_more_parameters):
            self.stack.pop()
            self.state = self.AFTER_VALUE if self.stack else self.DONE
            return True
        return False

    def clone(self) -> "JsonTokenizer":
        """Return an independent copy of the tokenizer state.

        The callbacks are shared with the original instance.
        """
        result = JsonTokenizer(
            self._on_key_closed,
            self._on_value_enter,
            self._on_value_char,
            self._on_value_closed
        )
        result.stack = self.stack.copy()
        result.state = self.state
        result.escaped = self.escaped
        result.no_more_parameters = self.no_more_parameters
        result.ws_run = self.ws_run
        result._key_buffer = self._key_buffer
        return result
