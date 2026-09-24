class JsonTokenizer:
    START, OBJ_OPEN, KEY_STRING, AFTER_KEY, AFTER_COLON = range(5)
    STRING_VALUE, AFTER_VALUE, NUMBER, ARR_OPEN, DONE = range(5, 10)

    ESCAPE_CHARS = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    def __init__(self,
                 on_key_closed: callable,
                 on_value_enter: callable,
                 on_value_char: callable, 
                 on_value_closed: callable):
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

    def check_token(self, token: str) -> bool:
        """Check if a token is valid for a json output"""
        escaped = self.escaped
        if isinstance(token, str):
            token = token.replace('Ġ', ' ')
            for ch in token:
                if not self.step(ch, escaped):
                    self.escaped = escaped
                    return False
                escaped = (ch == '\\') and not escaped
        self.escaped = escaped
        return True

    def step(self, char: str, escaped: bool = False) -> bool:
        if (char in (' ', '\t', '\n')
                and self.state not in (self.KEY_STRING, self.STRING_VALUE)):
            return True

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
                if self._on_key_closed:
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
                if self._on_value_enter and not self._on_value_enter('string'):
                    return False
                self.state = self.STRING_VALUE
                return True
            if char.isdigit() or char == '-':
                if self._on_value_enter and not self._on_value_enter('number'):
                    return False
                if self._on_value_char and not self._on_value_char(char):
                    return False
                self.state = self.NUMBER
                return True
            if char == '{':
                self.stack.append('{')
                self.state = self.OBJ_OPEN
                return True
            if char == '[':
                self.stack.append('[')
                self.state = self.ARR_OPEN
                return True
            if char in ('t', 'f', 'n'):
                self.state = self.AFTER_VALUE
                return True
            return False

        if s == self.STRING_VALUE:
            if escaped:
                if char not in self.ESCAPE_CHARS:
                    return False
                return True
            if char == '"':
                if self._on_value_closed and not self._on_value_closed():
                    return False
                self.state = self.AFTER_VALUE
                return True
            if self._on_value_char and not self._on_value_char(char):
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
                if self._on_value_char and not self._on_value_char(char):
                    return False
                return True
            if self._on_value_closed and not self._on_value_closed():
                return False
            return self._close(char)

        if s == self.AFTER_VALUE:
            return self._close(char)

        return False

    def _close(self, char: str) -> bool:
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

    def clone(self):
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
        return result

    def print_tokenizer(self):
        print(f"state: {self.state}")
        print(f"stack: {self.stack}")
        print(f"escaped: {self.escaped}")
