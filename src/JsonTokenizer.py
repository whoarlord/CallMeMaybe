class JsonTokenizer:
    START, OBJ_OPEN, KEY_STRING, AFTER_KEY, AFTER_COLON = range(5)
    STRING_VALUE, AFTER_VALUE, NUMBER, ARR_OPEN, DONE = range(5, 10)

    def __init__(self):
        self.state = self.START
        self.stack: list[str] = []
        self.blacklist: set[str] = {'\n', '\t'}

    def check_token(self, token: str) -> bool:
        """Intenta consumir un token completo, carácter por carácter."""
        if isinstance(token, str):
            token = token.replace('Ġ', ' ')
            for ch in token:
                if not self.step(ch):
                    return False
        return True

    def step(self, char: str) -> bool:
        if char in (' ', '\t', '\n') and self.state not in (self.KEY_STRING, self.STRING_VALUE):
            return True

        s = self.state

        if s == self.START:
            if char == '{':
                self.stack.append('{'); self.state = self.OBJ_OPEN; return True
            return False

        if s == self.OBJ_OPEN:
            if char == '"':
                self.state = self.KEY_STRING; return True
            if char == '}' and self.stack and self.stack[-1] == '{':
                self.stack.pop(); self.state = self.AFTER_VALUE if self.stack else self.DONE; return True
            return False

        if s == self.KEY_STRING:
            if char == '"':
                self.state = self.AFTER_KEY; return True
            if char in self.blacklist:
                return False
            return True

        if s == self.AFTER_KEY:
            if char == ':':
                self.state = self.AFTER_COLON; return True
            return False

        if s == self.AFTER_COLON:
            if char == '"':
                self.state = self.STRING_VALUE; return True
            if char == '{':
                self.stack.append('{'); self.state = self.OBJ_OPEN; return True
            if char == '[':
                self.stack.append('['); self.state = self.ARR_OPEN; return True
            if char.isdigit() or char == '-':
                self.state = self.NUMBER; return True
            if char in ('t', 'f', 'n'):
                self.state = self.AFTER_VALUE; return True
            return False

        if s == self.STRING_VALUE:
            if char == '"':
                self.state = self.AFTER_VALUE; return True
            if char in self.blacklist:
                return False
            return True

        if s == self.ARR_OPEN:
            if char == ']' and self.stack and self.stack[-1] == '[':
                self.stack.pop(); self.state = self.AFTER_VALUE if self.stack else self.DONE; return True
            if char == '{':
                self.stack.append('{'); self.state = self.OBJ_OPEN; return True
            if char == '[':
                self.stack.append('['); self.state = self.ARR_OPEN; return True
            if char == '"':
                self.state = self.STRING_VALUE; return True
            if char.isdigit() or char == '-':
                self.state = self.NUMBER; return True
            return False

        if s == self.NUMBER:
            if char.isdigit() or char in ('.'):
                return True
            return self._close(char)

        if s == self.AFTER_VALUE:
            return self._close(char)

        return False 

    def _close(self, char: str) -> bool:
        if char == ',':
            if self.stack and self.stack[-1] == '{':
                self.state = self.OBJ_OPEN; return True
            if self.stack and self.stack[-1] == '[':
                self.state = self.ARR_OPEN; return True
            return False
        if char == '}' and self.stack and self.stack[-1] == '{':
            self.stack.pop(); self.state = self.AFTER_VALUE if self.stack else self.DONE; return True
        if char == ']' and self.stack and self.stack[-1] == '[':
            self.stack.pop(); self.state = self.AFTER_VALUE if self.stack else self.DONE; return True
        return False

    def clone(self):
        result = JsonTokenizer()
        result.stack = self.stack.copy()
        result.state = self.state
        return result

    def print_tokenizer(self):
        print(f"state: {self.state}")
        print(f"stack: {self.stack}")