from abc import ABC, abstractmethod


class ValueConstraint(ABC):
    """Interfaz común: decide, carácter a carácter, si el valor
    generado hasta ahora sigue siendo válido para su tipo."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def feed(self, char: str) -> bool: ...

    @abstractmethod
    def close(self) -> bool:
        """Se llama al cerrarse el valor (comilla, o fin de NUMBER)."""
        ...

    @abstractmethod
    def clone(self):
        ...


class PrefixTrieConstraint(ValueConstraint):
    """Restringe a que el buffer sea siempre prefijo de al menos un
    candidato, y que al cerrar coincida exactamente con uno."""

    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        self.buffer = ""

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, char: str) -> bool:
        candidate = self.buffer + char
        if not any(c.startswith(candidate) for c in self.candidates):
            return False
        self.buffer = candidate
        return True

    def close(self) -> bool:
        return self.buffer in self.candidates

    def clone(self):
        clone = PrefixTrieConstraint(self.candidates)
        clone.buffer = self.buffer
        return clone


class NumberConstraint(ValueConstraint):
    """Para type 'number': dígitos, un único '.', un único '-' inicial."""

    def __init__(self, allow_decimal: bool = True):
        self.allow_decimal = allow_decimal
        self.buffer = ""

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, char: str) -> bool:
        if char == '-':
            if self.buffer != "":
                return False  # solo válido como primer carácter
        elif char == '.':
            if not self.allow_decimal or '.' in self.buffer:
                return False
        elif not char.isdigit():
            return False
        self.buffer += char
        return True

    def close(self) -> bool:
        stripped = self.buffer.lstrip('-')
        return ((stripped != "" and stripped != ".") or
                (self.allow_decimal and stripped.find('.') != -1))

    def clone(self):
        clone = NumberConstraint(self.allow_decimal)
        clone.buffer = self.buffer
        return clone


class IntConstraint(NumberConstraint):
    def __init__(self):
        super().__init__(allow_decimal=False)


class FreeStringConstraint(ValueConstraint):
    """Para type 'string' libre: no añade restricción propia
    (el JsonTokenizer ya valida comillas/escapes/blacklist)."""

    def reset(self) -> None:
        pass

    def feed(self, char: str) -> bool:
        return True

    def close(self) -> bool:
        return True
    
    def clone(self):
        return self
