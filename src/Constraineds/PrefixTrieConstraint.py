class PrefixTrieConstraint:
    """Restringe un string a ser, en todo momento, prefijo de al menos
    uno de los candidatos; y al cerrarse, coincidir con uno completo."""

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