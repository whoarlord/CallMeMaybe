from abc import ABC, abstractmethod


class ValueConstraint(ABC):
    """Common interface for value constraints.

    Decides, character by character, whether the value generated so
    far is still valid for its type.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the constraint to its initial state."""

    @abstractmethod
    def feed(self, char: str) -> bool:
        """Feed one character; return False if it is not allowed."""

    @abstractmethod
    def close(self) -> bool:
        """Finish the value.

        Called when the value is closed (closing quote, or end of a
        NUMBER). Return True if the accumulated value is valid.
        """

    @abstractmethod
    def clone(self) -> "ValueConstraint":
        """Return an independent copy of the constraint."""


class PrefixTrieConstraint(ValueConstraint):
    """Constrain the buffer to be a prefix of at least one candidate.

    When closed, the buffer must match exactly one of the candidates.
    """

    def __init__(self, candidates: list[str]) -> None:
        """Initialize the constraint.

        Args:
            candidates: Strings the final value is allowed to be.
        """
        self.candidates = candidates
        self.buffer = ""

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer = ""

    def feed(self, char: str) -> bool:
        """Accept the character only if it keeps a valid prefix."""
        candidate = self.buffer + char
        if not any(c.startswith(candidate) for c in self.candidates):
            return False
        self.buffer = candidate
        return True

    def close(self) -> bool:
        """Return True if the buffer is exactly one of the candidates."""
        return self.buffer in self.candidates

    def clone(self) -> "PrefixTrieConstraint":
        """Return an independent copy sharing the candidates list."""
        clone = PrefixTrieConstraint(self.candidates)
        clone.buffer = self.buffer
        return clone


class NumberConstraint(ValueConstraint):
    """Constraint for type 'number'.

    Allows digits, a single '.', and a single leading '-'.
    """

    MAX_INT_DIGITS = 15
    MAX_FRAC_DIGITS = 10

    def __init__(self, allow_decimal: bool = True) -> None:
        """Initialize the constraint.

        Args:
            allow_decimal: Whether a decimal point is accepted.
        """
        self.allow_decimal = allow_decimal
        self.buffer = ""

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer = ""

    def _parts(self) -> tuple[str, str | None]:
        """Split the buffer into integer and fractional parts.

        Returns:
            The integer part (without sign) and the fractional part,
            or None if no decimal point has been typed yet.
        """
        stripped = self.buffer.lstrip('-')
        if '.' in stripped:
            int_part, frac_part = stripped.split('.', 1)
            return int_part, frac_part
        return stripped, None

    def feed(self, char: str) -> bool:
        """Accept the character only if the number stays well formed."""
        int_part, frac_part = self._parts()

        if char == '-':
            if self.buffer != "":
                return False
        elif char == '.':
            if (not self.allow_decimal or
                    frac_part is not None or int_part == ""):
                return False
        elif char.isdigit():
            if frac_part is None:
                if len(int_part) >= self.MAX_INT_DIGITS:
                    return False
                if int_part == "0":
                    return False
            else:
                if len(frac_part) >= self.MAX_FRAC_DIGITS:
                    return False
        else:
            return False

        self.buffer += char
        return True

    def close(self) -> bool:
        """Return True if the buffer is a complete valid number."""
        stripped = self.buffer.lstrip('-')
        if self.allow_decimal:
            return (stripped != "" and stripped != "."
                    and stripped.find('.') != -1)
        return stripped != ""

    def clone(self) -> "NumberConstraint":
        """Return an independent copy of the constraint."""
        clone = NumberConstraint(self.allow_decimal)
        clone.buffer = self.buffer
        return clone


class IntConstraint(NumberConstraint):
    """Constraint for type 'integer': a number without decimals."""

    def __init__(self) -> None:
        """Initialize a number constraint with decimals disabled."""
        super().__init__(allow_decimal=False)

    def clone(self) -> "IntConstraint":
        """Return an independent copy of the constraint."""
        clone = IntConstraint()
        clone.buffer = self.buffer
        return clone


class FreeStringConstraint(ValueConstraint):
    """Constraint for an unrestricted 'string'.

    It adds no restriction of its own (JsonTokenizer already validates
    quotes, escapes and the blacklist).
    """

    def reset(self) -> None:
        """Do nothing: this constraint is stateless."""

    def feed(self, char: str) -> bool:
        """Accept any character."""
        return True

    def close(self) -> bool:
        """Always valid."""
        return True

    def clone(self) -> "FreeStringConstraint":
        """Return self, since the constraint has no state."""
        return self
