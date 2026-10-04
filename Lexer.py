from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from typing import Iterator


# O limite dos inteiros será validado na análise semântica.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


class TokenKind(enum.Enum):
    """Classe já implementada: nomes e números não devem ser alterados."""

    EOF = -1

    IDENTIFIER = 1
    INT_LITERAL = 2
    STRING_LITERAL = 3

    KW_INT = 10
    KW_BOOL = 11
    KW_VOID = 12
    KW_TRUE = 13
    KW_FALSE = 14
    KW_IF = 15
    KW_ELSE = 16
    KW_WHILE = 17
    KW_RETURN = 18
    KW_PRINT = 19

    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    LESS = 25
    LESS_EQUAL = 26
    GREATER = 27
    GREATER_EQUAL = 28
    EQUAL_EQUAL = 29
    NOT_EQUAL = 30
    LOGICAL_AND = 31
    LOGICAL_OR = 32
    LOGICAL_NOT = 33
    ASSIGN = 34

    LEFT_PAREN = 40
    RIGHT_PAREN = 41
    LEFT_BRACE = 42
    RIGHT_BRACE = 43
    COMMA = 44
    SEMICOLON = 45


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: int | str | bool | None
    line: int
    column: int

    def __str__(self) -> str:
        return (
            f"<{self.kind.value}, {self.kind.name}, {self.lexeme!r}, "
            f"{self.value!r}, {self.line}, {self.column}>"
        )


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"erro léxico em {self.line}:{self.column}: {self.message}"


_KEYWORDS: dict[str, TokenKind] = {
    "int": TokenKind.KW_INT,
    "bool": TokenKind.KW_BOOL,
    "void": TokenKind.KW_VOID,
    "true": TokenKind.KW_TRUE,
    "false": TokenKind.KW_FALSE,
    "if": TokenKind.KW_IF,
    "else": TokenKind.KW_ELSE,
    "while": TokenKind.KW_WHILE,
    "return": TokenKind.KW_RETURN,
    "print": TokenKind.KW_PRINT,
}

_SIMPLE_TOKENS: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "%": TokenKind.PERCENT,
    "(": TokenKind.LEFT_PAREN,
    ")": TokenKind.RIGHT_PAREN,
    "{": TokenKind.LEFT_BRACE,
    "}": TokenKind.RIGHT_BRACE,
    ",": TokenKind.COMMA,
    ";": TokenKind.SEMICOLON,
}

_ESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    '"': '"',
    "\\": "\\",
}


def _is_ident_start(c: str) -> bool:
    return c == "_" or ("a" <= c <= "z") or ("A" <= c <= "Z")


def _is_ident_continue(c: str) -> bool:
    return _is_ident_start(c) or ("0" <= c <= "9")


def _is_digit(c: str) -> bool:
    return "0" <= c <= "9"


class Lexer:
    """Converte texto-fonte MicroC em uma sequência de tokens."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self._length = len(source)

    def _peek(self, offset: int = 0) -> str | None:
        idx = self.pos + offset
        if idx >= self._length:
            return None
        return self.source[idx]

    def _advance(self) -> str:
        c = self.source[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return c

    def _skip_trivia(self) -> None:
        while self.pos < self._length:
            c = self.source[self.pos]
            if c in " \t\r\n":
                self._advance()
                continue
            if c == "/" and self._peek(1) == "/":
                self._skip_line_comment()
                continue
            if c == "/" and self._peek(1) == "*":
                self._skip_block_comment()
                continue
            break

    def _skip_line_comment(self) -> None:
        self._advance()  # primeira '/'
        self._advance()  # segunda '/'
        while self.pos < self._length and self.source[self.pos] != "\n":
            c = self.source[self.pos]
            if ord(c) > 127:
                raise LexerError("caractere não ASCII", self.line, self.column)
            self._advance()

    def _skip_block_comment(self) -> None:
        start_line, start_col = self.line, self.column
        self._advance()  # '/'
        self._advance()  # '*'
        while True:
            if self.pos >= self._length:
                raise LexerError(
                    "comentário de bloco não terminado", start_line, start_col
                )
            c = self.source[self.pos]
            if ord(c) > 127:
                raise LexerError("caractere não ASCII", self.line, self.column)
            if c == "*" and self._peek(1) == "/":
                self._advance()
                self._advance()
                return
            self._advance()

    def _scan_identifier(self, line: int, column: int) -> Token:
        start = self.pos
        while self.pos < self._length and _is_ident_continue(self.source[self.pos]):
            self._advance()
        lexeme = self.source[start : self.pos]
        kind = _KEYWORDS.get(lexeme)
        if kind is TokenKind.KW_TRUE:
            return Token(kind, lexeme, True, line, column)
        if kind is TokenKind.KW_FALSE:
            return Token(kind, lexeme, False, line, column)
        if kind is not None:
            return Token(kind, lexeme, None, line, column)
        return Token(TokenKind.IDENTIFIER, lexeme, lexeme, line, column)

    def _scan_number(self, line: int, column: int) -> Token:
        start = self.pos
        while self.pos < self._length and _is_digit(self.source[self.pos]):
            self._advance()
        lexeme = self.source[start : self.pos]
        return Token(TokenKind.INT_LITERAL, lexeme, int(lexeme), line, column)

    def _scan_string(self, line: int, column: int) -> Token:
        start_line, start_col = line, column
        self._advance()  # aspa de abertura
        raw: list[str] = ['"']
        decoded: list[str] = []
        while True:
            if self.pos >= self._length:
                raise LexerError(
                    "string não terminada antes do fim do arquivo",
                    start_line,
                    start_col,
                )
            c = self.source[self.pos]
            if c == "\n" or c == "\r":
                raise LexerError(
                    "quebra de linha em string não terminada",
                    self.line,
                    self.column,
                )
            if ord(c) > 127:
                raise LexerError(
                    "caractere não ASCII em string", self.line, self.column
                )
            if c == '"':
                self._advance()
                raw.append('"')
                break
            if c == "\\":
                esc_line, esc_col = self.line, self.column
                self._advance()
                raw.append("\\")
                if self.pos >= self._length:
                    raise LexerError(
                        "string não terminada antes do fim do arquivo",
                        start_line,
                        start_col,
                    )
                e = self.source[self.pos]
                if e == "\n" or e == "\r":
                    raise LexerError(
                        "quebra de linha em string não terminada",
                        self.line,
                        self.column,
                    )
                if e not in _ESCAPES:
                    raise LexerError(
                        "sequência de escape inválida", esc_line, esc_col
                    )
                self._advance()
                raw.append(e)
                decoded.append(_ESCAPES[e])
                continue
            self._advance()
            raw.append(c)
            decoded.append(c)
        return Token(
            TokenKind.STRING_LITERAL,
            "".join(raw),
            "".join(decoded),
            start_line,
            start_col,
        )

    def _scan_operator(self, line: int, column: int) -> Token:
        c = self._advance()

        if c == "<":
            if self._peek() == "=":
                self._advance()
                return Token(TokenKind.LESS_EQUAL, "<=", None, line, column)
            return Token(TokenKind.LESS, "<", None, line, column)
        if c == ">":
            if self._peek() == "=":
                self._advance()
                return Token(TokenKind.GREATER_EQUAL, ">=", None, line, column)
            return Token(TokenKind.GREATER, ">", None, line, column)
        if c == "=":
            if self._peek() == "=":
                self._advance()
                return Token(TokenKind.EQUAL_EQUAL, "==", None, line, column)
            return Token(TokenKind.ASSIGN, "=", None, line, column)
        if c == "!":
            if self._peek() == "=":
                self._advance()
                return Token(TokenKind.NOT_EQUAL, "!=", None, line, column)
            return Token(TokenKind.LOGICAL_NOT, "!", None, line, column)
        if c == "&":
            if self._peek() == "&":
                self._advance()
                return Token(TokenKind.LOGICAL_AND, "&&", None, line, column)
            raise LexerError("uso isolado de '&'", line, column)
        if c == "|":
            if self._peek() == "|":
                self._advance()
                return Token(TokenKind.LOGICAL_OR, "||", None, line, column)
            raise LexerError("uso isolado de '|'", line, column)
        if c in _SIMPLE_TOKENS:
            return Token(_SIMPLE_TOKENS[c], c, None, line, column)

        raise LexerError(f"caractere inválido {c!r}", line, column)

    def tokens(self) -> Iterator[Token]:
        """Produza todos os tokens significativos e um único EOF ao final."""
        self.pos = 0
        self.line = 1
        self.column = 1

        while True:
            self._skip_trivia()

            if self.pos >= self._length:
                yield Token(TokenKind.EOF, "", None, self.line, self.column)
                return

            start_line, start_col = self.line, self.column
            c = self.source[self.pos]

            if ord(c) > 127:
                raise LexerError("caractere não ASCII", start_line, start_col)

            if _is_ident_start(c):
                yield self._scan_identifier(start_line, start_col)
                continue
            if _is_digit(c):
                yield self._scan_number(start_line, start_col)
                continue
            if c == '"':
                yield self._scan_string(start_line, start_col)
                continue

            yield self._scan_operator(start_line, start_col)

    def scan(self) -> list[Token]:
        return list(self.tokens())
