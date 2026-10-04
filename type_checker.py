from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    BinaryOperator,
    Block,
    BoolLiteral,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    Node,
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind


# obs: None faz o papel de tipo desconhecido, pra evitar erro em cascata

MAX_INT = 2**63 - 1

ARITHMETIC = {
    BinaryOperator.ADD,
    BinaryOperator.SUBTRACT,
    BinaryOperator.MULTIPLY,
    BinaryOperator.DIVIDE,
    BinaryOperator.REMAINDER,
}
RELATIONAL = {
    BinaryOperator.LESS,
    BinaryOperator.LESS_EQUAL,
    BinaryOperator.GREATER,
    BinaryOperator.GREATER_EQUAL,
}
EQUALITY = {BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL}
LOGICAL = {BinaryOperator.LOGICAL_AND, BinaryOperator.LOGICAL_OR}


def check_types(program: Program) -> None:
    """Determine tipos de expressões e valide seus contextos."""

    checker = _TypeChecker()
    for function in program.functions:
        checker.check_function(function)

    if checker.diagnostics:
        raise SemanticError(checker.diagnostics)


class _TypeChecker:
    """Calcula o tipo de cada expressao e confere os contextos de uso."""

    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = []
        self.return_type = TypeName.VOID

    def check_function(self, function: FunctionDecl) -> None:
        self.return_type = function.return_type

        for parameter in function.parameters:
            if parameter.type is TypeName.VOID:
                self.error(
                    SemanticErrorKind.VOID_PARAMETER,
                    f"parametro '{parameter.name}' nao pode ser void",
                    parameter,
                )

        self.check_block(function.body)

    def check_block(self, block: Block) -> None:
        for statement in block.statements:
            self.check_statement(statement)

    def check_statement(self, statement: Stmt) -> None:
        if isinstance(statement, Block):
            self.check_block(statement)
        elif isinstance(statement, VarDecl):
            self.check_var_decl(statement)
        elif isinstance(statement, Assignment):
            target = self.value(statement.target)
            value = self.value(statement.value)
            if target is not None and value is not None and target is not value:
                self.error(
                    SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                    f"esperado {target.value}, recebido {value.value}",
                    statement.value,
                )
        elif isinstance(statement, CallStmt):
            # aqui chamada void e permitida, o valor e descartado
            self.check_expression(statement.call)
        elif isinstance(statement, IfStmt):
            self.check_condition(statement.condition)
            self.check_block(statement.then_block)
            if statement.else_block is not None:
                self.check_block(statement.else_block)
        elif isinstance(statement, WhileStmt):
            self.check_condition(statement.condition)
            self.check_block(statement.body)
        elif isinstance(statement, ReturnStmt):
            self.check_return(statement)
        elif isinstance(statement, PrintStmt):
            for item in statement.items:
                # string passa direto, expressao precisa ter valor
                if isinstance(item, Expr):
                    self.value(item)

    def check_var_decl(self, declaration: VarDecl) -> None:
        if declaration.type is TypeName.VOID:
            self.error(
                SemanticErrorKind.VOID_VARIABLE,
                f"variavel '{declaration.name}' nao pode ser void",
                declaration,
            )

        if declaration.initializer is None:
            return

        value = self.value(declaration.initializer)
        # se a variavel e void o erro ja foi dado acima
        if (
            declaration.type is not TypeName.VOID
            and value is not None
            and value is not declaration.type
        ):
            self.error(
                SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                f"esperado {declaration.type.value}, recebido {value.value}",
                declaration.initializer,
            )

    def check_condition(self, condition: Expr) -> None:
        kind = self.value(condition)
        if kind is not None and kind is not TypeName.BOOL:
            self.error(
                SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                f"condicao deve ser bool, recebido {kind.value}",
                condition,
            )

    def check_return(self, statement: ReturnStmt) -> None:
        if statement.value is None:
            if self.return_type is not TypeName.VOID:
                self.error(
                    SemanticErrorKind.RETURN_MISMATCH,
                    f"funcao deve retornar {self.return_type.value}",
                    statement,
                )
            return

        value = self.value(statement.value)
        if self.return_type is TypeName.VOID:
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                "funcao void nao retorna valor",
                statement.value,
            )
        elif value is not None and value is not self.return_type:
            self.error(
                SemanticErrorKind.RETURN_MISMATCH,
                f"esperado {self.return_type.value}, recebido {value.value}",
                statement.value,
            )

    def value(self, expression: Expr) -> TypeName | None:
        # mesma coisa que check_expression, mas void nao pode virar valor
        kind = self.check_expression(expression)
        if kind is TypeName.VOID:
            self.error(
                SemanticErrorKind.VOID_VALUE_USED,
                "chamada void usada como valor",
                expression,
            )
            return None
        return kind

    def check_expression(self, expression: Expr) -> TypeName | None:
        kind = self.compute_type(expression)
        # so anota quando o tipo e conhecido
        if kind is not None:
            expression.metadata["type"] = kind
        return kind

    def compute_type(self, expression: Expr) -> TypeName | None:
        if isinstance(expression, IntLiteral):
            if not 0 <= expression.value <= MAX_INT:
                self.error(
                    SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                    f"literal {expression.value} fora do intervalo",
                    expression,
                )
            return TypeName.INT
        if isinstance(expression, BoolLiteral):
            return TypeName.BOOL
        if isinstance(expression, IdentifierExpr):
            kind = expression.metadata["symbol"].type
            # variavel void ja deu erro na declaracao
            return None if kind is TypeName.VOID else kind
        if isinstance(expression, UnaryExpr):
            return self.check_unary(expression)
        if isinstance(expression, BinaryExpr):
            return self.check_binary(expression)
        if isinstance(expression, CallExpr):
            return self.check_call(expression)
        return None

    def check_unary(self, expression: UnaryExpr) -> TypeName | None:
        operand = self.value(expression.operand)
        if operand is None:
            return None

        expected = TypeName.INT if expression.operator is UnaryOperator.NEGATE else TypeName.BOOL
        if operand is not expected:
            self.error(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                f"'{expression.operator.value}' espera {expected.value}",
                expression,
            )
            return None
        return expected

    def check_binary(self, expression: BinaryExpr) -> TypeName | None:
        # visita os dois lados antes de decidir
        left = self.value(expression.left)
        right = self.value(expression.right)
        if left is None or right is None:
            return None

        operator = expression.operator
        if operator in ARITHMETIC:
            valid, result = left is right is TypeName.INT, TypeName.INT
        elif operator in RELATIONAL:
            valid, result = left is right is TypeName.INT, TypeName.BOOL
        elif operator in EQUALITY:
            valid, result = left is right, TypeName.BOOL
        else:
            valid, result = left is right is TypeName.BOOL, TypeName.BOOL

        if not valid:
            self.error(
                SemanticErrorKind.INVALID_BINARY_OPERANDS,
                f"operandos invalidos para '{operator.value}': "
                f"{left.value} e {right.value}",
                expression,
            )
            return None
        return result

    def check_call(self, call: CallExpr) -> TypeName | None:
        symbol = call.metadata["symbol"]
        # todos os argumentos sao visitados, mesmo com aridade errada
        arguments = [self.value(argument) for argument in call.arguments]

        if len(arguments) != len(symbol.parameter_types):
            self.error(
                SemanticErrorKind.ARITY_MISMATCH,
                f"'{call.name}' espera {len(symbol.parameter_types)} "
                f"argumentos, recebeu {len(arguments)}",
                call,
            )

        # zip para no menor, entao so compara os que existem dos dois lados
        for argument, kind, expected in zip(call.arguments, arguments, symbol.parameter_types):
            if kind is not None and expected is not TypeName.VOID and kind is not expected:
                self.error(
                    SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                    f"esperado {expected.value}, recebido {kind.value}",
                    argument,
                )

        return symbol.type

    def error(self, kind: SemanticErrorKind, message: str, node: Node) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, node.span))
