from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    Block,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    Node,
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    TypeName,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, Scope, Symbol, SymbolKind


def resolve_names(program: Program) -> None:
    """Construa escopos, símbolos e vínculos entre usos e declarações."""

    diagnostics: list[SemanticDiagnostic] = []
    functions = _collect_functions(program, diagnostics)
    _validate_main(program, functions, diagnostics)
    _resolve_bodies(program, functions, diagnostics)

    # os erros so sao lancados depois de visitar todos os corpos
    if diagnostics:
        raise SemanticError(diagnostics)


def _collect_functions(
    program: Program,
    diagnostics: list[SemanticDiagnostic],
) -> dict[str, FunctionSymbol]:
    functions: dict[str, FunctionSymbol] = {}

    # registra as funcoes antes de analisar as chamadas
    for function in program.functions:
        symbol = FunctionSymbol(
            name=function.name,
            kind=SymbolKind.FUNCTION,
            type=function.return_type,
            declaration=function,
            parameter_types=tuple(parameter.type for parameter in function.parameters),
        )
        function.metadata["symbol"] = symbol

        if function.name in functions:
            diagnostics.append(SemanticDiagnostic(
                SemanticErrorKind.DUPLICATE_FUNCTION,
                f"funcao '{function.name}' ja declarada",
                function.span,
            ))
            # mantem a primeira funcao na tabela
            continue

        functions[function.name] = symbol

    return functions


def _validate_main(
    program: Program,
    functions: dict[str, FunctionSymbol],
    diagnostics: list[SemanticDiagnostic],
) -> None:
    main = functions.get("main")
    if main is None:
        diagnostics.append(SemanticDiagnostic(
            SemanticErrorKind.INVALID_MAIN,
            "funcao main nao declarada",
            program.span,
        ))
    elif main.type is not TypeName.INT or main.parameter_types:
        diagnostics.append(SemanticDiagnostic(
            SemanticErrorKind.INVALID_MAIN,
            "main deve retornar int e nao receber parametros",
            main.declaration.span,
        ))


def _resolve_bodies(
    program: Program,
    functions: dict[str, FunctionSymbol],
    diagnostics: list[SemanticDiagnostic],
) -> None:
    for function in program.functions:
        resolver = _BodyResolver(functions, diagnostics)
        resolver.resolve_function(function)


class _BodyResolver:
    """Percorre o corpo de uma funcao ligando cada uso a sua declaracao."""

    def __init__(
        self,
        functions: dict[str, FunctionSymbol],
        diagnostics: list[SemanticDiagnostic],
    ) -> None:
        self.functions = functions
        self.diagnostics = diagnostics
        self.scope: Scope | None = None

    def resolve_function(self, function: FunctionDecl) -> None:
        # parametros e corpo dividem o mesmo escopo
        scope = Scope(parent=None)
        function.body.metadata["scope"] = scope
        self.scope = scope

        for parameter in function.parameters:
            self.declare(parameter, parameter.name, parameter.type, SymbolKind.PARAMETER)

        for statement in function.body.statements:
            self.visit_statement(statement)

    def declare(self, node: Node, name: str, type: TypeName, kind: SymbolKind) -> None:
        symbol = Symbol(name=name, kind=kind, type=type, declaration=node)
        node.metadata["symbol"] = symbol

        if name in self.scope.symbols:
            self.error(
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"'{name}' ja declarado neste escopo",
                node,
            )
            return

        self.scope.symbols[name] = symbol

    def lookup(self, name: str) -> Symbol | None:
        # sobe do escopo atual ate o mais externo
        scope = self.scope
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        return None

    def visit_block(self, block: Block) -> None:
        # bloco interno ganha um escopo filho
        scope = Scope(parent=self.scope)
        block.metadata["scope"] = scope
        self.scope = scope

        for statement in block.statements:
            self.visit_statement(statement)

        self.scope = scope.parent

    def visit_statement(self, statement: Stmt) -> None:
        if isinstance(statement, Block):
            self.visit_block(statement)
        elif isinstance(statement, VarDecl):
            # declara antes de olhar o inicializador (int x = x;)
            self.declare(statement, statement.name, statement.type, SymbolKind.VARIABLE)
            if statement.initializer is not None:
                self.visit_expression(statement.initializer)
        elif isinstance(statement, Assignment):
            self.visit_expression(statement.target)
            self.visit_expression(statement.value)
        elif isinstance(statement, CallStmt):
            self.visit_expression(statement.call)
        elif isinstance(statement, IfStmt):
            self.visit_expression(statement.condition)
            self.visit_block(statement.then_block)
            if statement.else_block is not None:
                self.visit_block(statement.else_block)
        elif isinstance(statement, WhileStmt):
            self.visit_expression(statement.condition)
            self.visit_block(statement.body)
        elif isinstance(statement, ReturnStmt):
            if statement.value is not None:
                self.visit_expression(statement.value)
        elif isinstance(statement, PrintStmt):
            for item in statement.items:
                # strings nao tem nomes pra resolver
                if isinstance(item, Expr):
                    self.visit_expression(item)

    def visit_expression(self, expression: Expr) -> None:
        if isinstance(expression, IdentifierExpr):
            symbol = self.lookup(expression.name)
            if symbol is None:
                self.error(
                    SemanticErrorKind.UNDECLARED_VARIABLE,
                    f"variavel '{expression.name}' nao declarada",
                    expression,
                )
            else:
                expression.metadata["symbol"] = symbol
        elif isinstance(expression, CallExpr):
            # funcoes so ficam na tabela global
            symbol = self.functions.get(expression.name)
            if symbol is None:
                self.error(
                    SemanticErrorKind.UNDECLARED_FUNCTION,
                    f"funcao '{expression.name}' nao declarada",
                    expression,
                )
            else:
                expression.metadata["symbol"] = symbol
            for argument in expression.arguments:
                self.visit_expression(argument)
        elif isinstance(expression, UnaryExpr):
            self.visit_expression(expression.operand)
        elif isinstance(expression, BinaryExpr):
            self.visit_expression(expression.left)
            self.visit_expression(expression.right)
        # literais nao precisam de nada

    def error(self, kind: SemanticErrorKind, message: str, node: Node) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, node.span))
