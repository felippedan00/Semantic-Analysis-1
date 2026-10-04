from __future__ import annotations

from ast_nodes import Program, TypeName
from semantic_errors import SemanticDiagnostic, SemanticError, SemanticErrorKind
from symbols import FunctionSymbol, SymbolKind


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
    # falta implementar escopos e resolver os nomes dentro dos corpos
    raise NotImplementedError("implemente a resolucao dos corpos das funcoes")
