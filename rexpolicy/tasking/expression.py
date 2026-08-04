"""Safe typed compilation and stack evaluation for task expressions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint
from .capabilities import EventSchema, MetricSpec, ParameterCapability
from .immutable import freeze_json
from .model import ExpressionSpec

NUMBER = "number"
BOOLEAN = "boolean"
BOOLEAN_PURPOSES = frozenset(("goal", "failure", "safety"))
EXPRESSION_PURPOSES = frozenset((*BOOLEAN_PURPOSES, "reward", "process"))


@dataclass(frozen=True)
class CompileLimits:
    max_nodes: int = 128
    max_instructions: int = 256
    max_stack: int = 64
    minimum_abs_divisor: float = 1.0e-12

    def validate(self) -> None:
        if min(self.max_nodes, self.max_instructions, self.max_stack) < 1:
            raise ValueError("Expression compile limits must be positive")
        if (
            not math.isfinite(self.minimum_abs_divisor)
            or self.minimum_abs_divisor <= 0.0
        ):
            raise ValueError("minimum_abs_divisor must be positive and finite")


@dataclass(frozen=True)
class NumericInterval:
    minimum: float | None
    maximum: float | None

    @classmethod
    def exact(cls, value: float) -> NumericInterval:
        return cls(value, value)


@dataclass(frozen=True)
class MetricOperand:
    name: str
    at: str

    def to_record(self) -> dict[str, str]:
        return {"name": self.name, "at": self.at}


@dataclass(frozen=True)
class ParameterOperand:
    name: str
    value: float
    unit: str

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class Instruction:
    opcode: str
    operand: Any = None

    def __post_init__(self) -> None:
        if self.opcode == "load_metric":
            valid = isinstance(self.operand, MetricOperand)
        elif self.opcode == "load_parameter":
            valid = isinstance(self.operand, ParameterOperand)
        elif self.opcode == "push_const":
            valid = (
                not isinstance(self.operand, bool)
                and isinstance(self.operand, (int, float))
                and math.isfinite(float(self.operand))
            )
        elif self.opcode in {"all", "any"}:
            valid = type(self.operand) is int and self.operand >= 1
        else:
            valid = self.operand is None
        if not valid:
            raise ValueError(f"Expression opcode {self.opcode!r} has invalid operand")

    def to_record(self) -> dict[str, Any]:
        record = {"opcode": self.opcode}
        if self.operand is not None:
            record["operand"] = (
                self.operand.to_record()
                if isinstance(self.operand, (MetricOperand, ParameterOperand))
                else self.operand
            )
        return record


@dataclass(frozen=True)
class MetricFrame:
    """Reward-agnostic previous/current scalar event values."""

    event_schema_id: str
    event_schema_fingerprint: str
    previous: Mapping[str, Any]
    current: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.previous, Mapping) or not isinstance(
            self.current,
            Mapping,
        ):
            raise ValueError("Metric frame values must be mappings")
        object.__setattr__(self, "previous", freeze_json(self.previous))
        object.__setattr__(self, "current", freeze_json(self.current))


@dataclass(frozen=True)
class ExpressionProgram:
    bytecode_version: int
    purpose: str
    result_type: str
    result_unit: str
    task_spec_fingerprint: str
    event_schema_id: str
    event_schema_fingerprint: str
    instructions: tuple[Instruction, ...]
    referenced_metrics: tuple[str, ...]
    max_stack: int
    result_interval: NumericInterval | None

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_type": "rexpolicy_expression_program",
            "program_schema_version": 1,
            "bytecode_version": self.bytecode_version,
            "purpose": self.purpose,
            "result_type": self.result_type,
            "result_unit": self.result_unit,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "event_schema_id": self.event_schema_id,
            "event_schema_fingerprint": self.event_schema_fingerprint,
            "instructions": [item.to_record() for item in self.instructions],
            "referenced_metrics": list(self.referenced_metrics),
            "max_stack": self.max_stack,
            "result_interval": (
                None
                if self.result_interval is None
                else {
                    "minimum": self.result_interval.minimum,
                    "maximum": self.result_interval.maximum,
                }
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass
class _Compiled:
    value_type: str
    unit: str
    interval: NumericInterval | None
    instructions: list[Instruction]
    referenced_metrics: set[str]
    nodes: int


def _arity(expression: ExpressionSpec, expected: int | None) -> tuple[ExpressionSpec, ...]:
    args = expression.payload.get("args")
    if not isinstance(args, tuple):
        raise ValueError(f"Expression op {expression.op!r} requires args")
    if expected is not None and len(args) != expected:
        raise ValueError(
            f"Expression op {expression.op!r} requires {expected} args"
        )
    return args


def _require_type(compiled: _Compiled, value_type: str, op: str) -> None:
    if compiled.value_type != value_type:
        raise ValueError(f"Expression op {op!r} requires {value_type} operands")


def _require_same_unit(left: _Compiled, right: _Compiled, op: str) -> str:
    if left.unit != right.unit:
        raise ValueError(
            f"Expression op {op!r} mixes units {left.unit!r} and {right.unit!r}"
        )
    return left.unit


def _bounded_binary(
    left: NumericInterval | None,
    right: NumericInterval | None,
    op: str,
) -> NumericInterval:
    if (
        left is None
        or right is None
        or left.minimum is None
        or left.maximum is None
        or right.minimum is None
        or right.maximum is None
    ):
        return NumericInterval(None, None)
    if op == "add":
        return NumericInterval(
            left.minimum + right.minimum,
            left.maximum + right.maximum,
        )
    if op == "sub":
        return NumericInterval(
            left.minimum - right.maximum,
            left.maximum - right.minimum,
        )
    if op == "mul":
        products = (
            left.minimum * right.minimum,
            left.minimum * right.maximum,
            left.maximum * right.minimum,
            left.maximum * right.maximum,
        )
        return NumericInterval(min(products), max(products))
    if op == "min":
        return NumericInterval(
            min(left.minimum, right.minimum),
            min(left.maximum, right.maximum),
        )
    if op == "max":
        return NumericInterval(
            max(left.minimum, right.minimum),
            max(left.maximum, right.maximum),
        )
    raise AssertionError(op)


def _compile(
    expression: ExpressionSpec,
    *,
    schema: EventSchema,
    parameters: Mapping[str, Any],
    parameter_capabilities: Mapping[str, ParameterCapability],
    limits: CompileLimits,
) -> _Compiled:
    op = expression.op
    if op == "const":
        value = float(expression.payload["value"])
        return _Compiled(
            value_type=NUMBER,
            unit="1",
            interval=NumericInterval.exact(value),
            instructions=[Instruction("push_const", value)],
            referenced_metrics=set(),
            nodes=1,
        )
    if op == "metric":
        name = expression.payload["name"]
        at = expression.payload["at"]
        try:
            metric = schema.by_name[name]
        except KeyError as error:
            raise ValueError(f"Expression references unknown metric {name!r}") from error
        if at not in metric.available_at:
            raise ValueError(f"Metric {name!r} is unavailable at {at!r}")
        interval = (
            NumericInterval(metric.minimum, metric.maximum)
            if metric.value_type == NUMBER
            else None
        )
        return _Compiled(
            value_type=metric.value_type,
            unit=metric.unit,
            interval=interval,
            instructions=[
                Instruction("load_metric", MetricOperand(name=name, at=at))
            ],
            referenced_metrics={name},
            nodes=1,
        )
    if op == "parameter":
        name = expression.payload["name"]
        if name not in parameters:
            raise ValueError(f"Expression references unknown parameter {name!r}")
        try:
            capability = parameter_capabilities[name]
        except KeyError as error:
            raise ValueError(
                f"Expression parameter {name!r} is not in the trusted capability"
            ) from error
        value = parameters[name]
        if capability.value_type != NUMBER:
            raise ValueError(f"Expression parameter {name!r} is not scalar numeric")
        capability.validate_value(value, f"parameter.{name}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Expression parameter {name!r} is non-finite")
        return _Compiled(
            value_type=NUMBER,
            unit=capability.unit,
            interval=NumericInterval.exact(number),
            instructions=[
                Instruction(
                    "load_parameter",
                    ParameterOperand(name=name, value=number, unit=capability.unit),
                )
            ],
            referenced_metrics=set(),
            nodes=1,
        )

    unary_number = {"neg", "abs"}
    if op in unary_number:
        child = _compile(
            _arity(expression, 1)[0],
            schema=schema,
            parameters=parameters,
            parameter_capabilities=parameter_capabilities,
            limits=limits,
        )
        _require_type(child, NUMBER, op)
        interval = child.interval or NumericInterval(None, None)
        if op == "neg":
            result_interval = NumericInterval(
                None if interval.maximum is None else -interval.maximum,
                None if interval.minimum is None else -interval.minimum,
            )
        elif interval.minimum is None or interval.maximum is None:
            result_interval = NumericInterval(0.0, None)
        else:
            values = (abs(interval.minimum), abs(interval.maximum))
            result_interval = NumericInterval(
                0.0 if interval.minimum <= 0.0 <= interval.maximum else min(values),
                max(values),
            )
        return _Compiled(
            value_type=NUMBER,
            unit=child.unit,
            interval=result_interval,
            instructions=[*child.instructions, Instruction(op)],
            referenced_metrics=set(child.referenced_metrics),
            nodes=child.nodes + 1,
        )
    if op == "not":
        child = _compile(
            _arity(expression, 1)[0],
            schema=schema,
            parameters=parameters,
            parameter_capabilities=parameter_capabilities,
            limits=limits,
        )
        _require_type(child, BOOLEAN, op)
        return _Compiled(
            value_type=BOOLEAN,
            unit="bool",
            interval=None,
            instructions=[*child.instructions, Instruction("not")],
            referenced_metrics=set(child.referenced_metrics),
            nodes=child.nodes + 1,
        )

    if op in {"all", "any"}:
        args = _arity(expression, None)
        if not args:
            raise ValueError(f"Expression op {op!r} requires at least one arg")
        children = [
            _compile(
                arg,
                schema=schema,
                parameters=parameters,
                parameter_capabilities=parameter_capabilities,
                limits=limits,
            )
            for arg in args
        ]
        for child in children:
            _require_type(child, BOOLEAN, op)
        return _combine(
            children,
            BOOLEAN,
            "bool",
            None,
            Instruction(op, len(children)),
        )

    binary_numeric = {"add", "sub", "mul", "min", "max"}
    comparisons = {"lt", "le", "gt", "ge"}
    if op in binary_numeric | comparisons:
        args = _arity(expression, 2)
        children = [
            _compile(
                arg,
                schema=schema,
                parameters=parameters,
                parameter_capabilities=parameter_capabilities,
                limits=limits,
            )
            for arg in args
        ]
        for child in children:
            _require_type(child, NUMBER, op)
        if op == "mul":
            if children[0].unit == "1":
                result_unit = children[1].unit
            elif children[1].unit == "1":
                result_unit = children[0].unit
            else:
                raise ValueError(
                    "Expression op 'mul' requires at least one unitless operand"
                )
        else:
            operand_unit = _require_same_unit(children[0], children[1], op)
            result_unit = "bool" if op in comparisons else operand_unit
        interval = (
            None
            if op in comparisons
            else _bounded_binary(children[0].interval, children[1].interval, op)
        )
        return _combine(
            children,
            BOOLEAN if op in comparisons else NUMBER,
            result_unit,
            interval,
            Instruction(op),
        )
    if op == "eq":
        args = _arity(expression, 2)
        children = [
            _compile(
                arg,
                schema=schema,
                parameters=parameters,
                parameter_capabilities=parameter_capabilities,
                limits=limits,
            )
            for arg in args
        ]
        if children[0].value_type != children[1].value_type:
            raise ValueError("Expression op 'eq' requires equal operand types")
        if children[0].value_type == NUMBER:
            _require_same_unit(children[0], children[1], op)
        return _combine(
            children,
            BOOLEAN,
            "bool",
            None,
            Instruction("eq"),
        )
    if op == "div":
        args = _arity(expression, 2)
        numerator = _compile(
            args[0],
            schema=schema,
            parameters=parameters,
            parameter_capabilities=parameter_capabilities,
            limits=limits,
        )
        denominator = _compile(
            args[1],
            schema=schema,
            parameters=parameters,
            parameter_capabilities=parameter_capabilities,
            limits=limits,
        )
        _require_type(numerator, NUMBER, op)
        _require_type(denominator, NUMBER, op)
        interval = denominator.interval
        if (
            interval is None
            or interval.minimum is None
            or interval.maximum is None
            or interval.minimum != interval.maximum
            or abs(interval.minimum) < limits.minimum_abs_divisor
        ):
            raise ValueError("Division requires a proven non-zero constant divisor")
        divisor = interval.minimum
        if denominator.unit == "1":
            result_unit = numerator.unit
        elif numerator.unit == denominator.unit:
            result_unit = "1"
        else:
            raise ValueError(
                f"Expression op 'div' mixes units {numerator.unit!r} "
                f"and {denominator.unit!r}"
            )
        numerator_interval = numerator.interval or NumericInterval(None, None)
        if numerator_interval.minimum is None or numerator_interval.maximum is None:
            result_interval = NumericInterval(None, None)
        else:
            values = (
                numerator_interval.minimum / divisor,
                numerator_interval.maximum / divisor,
            )
            result_interval = NumericInterval(min(values), max(values))
        return _combine(
            [numerator, denominator],
            NUMBER,
            result_unit,
            result_interval,
            Instruction("div"),
        )
    if op == "clip":
        args = _arity(expression, 3)
        children = [
            _compile(
                arg,
                schema=schema,
                parameters=parameters,
                parameter_capabilities=parameter_capabilities,
                limits=limits,
            )
            for arg in args
        ]
        for child in children:
            _require_type(child, NUMBER, op)
        lower = children[1].interval
        upper = children[2].interval
        result_unit = _require_same_unit(children[0], children[1], op)
        _require_same_unit(children[0], children[2], op)
        if (
            lower is None
            or upper is None
            or lower.minimum is None
            or lower.minimum != lower.maximum
            or upper.minimum is None
            or upper.minimum != upper.maximum
            or lower.minimum > upper.minimum
        ):
            raise ValueError("Clip bounds must be ordered numeric constants")
        return _combine(
            children,
            NUMBER,
            result_unit,
            NumericInterval(lower.minimum, upper.minimum),
            Instruction("clip"),
        )
    raise ValueError(f"Unsupported expression op {op!r}")


def _combine(
    children: list[_Compiled],
    value_type: str,
    unit: str,
    interval: NumericInterval | None,
    instruction: Instruction,
) -> _Compiled:
    instructions = []
    metrics = set()
    nodes = 1
    for child in children:
        instructions.extend(child.instructions)
        metrics.update(child.referenced_metrics)
        nodes += child.nodes
    instructions.append(instruction)
    return _Compiled(value_type, unit, interval, instructions, metrics, nodes)


def _stack_depth(instructions: list[Instruction]) -> int:
    depth = 0
    maximum = 0
    leaf = {"push_const", "load_metric", "load_parameter"}
    unary = {"neg", "abs", "not"}
    binary = {"add", "sub", "mul", "div", "min", "max", "lt", "le", "gt", "ge", "eq"}
    for instruction in instructions:
        if instruction.opcode in leaf:
            depth += 1
        elif instruction.opcode in unary:
            if depth < 1:
                raise ValueError("Expression bytecode underflows the stack")
        elif instruction.opcode in binary:
            if depth < 2:
                raise ValueError("Expression bytecode underflows the stack")
            depth -= 1
        elif instruction.opcode == "clip":
            if depth < 3:
                raise ValueError("Expression bytecode underflows the stack")
            depth -= 2
        elif instruction.opcode in {"all", "any"}:
            count = instruction.operand
            if type(count) is not int or count < 1 or depth < count:
                raise ValueError("Expression bytecode has invalid Boolean arity")
            depth -= count - 1
        else:
            raise ValueError(f"Expression bytecode has unknown opcode {instruction.opcode!r}")
        maximum = max(maximum, depth)
    if depth != 1:
        raise ValueError("Expression bytecode must leave exactly one value")
    return maximum


def compile_expression(
    expression: ExpressionSpec,
    *,
    schema: EventSchema,
    parameters: Mapping[str, Any],
    parameter_capabilities: Mapping[str, ParameterCapability],
    purpose: str,
    task_spec_fingerprint: str,
    limits: CompileLimits | None = None,
) -> ExpressionProgram:
    """Compile an allowlisted AST into deterministic typed stack bytecode."""
    if purpose not in EXPRESSION_PURPOSES:
        raise ValueError(f"Unsupported expression purpose {purpose!r}")
    active_limits = limits or CompileLimits()
    active_limits.validate()
    compiled = _compile(
        expression,
        schema=schema,
        parameters=parameters,
        parameter_capabilities=parameter_capabilities,
        limits=active_limits,
    )
    disallowed_metrics = sorted(
        name
        for name in compiled.referenced_metrics
        if purpose not in schema.by_name[name].allowed_purposes
    )
    if disallowed_metrics:
        raise ValueError(
            f"{purpose} expression cannot use metrics: "
            + ", ".join(disallowed_metrics)
        )
    expected = BOOLEAN if purpose in BOOLEAN_PURPOSES else NUMBER
    if compiled.value_type != expected:
        raise ValueError(f"{purpose} expression must return {expected}")
    if purpose == "reward" and compiled.unit != "1":
        raise ValueError("reward expression must return a unitless number")
    if compiled.nodes > active_limits.max_nodes:
        raise ValueError("Expression exceeds the AST node limit")
    if len(compiled.instructions) > active_limits.max_instructions:
        raise ValueError("Expression exceeds the instruction limit")
    max_stack = _stack_depth(compiled.instructions)
    if max_stack > active_limits.max_stack:
        raise ValueError("Expression exceeds the stack limit")
    return ExpressionProgram(
        bytecode_version=1,
        purpose=purpose,
        result_type=compiled.value_type,
        result_unit=compiled.unit,
        task_spec_fingerprint=task_spec_fingerprint,
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        instructions=tuple(compiled.instructions),
        referenced_metrics=tuple(sorted(compiled.referenced_metrics)),
        max_stack=max_stack,
        result_interval=compiled.interval,
    )


def _validated_metric(metric: MetricSpec, value: Any, path: str) -> Any:
    if metric.value_type == BOOLEAN:
        if type(value) is not bool:
            raise ValueError(f"{path} must be Boolean")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    if metric.minimum is not None and number < metric.minimum:
        raise ValueError(f"{path} is below the EventSchema minimum")
    if metric.maximum is not None and number > metric.maximum:
        raise ValueError(f"{path} exceeds the EventSchema maximum")
    return number


def validate_metric_frame(frame: MetricFrame, schema: EventSchema) -> None:
    """Require complete exact values for every declared temporal slot."""
    if (
        frame.event_schema_id != schema.event_schema_id
        or frame.event_schema_fingerprint != schema.fingerprint
    ):
        raise ValueError("Metric frame EventSchema fingerprint mismatch")
    for at, values in (("previous", frame.previous), ("current", frame.current)):
        expected = {
            metric.name for metric in schema.metrics if at in metric.available_at
        }
        if set(values) != expected:
            raise ValueError(f"Metric frame {at} fields do not match EventSchema")
        for name, value in values.items():
            _validated_metric(schema.by_name[name], value, f"{at}.{name}")


def evaluate_expression(
    program: ExpressionProgram,
    *,
    frame: MetricFrame,
    schema: EventSchema,
) -> bool | float:
    """Evaluate verified bytecode without Python expression execution."""
    if (
        program.bytecode_version != 1
        or program.event_schema_id != schema.event_schema_id
        or program.event_schema_fingerprint != schema.fingerprint
    ):
        raise ValueError("Expression program EventSchema fingerprint mismatch")
    validate_metric_frame(frame, schema)
    _stack_depth(list(program.instructions))
    stack: list[Any] = []
    for instruction in program.instructions:
        op = instruction.opcode
        if op == "push_const":
            stack.append(float(instruction.operand))
        elif op == "load_metric":
            operand = instruction.operand
            values = frame.previous if operand.at == "previous" else frame.current
            stack.append(values[operand.name])
        elif op == "load_parameter":
            stack.append(float(instruction.operand.value))
        elif op in {"neg", "abs", "not"}:
            value = stack.pop()
            stack.append(-value if op == "neg" else abs(value) if op == "abs" else not value)
        elif op in {"all", "any"}:
            count = instruction.operand
            values = stack[-count:]
            del stack[-count:]
            stack.append(all(values) if op == "all" else any(values))
        elif op == "clip":
            upper = stack.pop()
            lower = stack.pop()
            value = stack.pop()
            stack.append(min(max(value, lower), upper))
        else:
            right = stack.pop()
            left = stack.pop()
            operations = {
                "add": lambda: left + right,
                "sub": lambda: left - right,
                "mul": lambda: left * right,
                "div": lambda: left / right,
                "min": lambda: min(left, right),
                "max": lambda: max(left, right),
                "lt": lambda: left < right,
                "le": lambda: left <= right,
                "gt": lambda: left > right,
                "ge": lambda: left >= right,
                "eq": lambda: left == right,
            }
            try:
                stack.append(operations[op]())
            except KeyError as error:
                raise ValueError(f"Unknown expression opcode {op!r}") from error
        if stack and isinstance(stack[-1], float) and not math.isfinite(stack[-1]):
            raise FloatingPointError("Expression produced a non-finite value")
    result = stack[0]
    if program.result_type == BOOLEAN and type(result) is not bool:
        raise ValueError("Expression program returned the wrong type")
    if program.result_type == NUMBER and (
        isinstance(result, bool) or not isinstance(result, (int, float))
    ):
        raise ValueError("Expression program returned the wrong type")
    return result
