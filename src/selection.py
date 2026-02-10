import random
import re
from typing import TYPE_CHECKING, Any, Dict, List, TypeVar, Union

from src import dotty, log

if TYPE_CHECKING:
    from chat import Chat

logger = log.setup_logger(__name__)
T = TypeVar("T")


class BooleanExpressionNode:
    pass


class TerminalNode(BooleanExpressionNode):
    def __init__(self, term: str):
        self.term = term.strip()


class AndNode(BooleanExpressionNode):
    def __init__(self, left: BooleanExpressionNode, right: BooleanExpressionNode):
        self.left = left
        self.right = right


class OrNode(BooleanExpressionNode):
    def __init__(self, left: BooleanExpressionNode, right: BooleanExpressionNode):
        self.left = left
        self.right = right


def select_percentage(chat: "Chat", **kwargs):
    percentage = kwargs.get("percentage", 50)
    nodes = chat.plain()
    num_nodes = max(1, int(len(nodes) * (percentage / 100)))

    return nodes[:num_nodes]


def select_match(chat: "Chat", **kwargs):
    substring = kwargs.get("substring", "")
    role = kwargs.get("role", "")
    index = kwargs.get("index", None)

    nodes = chat.plain()

    if role:
        nodes = [node for node in nodes if node.role == role]

    if substring:
        nodes = [node for node in nodes if substring in node.content]

    if index is not None:
        nodes = [nodes[index]]

    return nodes


def select_from_user(chat: "Chat"):
    return select_match(chat, role="user")


def select_all(chat: "Chat"):
    return chat.plain()


def select_first(chat: "Chat"):
    return select_match(chat, index=0)


def select_last(chat: "Chat"):
    return select_match(chat, index=-1)


def select_any(chat: "Chat"):
    return [random.choice(chat.plain())]


selection_strategies = {
    "all": select_all,
    "first": select_first,
    "last": select_last,
    "any": select_any,
    "percentage": select_percentage,
    "match": select_match,
    "user": select_from_user,
}


def apply_strategy(chat: "Chat", strategy: str, params: dict):
    return selection_strategies[strategy](chat, **params)  # type: ignore


def match_regex(value, regex):
    return bool(re.match(regex, value))


def match_substring(value, substring):
    return substring in value


def match_exact(value, target):
    return value == target


def filter_array_elements(items: List[T], filter_obj: Dict[str, Any]) -> List[T]:
    """
    Filter elements of an array based on a filter object.

    Args:
        array: The array to filter
        filter_obj: A Hasura-style filter object

    Returns:
        list: Filtered array
    """
    return [item for item in items if match_filter(item, filter_obj)]


def find_array_element(items: List[T], filter_obj: Dict[str, Any]) -> Union[T, None]:
    """
    Find the first element in an array that matches a filter object.

    Args:
        array: The array to search
        filter_obj: A Hasura-style filter object
    Returns:
        Any: The first matching element, or None if no match is found
    """

    for item in items:
        if match_filter(item, filter_obj):
            return item
    return None


def match_filter(context: Any, filter_obj: Dict[str, Any]):
    """
    Match a context object (dict, class instance, etc.) against a Hasura-style filter.

    Args:
        context: The object to match against the filter
        filter_obj: A Hasura-style filter object

    Returns:
        bool: True if the context matches the filter, False otherwise
    """
    if filter_obj is None:
        return True

    # Handle empty filter
    if not filter_obj:
        return True

    # Convert class instance to dict if needed
    if not isinstance(context, dict):
        context = {
            k: getattr(context, k)
            for k in dir(context)
            if not k.startswith("_") and not callable(getattr(context, k))
        }

    # Process logical operators
    if "_and" in filter_obj:
        return all(match_filter(context, sub_filter) for sub_filter in filter_obj["_and"])

    if "_or" in filter_obj:
        return any(match_filter(context, sub_filter) for sub_filter in filter_obj["_or"])

    if "_not" in filter_obj:
        return not match_filter(context, filter_obj["_not"])

    # Process field conditions
    for field, conditions in filter_obj.items():
        if field in ("_and", "_or", "_not"):
            continue

        # Get field value, treating missing fields as None
        if dotty.has(context, field):
            current = dotty.get(context, field)
        else:
            current = None

        # Handle both direct operators and nested conditions
        if isinstance(conditions, dict):
            if any(k.startswith("_") for k in conditions.keys()):
                # Direct operators
                for operator, expected_value in conditions.items():
                    # For _is_null, we allow missing fields to be treated as None
                    if operator == "_is_null":
                        if not apply_operator(current, operator, expected_value):
                            return False
                    else:
                        # For other operators, missing fields should fail the match
                        if current is None and not dotty.has(context, field):
                            return False
                        if not apply_operator(current, operator, expected_value):
                            return False
            else:
                # Nested conditions
                if not match_filter(current, conditions):
                    return False
        else:
            # Direct value comparison
            if not match_exact(current, conditions):
                return False

    return True


def apply_operator(field_value, operator, expected_value):
    """Apply a Hasura operator to compare field_value with expected_value."""

    # Equal operators
    if operator == "_eq":
        return field_value == expected_value
    if operator == "_neq":
        return field_value != expected_value
    if operator == "_is_null":
        return (field_value is None) == expected_value

    # Comparison operators
    if operator == "_gt":
        return field_value > expected_value
    if operator == "_lt":
        return field_value < expected_value
    if operator == "_gte":
        return field_value >= expected_value
    if operator == "_lte":
        return field_value <= expected_value
    if operator == "_neq":
        return field_value != expected_value

    # Array operators
    if operator == "_in":
        return field_value in expected_value
    if operator == "_nin":
        return field_value not in expected_value
    if operator == "_some":
        if isinstance(field_value, list):
            return any(match_filter(item, expected_value) for item in field_value)
        return False
    if operator == "_every":
        if isinstance(field_value, list):
            return all(match_filter(item, expected_value) for item in field_value)
        return False
    if operator == "_none":
        if isinstance(field_value, list):
            return not any(match_filter(item, expected_value) for item in field_value)
        return True
    if operator == "_is_empty":
        if isinstance(field_value, list):
            return (len(field_value) == 0) == expected_value
        elif isinstance(field_value, dict):
            return (len(field_value) == 0) == expected_value
        return field_value == []

    # String operators
    if operator in ("_like", "_ilike", "_nlike", "_nilike"):
        pattern_value = str(expected_value).replace("%", "\x00")
        escaped_value = re.escape(pattern_value)
        pattern = escaped_value.replace("\x00", ".*")
        is_case_insensitive = "ilike" in operator
        is_negated = operator.startswith("_n")

        try:
            flags = re.DOTALL
            if is_case_insensitive:
                flags |= re.IGNORECASE

            result = bool(re.match(f"^{pattern}$", str(field_value), flags))

            return not result if is_negated else result
        except re.error:
            return False

    # Regex operators
    if operator in ("_regex", "_iregex", "_nregex", "_niregex"):
        is_case_insensitive = "iregex" in operator
        is_negated = operator.startswith("_n")

        try:
            if is_case_insensitive:
                result = bool(re.search(str(expected_value), str(field_value), re.IGNORECASE))
            else:
                result = bool(re.search(str(expected_value), str(field_value)))

            return not result if is_negated else result
        except re.error:
            return False

    # JSON operators
    if operator == "_contains":
        if isinstance(field_value, dict) and isinstance(expected_value, dict):
            # Check if all key-value pairs in expected_value exist in field_value
            return all(k in field_value and field_value[k] == v for k, v in expected_value.items())
        elif isinstance(field_value, list):
            # Check if expected_value is in the list
            return expected_value in field_value
        elif isinstance(expected_value, list) and not isinstance(field_value, (dict, list)):
            # If expected_value is a list but field_value is a scalar
            return field_value in expected_value
        else:
            # Direct comparison for other types
            return field_value == expected_value

    raise ValueError(f"Unsupported operator: {operator}")


def _tokenize_boolean_expression(text: str) -> List[str]:
    import re

    pattern = r"(\(|\)|AND|OR)"
    parts = re.split(pattern, text)

    tokens = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            if stripped in ("(", ")", "AND", "OR"):
                tokens.append(stripped)
            else:
                normalized = " ".join(stripped.split())
                if normalized:
                    tokens.append(normalized)

    open_count = tokens.count("(")
    close_count = tokens.count(")")
    if open_count > close_count:
        raise ValueError("Unbalanced parentheses: missing closing parenthesis")
    elif close_count > open_count:
        raise ValueError("Unbalanced parentheses: too many closing parentheses")

    return tokens


def _parse_or_expression(tokens: List[str], pos: int) -> tuple[BooleanExpressionNode, int]:
    left, pos = _parse_and_expression(tokens, pos)

    while pos < len(tokens) and tokens[pos] == "OR":
        pos += 1
        right, pos = _parse_and_expression(tokens, pos)
        left = OrNode(left, right)

    return left, pos


def _parse_and_expression(tokens: List[str], pos: int) -> tuple[BooleanExpressionNode, int]:
    left, pos = _parse_primary_expression(tokens, pos)

    while pos < len(tokens) and tokens[pos] == "AND":
        pos += 1
        right, pos = _parse_primary_expression(tokens, pos)
        left = AndNode(left, right)

    return left, pos


def _parse_primary_expression(tokens: List[str], pos: int) -> tuple[BooleanExpressionNode, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    if token == "(":
        pos += 1
        expr, pos = _parse_or_expression(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("Expected closing parenthesis")
        return expr, pos + 1

    if token in ("AND", "OR", ")"):
        raise ValueError(f"Unexpected token: {token}")

    return TerminalNode(token), pos + 1


def parse_boolean_expression(text: str) -> Union[str, BooleanExpressionNode]:
    if not text or not text.strip():
        raise ValueError("Expression cannot be empty")

    text = text.strip()

    if "AND" not in text and "OR" not in text:
        return text

    tokens = _tokenize_boolean_expression(text)

    if not tokens:
        raise ValueError("Expression cannot be empty")

    expr, pos = _parse_or_expression(tokens, 0)

    if pos < len(tokens):
        raise ValueError(f"Unexpected token after expression: {tokens[pos]}")

    return expr


def build_search_filter(expression: str, field_builder_fn) -> Dict[str, Any]:
    parsed = parse_boolean_expression(expression)

    if isinstance(parsed, str):
        return field_builder_fn(parsed)

    def convert_ast(node: BooleanExpressionNode) -> Dict[str, Any]:
        if isinstance(node, TerminalNode):
            return field_builder_fn(node.term)
        elif isinstance(node, AndNode):
            return {"_and": [convert_ast(node.left), convert_ast(node.right)]}
        elif isinstance(node, OrNode):
            return {"_or": [convert_ast(node.left), convert_ast(node.right)]}
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    return convert_ast(parsed)
