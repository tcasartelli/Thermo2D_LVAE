"""Utility functions for parsing arguments."""

import ast


def parse_list_from_string(value: str, field_name: str) -> list:
    """Parse a string representation of a list into an actual list.

    Args:
        value: The string to parse.
        field_name: The name of the field being parsed (for error messages).

    Returns:
        list: The parsed list.

    Raises:
        TypeError: If the parsed value is not a list.
        ValueError: If the string cannot be parsed.
    """
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list):
            return parsed_value
        raise TypeError(f"Expected list for {field_name}")  # noqa: TRY301
    except Exception as e:
        raise ValueError(f"Invalid format for {field_name}") from e


def parse_list_from_single_item_list(value_list: list, field_name: str) -> list:
    """Parse a list containing a single string item that might be a string representation of a list.

    Args:
        value_list: A list containing a single string item.
        field_name: The name of the field being parsed (for error messages).

    Returns:
        list: The parsed list or the original list if parsing is not needed.
    """
    if not value_list or not isinstance(value_list[0], str):
        return value_list

    first = value_list[0].strip()
    if first and first[0] in ("[", "{"):
        try:
            parsed_value = ast.literal_eval(first)
            if isinstance(parsed_value, list):
                return parsed_value
        except Exception as e:
            raise ValueError(f"Invalid format for {field_name}") from e

    return value_list
