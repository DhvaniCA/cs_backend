def table_to_text(table_text: str) -> str:
    """
    Convert raw table text into structured readable format
    """

    lines = table_text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    if not lines:
        return ""

    # Limit size (important)
    lines = lines[:20]

    formatted = "Table Data:\n"

    for line in lines:
        formatted += f"- {line}\n"

    return formatted