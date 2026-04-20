"""
Enhanced Table Processor for CA Study Materials

Handles:
- Complex accounting tables
- Financial statements
- Tax computation tables
- Comparison tables
"""

from typing import List, Dict, Any, Optional
import re


def clean_cell_value(value: Any) -> str:
    """Clean and normalize cell values"""
    if value is None:
        return ""
    
    value_str = str(value).strip()
    
    # Remove excessive whitespace
    value_str = re.sub(r'\s+', ' ', value_str)
    
    return value_str


def detect_table_type(headers: List[str], rows: List[List[str]]) -> str:
    """
    Detect the type of CA table for appropriate formatting
    """
    header_text = ' '.join([str(h).lower() for h in headers])
    
    # Financial statement indicators
    if any(keyword in header_text for keyword in 
           ['debit', 'credit', 'balance sheet', 'p&l', 'profit', 'loss', 'assets', 'liabilities']):
        return 'financial'
    
    # Tax computation
    if any(keyword in header_text for keyword in 
           ['taxable', 'tax', 'deduction', 'income', 'rate', 'amount']):
        return 'tax'
    
    # Comparison table
    if any(keyword in header_text for keyword in 
           ['before', 'after', 'old', 'new', 'comparison', 'difference']):
        return 'comparison'
    
    # Schedule/Format table
    if any(keyword in header_text for keyword in 
           ['schedule', 'format', 'particulars', 'sr.', 'no.']):
        return 'schedule'
    
    return 'general'


def format_financial_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format financial statements and accounting tables"""
    lines = ["Financial Table:"]
    lines.append("=" * 60)
    
    # Headers
    header_line = " | ".join([str(h).ljust(15) for h in headers[:4]])  # Limit columns
    lines.append(header_line)
    lines.append("-" * 60)
    
    # Rows (limit to important rows)
    for row in rows[:15]:  # First 15 rows
        row_data = [clean_cell_value(cell) for cell in row[:4]]
        row_line = " | ".join([str(cell).ljust(15) for cell in row_data])
        if any(row_data):  # Only include non-empty rows
            lines.append(row_line)
    
    if len(rows) > 15:
        lines.append(f"... ({len(rows) - 15} more rows)")
    
    return "\n".join(lines)


def format_structured_table(
    headers: List[str], 
    rows: List[List[str]], 
    table_type: str = 'general'
) -> str:
    """
    Format structured tables from pdfplumber extraction
    
    This creates a clean, readable text representation
    optimized for embedding and retrieval
    """
    if not headers and not rows:
        return ""
    
    lines = []
    
    # Add table type indicator
    type_labels = {
        'financial': 'Financial Table',
        'tax': 'Tax Computation Table',
        'comparison': 'Comparison Table',
        'schedule': 'Schedule/Format',
        'general': 'Data Table'
    }
    lines.append(f"[{type_labels.get(table_type, 'Table')}]")
    
    # Format headers
    if headers:
        cleaned_headers = [clean_cell_value(h) for h in headers]
        # Create header row
        header_parts = []
        for i, header in enumerate(cleaned_headers[:6]):  # Limit to 6 columns
            if header:
                header_parts.append(f"Column {i+1}: {header}")
        
        if header_parts:
            lines.append("Headers: " + " | ".join(header_parts))
    
    # Format data rows
    lines.append("\nData:")
    
    max_rows = 20  # Limit rows to prevent huge chunks
    for row_idx, row in enumerate(rows[:max_rows]):
        cleaned_row = [clean_cell_value(cell) for cell in row[:6]]
        
        # Skip completely empty rows
        if not any(cleaned_row):
            continue
        
        # Format row
        row_text = " | ".join([cell if cell else "—" for cell in cleaned_row])
        lines.append(f"  Row {row_idx + 1}: {row_text}")
    
    if len(rows) > max_rows:
        lines.append(f"  ... and {len(rows) - max_rows} more rows")
    
    return "\n".join(lines)


def table_to_text(table_text: str) -> str:
    """
    Convert raw Docling table text into structured readable format
    
    This is for backward compatibility with Docling table extraction
    """
    if not table_text:
        return ""
    
    lines = table_text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    
    if not lines:
        return ""
    
    # Limit size (important for chunk size management)
    lines = lines[:25]
    
    formatted = "[Table Data]\n"
    
    # Try to detect header row
    if lines:
        formatted += f"Headers: {lines[0]}\n"
        formatted += "Data:\n"
        
        for idx, line in enumerate(lines[1:], 1):
            formatted += f"  {idx}. {line}\n"
    
    return formatted


def table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """
    Convert table to markdown format
    
    Useful for structured storage or display
    """
    if not headers:
        return ""
    
    lines = []
    
    # Headers
    header_line = "| " + " | ".join([clean_cell_value(h) for h in headers]) + " |"
    lines.append(header_line)
    
    # Separator
    separator = "| " + " | ".join(["---" for _ in headers]) + " |"
    lines.append(separator)
    
    # Data rows
    for row in rows[:30]:  # Limit rows
        row_cells = [clean_cell_value(cell) for cell in row]
        
        # Pad row if it has fewer columns than headers
        while len(row_cells) < len(headers):
            row_cells.append("")
        
        row_line = "| " + " | ".join(row_cells[:len(headers)]) + " |"
        lines.append(row_line)
    
    return "\n".join(lines)


def create_table_summary(table_data: Dict[str, Any]) -> str:
    """
    Create a concise summary of a table for metadata
    
    This helps with retrieval by providing semantic description
    """
    headers = table_data.get("headers", [])
    num_rows = table_data.get("num_rows", 0)
    num_cols = table_data.get("num_columns", 0)
    page = table_data.get("page", "unknown")
    
    header_names = [clean_cell_value(h) for h in headers[:4]]
    header_text = ", ".join([h for h in header_names if h])
    
    summary = f"Table on page {page} with {num_rows} rows and {num_cols} columns"
    if header_text:
        summary += f". Columns: {header_text}"
    
    return summary


def process_table_for_embedding(table_data: Dict[str, Any]) -> str:
    """
    Process a structured table into optimal text for embedding
    
    This creates a format that works well with semantic search
    """
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    page = table_data.get("page", "")
    
    # Detect table type
    table_type = detect_table_type(headers, rows)
    
    # Create summary
    summary = create_table_summary(table_data)
    
    # Format the table
    if table_type == 'financial':
        table_text = format_financial_table(headers, rows)
    else:
        table_text = format_structured_table(headers, rows, table_type)
    
    # Combine summary and content
    result = f"{summary}\n\n{table_text}"
    
    return result