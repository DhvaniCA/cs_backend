"""
Enhanced Docling Parser - FIXED VERSION with Detailed Comments

This module combines multiple PDF parsing libraries to extract:
1. Text and structure (via Docling)
2. Tables (via pdfplumber - better than Docling alone)
3. Images (via PyMuPDF/fitz)

Author: Enhanced for CA Education Platform
Version: 2.0 (Fixed for latest Docling API)
"""

# ============================================================
# IMPORTS - These are all the libraries we need
# ============================================================

from docling.document_converter import DocumentConverter
import pdfplumber
import fitz  # This is PyMuPDF - confusing name but it's the package name
from PIL import Image
import io
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

print("✅ Enhanced Docling Parser (Fixed Version) loaded successfully")


class EnhancedDoclingParser:
    """
    This class handles all PDF parsing operations.
    
    Think of it as your PDF expert that knows how to:
    - Read text from PDFs
    - Extract tables properly
    - Find and save images
    
    WHY DO WE NEED THIS?
    - Regular PDF readers lose table structure
    - Images get ignored completely
    - Context gets fragmented
    """
    
    def __init__(self):
        """
        Initialize the parser.
        
        This is like setting up your workspace before starting work.
        We create a DocumentConverter from Docling that will handle
        the heavy lifting of understanding PDF structure.
        """
        self.converter = DocumentConverter()
        print("  🔧 PDF Parser initialized and ready")
    
    def extract_images_with_metadata(
        self, 
        file_path: str,
        output_dir: str = "/tmp/extracted_images"
    ) -> List[Dict[str, Any]]:
        """
        Extract images from PDF with detailed information.
        
        WHAT THIS DOES:
        1. Opens the PDF
        2. Goes through each page
        3. Finds all images
        4. Saves significant images (filters out tiny logos)
        5. Records where each image is located
        
        WHY WE NEED THIS:
        - CA books have important flowcharts and diagrams
        - These visuals help students understand concepts
        - We can later describe these images using AI
        
        PARAMETERS:
        - file_path: Where the PDF file is located
        - output_dir: Where to save extracted images
        
        RETURNS:
        - List of dictionaries, each containing image info
        """
        
        # Create folder to store images if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # This list will store information about all images we find
        images = []
        
        try:
            # Open the PDF file using PyMuPDF
            # Think of this like opening a book to read it
            doc = fitz.open(file_path)
            
            # Go through each page one by one
            # Page numbers in computers start at 0, so page 0 = page 1 in the book
            for page_num in range(len(doc)):
                # Get the current page
                page = doc[page_num]
                
                # Ask: "What images are on this page?"
                # full=True means we want complete image information
                image_list = page.get_images(full=True)
                
                # Now process each image we found
                for img_index, img in enumerate(image_list):
                    # xref is like a reference number for the image
                    xref = img[0]
                    
                    # Extract the actual image data (the picture itself)
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]  # The actual image data
                        image_ext = base_image["ext"]      # File type (jpg, png, etc.)
                    except Exception as e:
                        # Some images can't be extracted (they're embedded weird)
                        # Just skip them and move on
                        print(f"    ⚠️  Skipping problematic image on page {page_num + 1}")
                        continue
                    
                    # Get image size (width and height in pixels)
                    pix = fitz.Pixmap(doc, xref)
                    width = pix.width
                    height = pix.height
                    
                    # IMPORTANT FILTER: Skip tiny images
                    # Why? Because tiny images are usually logos, icons, or decorations
                    # We only want meaningful images like charts and diagrams
                    if width < 50 or height < 50:
                        continue  # Skip this image and go to the next one
                    
                    # Create a unique filename for this image
                    # Format: page_1_img_0.jpg (page number, image number)
                    image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    # Save the image to disk
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Get the position of the image on the page
                    # This tells us where exactly the image appears
                    rect_list = page.get_image_rects(xref)
                    position = rect_list[0] if rect_list else None
                    
                    # Store all the information about this image
                    images.append({
                        "page": page_num + 1,           # Which page (human-readable)
                        "index": img_index,             # Which image on that page
                        "path": image_path,             # Where we saved it
                        "filename": image_filename,     # The filename we gave it
                        "width": width,                 # How wide it is
                        "height": height,               # How tall it is
                        "format": image_ext,            # jpg, png, etc.
                        "position": {                   # Where on the page
                            "x0": position.x0 if position else 0,
                            "y0": position.y0 if position else 0,
                            "x1": position.x1 if position else 0,
                            "y1": position.y1 if position else 0,
                        } if position else None,
                        "size_bytes": len(image_bytes) # File size
                    })
            
            # Close the PDF file (clean up)
            doc.close()
            
            # Report what we found
            print(f"  📸 Extracted {len(images)} significant images")
            
        except Exception as e:
            # If something goes wrong, print the error but don't crash
            print(f"  ❌ Error extracting images: {e}")
        
        # Return the list of images we found
        return images
    
    def extract_tables_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber - much better than basic text extraction.
        
        WHAT THIS DOES:
        1. Opens PDF with pdfplumber
        2. Looks for tables on each page
        3. Extracts table structure (rows and columns)
        4. Cleans up the data
        
        WHY THIS IS IMPORTANT FOR CA:
        - Financial statements are tables
        - Tax computations are tables
        - Comparison charts are tables
        - Without this, students get messy unstructured text
        
        RETURNS:
        - List of tables with structure preserved
        """
        
        # This will store all the tables we find
        tables = []
        
        try:
            # Open the PDF using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                
                # Go through each page
                for page_num, page in enumerate(pdf.pages):
                    
                    # Extract tables with OPTIMIZED settings for CA documents
                    # These settings tell pdfplumber how to find table boundaries
                    page_tables = page.extract_tables(table_settings={
                        "vertical_strategy": "lines_strict",    # Look for vertical lines
                        "horizontal_strategy": "lines_strict",  # Look for horizontal lines
                        "intersection_x_tolerance": 3,          # How close lines need to be
                        "intersection_y_tolerance": 3,
                        "min_words_vertical": 3,                # Minimum to consider a column
                        "min_words_horizontal": 1,
                    })
                    
                    # Process each table we found on this page
                    for table_index, table in enumerate(page_tables):
                        
                        # Skip empty or too-small tables
                        if not table or len(table) < 2:
                            continue  # Need at least header + 1 data row
                        
                        # Clean up the table data
                        # Sometimes cells have extra spaces or are None
                        cleaned_table = []
                        for row in table:
                            # Clean each cell in the row
                            cleaned_row = [
                                (cell.strip() if cell else "") 
                                for cell in row
                            ]
                            # Only keep rows that have some content
                            if any(cleaned_row):  # If at least one cell has content
                                cleaned_table.append(cleaned_row)
                        
                        # Make sure we still have enough rows after cleaning
                        if len(cleaned_table) >= 2:
                            
                            # Store the table with all its information
                            tables.append({
                                "page": page_num + 1,              # Which page
                                "table_index": table_index,        # Which table on that page
                                "headers": cleaned_table[0],       # First row = headers
                                "rows": cleaned_table[1:],         # Rest = data
                                "num_columns": len(cleaned_table[0]),  # How many columns
                                "num_rows": len(cleaned_table) - 1,    # How many data rows
                                "raw_data": cleaned_table          # The complete table
                            })
            
            # Report what we found
            print(f"  📊 Extracted {len(tables)} structured tables")
        
        except Exception as e:
            print(f"  ❌ Error extracting tables: {e}")
        
        return tables
    
    def parse_pdf_enhanced(
        self, 
        file_path: str,
        extract_images: bool = True,
        extract_tables_pdfplumber: bool = True
    ) -> Dict[str, Any]:
        """
        MAIN PARSING FUNCTION - This is where everything comes together!
        
        WHAT THIS DOES (Step by step):
        1. Use Docling to understand the document structure
        2. Extract all text elements (headings, paragraphs, etc.)
        3. Extract images using PyMuPDF
        4. Extract tables using pdfplumber
        5. Package everything nicely
        
        WHY THIS APPROACH?
        - Docling: Good at understanding document structure
        - pdfplumber: Best for tables
        - PyMuPDF: Best for images
        - Together they give us complete coverage!
        
        PARAMETERS:
        - file_path: Path to the PDF file
        - extract_images: Should we extract images? (True/False)
        - extract_tables_pdfplumber: Should we extract tables? (True/False)
        
        RETURNS:
        - Dictionary containing:
            * elements: Document structure (headings, paragraphs)
            * images: Extracted images with metadata
            * tables: Structured tables
            * metadata: Statistics and info
        """
        
        # ========================================
        # STEP 1: Parse document structure
        # ========================================
        print("  ├─ Using Docling to parse structure...")
        
        # Convert PDF using Docling
        # This gives us the document structure
        result = self.converter.convert(file_path)
        
        # Initialize our elements list
        # Elements are pieces of the document (headings, paragraphs, tables, etc.)
        elements = []
        
        # This will track which page each element is on
        page_mapping = {}
        
        # IMPORTANT: Handle the NEW Docling API
        # Newer versions of Docling changed how we access elements
        # We need to iterate through the document differently
        
        try:
            # Try the NEW API first (Docling 1.0+)
            # The document is in result.document
            # We iterate through it to get elements
            
            # For newer Docling versions, we need to iterate differently
            # The document object is iterable but doesn't have .elements attribute
            
            element_index = 0
            
            # Method 1: Try to iterate through the document
            if hasattr(result.document, '__iter__'):
                # The document is iterable
                for item in result.document:
                    # Each item has text and metadata
                    element_data = {
                        "type": getattr(item, 'type', 'paragraph'),  # Default to paragraph
                        "text": getattr(item, 'text', str(item))      # Get text
                    }
                    
                    # Try to get page number if available
                    if hasattr(item, 'page'):
                        element_data["page"] = item.page
                        page_mapping[element_index] = item.page
                    
                    elements.append(element_data)
                    element_index += 1
            
            # Method 2: Use export_to_dict() if available (common in newer versions)
            elif hasattr(result.document, 'export_to_dict'):
                doc_dict = result.document.export_to_dict()
                
                # Extract elements from the dictionary
                if 'texts' in doc_dict:
                    for idx, text_item in enumerate(doc_dict['texts']):
                        elements.append({
                            "type": text_item.get('type', 'paragraph'),
                            "text": text_item.get('text', ''),
                            "page": text_item.get('page')
                        })
            
            # Method 3: Try markdown export (fallback)
            elif hasattr(result.document, 'export_to_markdown'):
                # Get markdown text
                markdown_text = result.document.export_to_markdown()
                
                # Split by double newlines to get paragraphs
                paragraphs = markdown_text.split('\n\n')
                
                for para in paragraphs:
                    if para.strip():
                        # Detect if it's a heading (starts with #)
                        if para.startswith('#'):
                            element_type = 'heading'
                        else:
                            element_type = 'paragraph'
                        
                        elements.append({
                            "type": element_type,
                            "text": para.strip()
                        })
            
            # If we got no elements, create a basic one from the result
            if not elements:
                print("  ⚠️  Using fallback text extraction")
                # Last resort: convert to text
                if hasattr(result.document, 'export_to_text'):
                    full_text = result.document.export_to_text()
                else:
                    full_text = str(result.document)
                
                # Split into paragraphs
                paragraphs = full_text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        elements.append({
                            "type": "paragraph",
                            "text": para.strip()
                        })
        
        except Exception as e:
            print(f"  ⚠️  Docling parsing issue: {e}")
            print("  ⚠️  Using basic text extraction as fallback")
            
            # Ultimate fallback: just get the text somehow
            try:
                if hasattr(result, 'text'):
                    text = result.text
                elif hasattr(result.document, 'text'):
                    text = result.document.text
                else:
                    text = str(result.document)
                
                # Split into paragraphs
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        elements.append({
                            "type": "paragraph",
                            "text": para.strip()
                        })
            except:
                # If everything fails, return empty elements
                # The table and image extraction will still work
                print("  ❌ Could not extract text from Docling")
                elements = []
        
        print(f"  ├─ Extracted {len(elements)} text elements")
        
        # ========================================
        # STEP 2: Prepare the result dictionary
        # ========================================
        
        result_dict = {
            "elements": elements,
            "images": [],
            "tables": [],
            "metadata": {
                "total_elements": len(elements),
                "file_path": file_path,
                "filename": Path(file_path).name
            }
        }
        
        # ========================================
        # STEP 3: Extract images (if enabled)
        # ========================================
        
        if extract_images:
            print("  ├─ Extracting images...")
            images = self.extract_images_with_metadata(file_path)
            result_dict["images"] = images
            result_dict["metadata"]["total_images"] = len(images)
        
        # ========================================
        # STEP 4: Extract tables (if enabled)
        # ========================================
        
        if extract_tables_pdfplumber:
            print("  ├─ Extracting tables...")
            tables = self.extract_tables_with_pdfplumber(file_path)
            result_dict["tables"] = tables
            result_dict["metadata"]["total_tables"] = len(tables)
        
        # ========================================
        # DONE! Return everything
        # ========================================
        
        print(f"  └─ ✅ Parsing complete!")
        return result_dict


# ============================================================
# STANDALONE FUNCTIONS (for backward compatibility)
# ============================================================

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Simple function for basic parsing (backward compatible).
    
    This is a simplified version that just returns elements.
    Use this if you only need text, not images or tables.
    
    WHEN TO USE THIS:
    - Upgrading from old code
    - Only need text extraction
    - Don't need images or tables
    """
    parser = EnhancedDoclingParser()
    result = parser.parse_pdf_enhanced(
        file_path,
        extract_images=False,  # Don't extract images for speed
        extract_tables_pdfplumber=False  # Don't extract tables for speed
    )
    
    return result["elements"]


def parse_pdf_advanced(file_path: str) -> Dict[str, Any]:
    """
    Advanced parsing with everything enabled.
    
    Use this to get the full power of the enhanced parser:
    - Text elements
    - Structured tables
    - Images with metadata
    
    WHEN TO USE THIS:
    - Processing CA study materials
    - Need tables and images
    - Want the best quality
    """
    parser = EnhancedDoclingParser()
    return parser.parse_pdf_enhanced(file_path)


# ============================================================
# SIMPLE TESTING CODE
# ============================================================

if __name__ == "__main__":
    """
    This code runs only if you execute this file directly.
    It's for testing purposes.
    
    To test: python enhanced_docling_parser_fixed.py
    """
    print("\n" + "="*60)
    print("TESTING ENHANCED DOCLING PARSER")
    print("="*60 + "\n")
    
    # You would put a test PDF path here
    test_pdf = "test.pdf"
    
    if os.path.exists(test_pdf):
        parser = EnhancedDoclingParser()
        result = parser.parse_pdf_enhanced(test_pdf)
        
        print(f"\n📊 Results:")
        print(f"  - Text elements: {len(result['elements'])}")
        print(f"  - Tables: {len(result['tables'])}")
        print(f"  - Images: {len(result['images'])}")
    else:
        print("⚠️  No test PDF found. Skipping test.")