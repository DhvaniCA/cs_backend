"""
Enhanced Upload Service for CA Study Materials

Complete ingestion pipeline with:
- Advanced PDF parsing (Docling + pdfplumber + PyMuPDF)
- Table extraction and structuring
- Image extraction and description
- Optimized chunking for longer CA content
- Multi-modal embedding (text, tables, images)
"""

from typing import Dict, Any, List, Optional
import uuid
import os
from pathlib import Path

# Import enhanced processors
# from .enhanced_docling_parser import EnhancedDoclingParser, parse_pdf_enhanced
from .enhanced_chunking import create_chunks_enhanced
from .enhanced_table_processor import process_table_for_embedding, create_table_summary
from .enhanced_image_processor import generate_image_descriptions, process_image_for_embedding
from .metadata_builder import build_metadata
from .embedding_service import embed_texts
from .pinecone_service import upsert_vectors
from .fast_docling_parser import FastDoclingParser as EnhancedDoclingParser, parse_pdf_fast

async def process_pdf_enhanced(
    file_path: str, 
    file_name: str, 
    extra_meta: Dict[str, Any],
    enable_image_descriptions: bool = True,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced PDF processing pipeline
    
    Args:
        file_path: Path to PDF file
        file_name: Original filename
        extra_meta: Metadata from admin (course, subject, chapter, etc.)
        enable_image_descriptions: Whether to generate AI descriptions for images
        openai_api_key: OpenAI API key for image descriptions
    
    Returns:
        dict with processing statistics and results
    """
    
    print(f"📄 Processing PDF: {file_name}")
    
    # ============================================================
    # STEP 1: Parse PDF with Enhanced Parser
    # ============================================================
    print("  ├─ Parsing PDF structure...")
    parser = EnhancedDoclingParser()

    parse_result = parser.parse_pdf_fast(
        file_path,
        # extract_images=True,
        # extract_tables_pdfplumber=True
    )
    
    elements = parse_result["elements"]
    # images = parse_result["images"]
    images = []  # Disable image processing for now to save time and costs
    tables = parse_result["tables"]
    
    print(f"  ├─ Found {len(elements)} elements, {len(images)} images, {len(tables)} tables")
    
    # ============================================================
    # STEP 2: Build Context Map for Image Descriptions
    # ============================================================
    context_map = {}
    
    # Create page-to-text mapping for image context
    for el in elements:
        page = el.get("page")
        if page and el.get("text"):
            if page not in context_map:
                context_map[page] = ""
            context_map[page] += " " + el["text"][:500]  # First 500 chars per page
    
    # ============================================================
    # STEP 3: Process Images (Generate Descriptions)
    # ============================================================
    processed_images = []
    
    if images and enable_image_descriptions and openai_api_key:
        print("  ├─ Generating image descriptions...")
        try:
            from .enhanced_image_processor import generate_image_descriptions
            
            processed_images = await generate_image_descriptions(
                images,
                openai_api_key,
                context_map
            )
            
            print(f"  ├─ Generated descriptions for {len(processed_images)} significant images")
        
        except Exception as e:
            print(f"  ├─ Warning: Image description generation failed: {e}")
            # Fall back to metadata-based descriptions
            for img in images:
                processed = process_image_for_embedding(
                    img,
                    description=None,
                    context=context_map.get(img.get("page"))
                )
                if processed:
                    processed_images.append(processed)
    
    elif images:
        # Use metadata-based descriptions
        print("  ├─ Creating metadata-based image descriptions...")
        for img in images:
            processed = process_image_for_embedding(
                img,
                description=None,
                context=context_map.get(img.get("page"))
            )
            if processed:
                processed_images.append(processed)
    
    # ============================================================
    # STEP 4: Create Enhanced Chunks
    # ============================================================
    print("  ├─ Creating optimized chunks...")
    
    chunks = create_chunks_enhanced(
        elements=elements,
        images=images,
        tables=tables
    )
    
    print(f"  ├─ Created {len(chunks)} text chunks")
    
    # ============================================================
    # STEP 5: Process Tables into Separate Chunks
    # ============================================================
    table_chunks = []
    
    if tables:
        print("  ├─ Processing tables...")
        for table in tables:
            table_text = process_table_for_embedding(table)
            
            if table_text and len(table_text) > 100:  # Only meaningful tables
                table_chunks.append({
                    "text": table_text,
                    "heading": f"Table on Page {table.get('page', 'unknown')}",
                    "page_start": table.get("page"),
                    "page_end": table.get("page"),
                    "length": len(table_text),
                    "type": "table",
                    "table_metadata": {
                        "num_rows": table.get("num_rows", 0),
                        "num_columns": table.get("num_columns", 0),
                        "headers": table.get("headers", [])
                    }
                })
        
        print(f"  ├─ Processed {len(table_chunks)} table chunks")
    
    # ============================================================
    # STEP 6: Combine All Chunks
    # ============================================================
    all_chunks = chunks + table_chunks
    
    # Add image chunks if any
    for img_data in processed_images:
        all_chunks.append({
            "text": img_data["text"],
            "heading": f"Image on Page {img_data['metadata'].get('page', 'unknown')}",
            "page_start": img_data["metadata"].get("page"),
            "page_end": img_data["metadata"].get("page"),
            "length": len(img_data["text"]),
            "type": "image",
            "image_metadata": img_data["metadata"]
        })
    
    print(f"  ├─ Total chunks for embedding: {len(all_chunks)}")
    
    # ============================================================
    # STEP 7: Generate Embeddings
    # ============================================================
    print("  ├─ Generating embeddings...")
    
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = await embed_texts(texts)
    
    print(f"  ├─ Generated {len(embeddings)} embeddings")
    
    # ============================================================
    # STEP 8: Build Vectors with Metadata
    # ============================================================
    print("  ├─ Building vectors...")
    
    vectors = []
    
    for chunk, embedding in zip(all_chunks, embeddings):
        # Build metadata
        metadata = build_metadata(chunk, file_name, extra_meta)
        
        # Add chunk type
        metadata["chunk_type"] = chunk.get("type", "text")
        
        # Add table-specific metadata if present
        if "table_metadata" in chunk:
            metadata["table_rows"] = chunk["table_metadata"].get("num_rows")
            metadata["table_columns"] = chunk["table_metadata"].get("num_columns")
        
        # Add image-specific metadata if present
        if "image_metadata" in chunk:
            img_meta = chunk["image_metadata"]
            metadata["image_type"] = img_meta.get("type")
            metadata["image_width"] = img_meta.get("width")
            metadata["image_height"] = img_meta.get("height")
        
        # Create vector
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata
        })
    
    # ============================================================
    # STEP 9: Upsert to Pinecone
    # ============================================================
    print("  ├─ Uploading to Pinecone...")
    
    upsert_vectors(vectors)
    
    print(f"  └─ ✅ Completed! Indexed {len(vectors)} vectors")
    
    # ============================================================
    # Return Statistics
    # ============================================================
    return {
        "total_vectors": len(vectors),
        "text_chunks": len(chunks),
        "table_chunks": len(table_chunks),
        "image_chunks": len(processed_images),
        "total_images": len(images),
        "total_tables": len(tables),
        "filename": file_name,
        "metadata": extra_meta
    }


async def process_pdf(
    file_path: str, 
    file_name: str, 
    extra_meta: Dict[str, Any]
) -> int:
    """
    Backward compatible wrapper for existing code
    
    Returns:
        Number of vectors created
    """
    # Get OpenAI API key from settings
    from config import settings
    
    result = await process_pdf_enhanced(
        file_path,
        file_name,
        extra_meta,
        enable_image_descriptions=True,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    return result["total_vectors"]


async def process_pdf_batch(
    file_paths: List[str],
    file_names: List[str],
    extra_metas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs in batch
    
    Useful for bulk uploads
    """
    results = []
    
    for file_path, file_name, extra_meta in zip(file_paths, file_names, extra_metas):
        try:
            result = await process_pdf_enhanced(
                file_path,
                file_name,
                extra_meta,
                enable_image_descriptions=True
            )
            results.append(result)
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results.append({
                "filename": file_name,
                "error": str(e),
                "total_vectors": 0
            })
    
    return results