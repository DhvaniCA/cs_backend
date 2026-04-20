"""
Enhanced Image Processor for CA Study Materials

Handles:
- Flowcharts and diagrams
- Charts and graphs
- Formulas and equations (as images)
- Document scans
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import io
import base64
import os


def classify_image_type(width: int, height: int, filename: str) -> str:
    """
    Classify image type based on dimensions and filename
    """
    aspect_ratio = width / height if height > 0 else 1
    
    # Wide images (charts, flowcharts)
    if aspect_ratio > 2.0:
        return "chart_or_flowchart"
    
    # Tall images (organization charts, decision trees)
    if aspect_ratio < 0.5:
        return "vertical_diagram"
    
    # Square or near-square (logos, icons, formulas)
    if 0.8 <= aspect_ratio <= 1.2:
        if width < 200 or height < 200:
            return "icon_or_logo"
        return "diagram_or_formula"
    
    # Wide landscape (graphs, timelines)
    if 1.3 <= aspect_ratio <= 2.0:
        return "graph_or_timeline"
    
    return "general_image"


def is_significant_image(image_metadata: Dict[str, Any]) -> bool:
    """
    Determine if an image is significant enough to process
    
    Filters out:
    - Tiny icons
    - Logos
    - Decorative elements
    """
    width = image_metadata.get("width", 0)
    height = image_metadata.get("height", 0)
    size_bytes = image_metadata.get("size_bytes", 0)
    
    # Minimum dimensions
    if width < 100 or height < 100:
        return False
    
    # Minimum file size (avoid tiny icons)
    if size_bytes < 5000:  # 5KB
        return False
    
    # Maximum size (avoid full-page scans unless they're charts)
    if size_bytes > 5_000_000:  # 5MB
        return False
    
    return True


def create_image_description_prompt(
    image_metadata: Dict[str, Any],
    context: Optional[str] = None
) -> str:
    """
    Create a prompt for LLM to describe the image
    
    This helps generate semantic descriptions for embedding
    """
    image_type = image_metadata.get("type", "general_image")
    page = image_metadata.get("page", "unknown")
    
    base_prompt = f"Describe this image from page {page} of a CA study material. "
    
    type_specific_prompts = {
        "chart_or_flowchart": "This appears to be a flowchart or process diagram. Describe the flow, key steps, and decision points.",
        "graph_or_timeline": "This appears to be a graph or chart. Describe the type of chart, what data it shows, axes, and key insights.",
        "diagram_or_formula": "This appears to be a diagram or formula. Describe the concept illustrated and key components.",
        "vertical_diagram": "This appears to be an organizational chart or decision tree. Describe the hierarchy and relationships.",
    }
    
    specific_prompt = type_specific_prompts.get(image_type, 
        "Describe what this image shows and its relevance to CA studies.")
    
    full_prompt = base_prompt + specific_prompt
    
    if context:
        full_prompt += f"\n\nContext: {context[:200]}"
    
    return full_prompt


def encode_image_for_api(image_path: str) -> Optional[str]:
    """
    Encode image as base64 for API calls
    """
    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def create_image_metadata_text(image_metadata: Dict[str, Any]) -> str:
    """
    Create searchable text from image metadata
    
    This is used when we can't get LLM descriptions
    """
    page = image_metadata.get("page", "unknown")
    width = image_metadata.get("width", 0)
    height = image_metadata.get("height", 0)
    image_type = image_metadata.get("type", "general_image")
    
    type_labels = {
        "chart_or_flowchart": "flowchart or process diagram",
        "graph_or_timeline": "chart or graph",
        "diagram_or_formula": "diagram or formula",
        "vertical_diagram": "organizational chart or decision tree",
        "general_image": "figure or illustration"
    }
    
    type_label = type_labels.get(image_type, "image")
    
    text = f"[Image: {type_label} on page {page}, dimensions {width}x{height}px]"
    
    return text


def process_image_for_embedding(
    image_metadata: Dict[str, Any],
    description: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process image into a format suitable for vector embedding
    
    Returns:
        dict with text representation and metadata
    """
    # Check if image is significant
    if not is_significant_image(image_metadata):
        return None
    
    # Classify image type
    width = image_metadata.get("width", 0)
    height = image_metadata.get("height", 0)
    filename = image_metadata.get("filename", "")
    
    image_type = classify_image_type(width, height, filename)
    image_metadata["type"] = image_type
    
    # Create searchable text
    if description:
        # Use LLM-generated description if available
        text = f"[Image on page {image_metadata.get('page', 'unknown')}]\n{description}"
    else:
        # Fall back to metadata-based text
        text = create_image_metadata_text(image_metadata)
    
    # Add context if available
    if context:
        text += f"\n\nContext: {context[:300]}"
    
    return {
        "text": text,
        "metadata": image_metadata,
        "type": "image"
    }


async def generate_image_descriptions(
    images: List[Dict[str, Any]],
    openai_api_key: str,
    context_map: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate descriptions for images using OpenAI Vision API
    
    Args:
        images: List of image metadata
        openai_api_key: OpenAI API key
        context_map: Optional dict mapping page numbers to surrounding text context
    
    Returns:
        List of processed images with descriptions
    """
    import httpx
    
    processed_images = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for image in images:
            if not is_significant_image(image):
                continue
            
            image_path = image.get("path")
            page = image.get("page")
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            # Get context for this page
            context = context_map.get(page) if context_map else None
            
            # Encode image
            base64_image = encode_image_for_api(image_path)
            if not base64_image:
                continue
            
            # Classify image type
            width = image.get("width", 0)
            height = image.get("height", 0)
            filename = image.get("filename", "")
            image_type = classify_image_type(width, height, filename)
            
            # Create description prompt
            prompt = create_image_description_prompt(image, context)
            
            # Call OpenAI Vision API
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",  # Using mini for cost efficiency
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 300
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    description = result["choices"][0]["message"]["content"]
                    
                    # Process image with description
                    processed = process_image_for_embedding(
                        image,
                        description=description,
                        context=context
                    )
                    
                    if processed:
                        processed_images.append(processed)
                
                else:
                    # Fall back to metadata-based description
                    processed = process_image_for_embedding(image, context=context)
                    if processed:
                        processed_images.append(processed)
            
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                # Fall back to metadata-based description
                processed = process_image_for_embedding(image, context=context)
                if processed:
                    processed_images.append(processed)
    
    return processed_images