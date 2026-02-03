"""
Image Processing Module with GPT-5 Vision for Description Generation
"""

import base64
import io
import os
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from PIL import Image
from openai import AzureOpenAI
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# GPT-5 Vision credentials from environment
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")


@dataclass
class ImageMetadata:
    """Metadata structure for each image"""
    image_name: str
    image_path: str
    image_unique_id: str
    image_description: str
    image_width: int
    image_height: int
    image_resolution: str
    image_format: str
    image_size_bytes: int
    image_index: int


class ImageProcessor:
    """Process images and generate descriptions using GPT-5 Vision"""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the Image Processor
        
        Args:
            max_workers: Number of parallel workers for processing
        """
        self.max_workers = max_workers
        
        # Initialize GPT-5 Vision client
        self.vision_client = AzureOpenAI(
            api_key=VISION_API_KEY,
            api_version=VISION_API_VERSION,
            azure_endpoint=VISION_ENDPOINT
        )
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract image metadata (resolution, format, size)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing image metadata
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                image_format = img.format
                file_size = os.path.getsize(image_path)
                
                return {
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                    "format": image_format,
                    "size_bytes": file_size,
                    "size_kb": round(file_size / 1024, 2),
                    "size_mb": round(file_size / (1024 * 1024), 2)
                }
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return {
                "width": 0,
                "height": 0,
                "resolution": "Unknown",
                "format": "Unknown",
                "size_bytes": 0,
                "size_kb": 0,
                "size_mb": 0
            }
    
    def generate_image_description(
        self,
        image_path: str,
        custom_prompt: Optional[str] = None,
        max_tokens: int = 2000  # Increased from 500 to allow for reasoning + output
    ) -> str:
        """
        Generate description for an image using GPT-5 Vision
        
        Args:
            image_path: Path to the image file
            custom_prompt: Custom prompt for description generation
            max_tokens: Maximum tokens for the response (default 2000 to allow for reasoning + output)
            
        Returns:
            Generated description text
        """
        try:
            print(f"Generating description for: {image_path}")
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            print(f"Image encoded to base64, length: {len(base64_image)}")
            
            # Default prompt if none provided
            if not custom_prompt:
                custom_prompt = """Analyze this image and provide a detailed description including:
1. Main subject/objects in the image
2. Visual characteristics (colors, composition, style)
3. Context and setting
4. Any text or labels visible
5. Overall purpose or message of the image

Be descriptive and specific."""
            
            print(f"Calling GPT-5 Vision API with model: {VISION_MODEL}")
            
            # Call GPT-5 Vision API
            response = self.vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": custom_prompt
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
                max_completion_tokens=max_tokens
            )
            
            # Extract description from response
            choice = response.choices[0]
            description = choice.message.content
            
            # Check for issues
            if hasattr(choice.message, 'refusal') and choice.message.refusal:
                print(f"⚠️ API refusal: {choice.message.refusal}")
                return f"API refused to generate description: {choice.message.refusal}"
            
            # Validate description
            if not description or not description.strip():
                # Log token usage to diagnose empty responses
                if hasattr(response, 'usage'):
                    print(f"⚠️ Empty description returned. Token usage:")
                    print(f"   Completion tokens: {response.usage.completion_tokens}")
                    print(f"   Reasoning tokens: {getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) if hasattr(response.usage, 'completion_tokens_details') else 'N/A'}")
                    print(f"   Finish reason: {choice.finish_reason}")
                return "Error: Empty description generated by API"
            
            description = description.strip()
            print(f"Description generated successfully ({len(description)} chars)")
            
            return description
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating description for {image_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Full traceback:\n{error_details}")
            return f"Error generating description: {str(e)}"
    
    def process_single_image(
        self,
        image_path: str,
        image_index: int = 0,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single image: extract metadata and generate description
        
        Args:
            image_path: Path to the image file
            image_index: Index of the image
            custom_prompt: Custom prompt for description
            
        Returns:
            Dictionary containing all image data and metadata
        """
        image_name = Path(image_path).name
        image_unique_id = str(uuid.uuid4())
        
        print(f"Processing image: {image_name}")
        
        # Extract metadata
        metadata = self.get_image_metadata(image_path)
        
        # Generate description using GPT-5
        description = self.generate_image_description(image_path, custom_prompt)
        
        # Create ImageMetadata object
        image_data = ImageMetadata(
            image_name=image_name,
            image_path=image_path,
            image_unique_id=image_unique_id,
            image_description=description,
            image_width=metadata["width"],
            image_height=metadata["height"],
            image_resolution=metadata["resolution"],
            image_format=metadata["format"],
            image_size_bytes=metadata["size_bytes"],
            image_index=image_index
        )
        
        return asdict(image_data)
    
    def process_multiple_images(
        self,
        image_paths: List[str],
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image file paths
            custom_prompt: Custom prompt for all images
            
        Returns:
            List of dictionaries containing image data
        """
        results = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {
                executor.submit(
                    self.process_single_image,
                    image_path,
                    idx,
                    custom_prompt
                ): image_path
                for idx, image_path in enumerate(image_paths)
            }
            
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ Completed: {Path(image_path).name}")
                except Exception as e:
                    print(f"✗ Error processing {image_path}: {e}")
        
        # Sort by image_index
        results.sort(key=lambda x: x['image_index'])
        
        return results
    
    def create_documents_from_images(
        self,
        image_data_list: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Create LangChain Document objects from processed image data
        
        Args:
            image_data_list: List of processed image data dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for image_data in image_data_list:
            # Create document with description as content
            doc = Document(
                page_content=image_data["image_description"],
                metadata={
                    "image_name": image_data["image_name"],
                    "image_path": image_data["image_path"],
                    "image_unique_id": image_data["image_unique_id"],
                    "image_description": image_data["image_description"], 
                    "image_width": image_data["image_width"],
                    "image_height": image_data["image_height"],
                    "image_resolution": image_data["image_resolution"],
                    "image_format": image_data["image_format"],
                    "image_size_bytes": image_data["image_size_bytes"],
                    "image_index": image_data["image_index"],
                    "content_type": "image",
                    "source_type": "image_upload"
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_and_create_documents(
        self,
        image_paths: List[str],
        custom_prompt: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Document]]:
        """
        Complete pipeline: process images and create documents
        
        Args:
            image_paths: List of image file paths
            custom_prompt: Custom prompt for description generation
            
        Returns:
            Tuple of (image_data_list, documents)
        """
        # Process images
        image_data_list = self.process_multiple_images(image_paths, custom_prompt)
        
        # Create documents
        documents = self.create_documents_from_images(image_data_list)
        
        return image_data_list, documents


# Utility functions
def save_uploaded_image(uploaded_file, save_dir: str = "uploaded_images") -> str:
    """
    Save uploaded image file to disk
    
    Args:
        uploaded_file: Streamlit uploaded file object
        save_dir: Directory to save the image
        
    Returns:
        Path to saved image file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(uploaded_file.name).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(save_dir, unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def supported_image_formats() -> List[str]:
    """Return list of supported image formats"""
    return ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
