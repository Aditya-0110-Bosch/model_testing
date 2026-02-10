"""
PPT Processing Module with Slide-to-Image Conversion and GPT-5 Vision Integration
"""

import os
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from PIL import Image
from openai import AzureOpenAI
from langchain_core.documents import Document
from spire.presentation import Presentation

# Load environment variables
load_dotenv()

# GPT-5 Vision credentials from environment
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")


@dataclass
class SlideMetadata:
    """Metadata structure for each PPT slide"""
    ppt_name: str
    ppt_path: str
    ppt_unique_id: str
    slide_index: int
    slide_number: int  # 1-based for user display
    slide_image_path: str
    slide_description: str
    slide_image_width: int
    slide_image_height: int
    slide_image_resolution: str
    slide_image_format: str
    slide_image_size_bytes: int


class PPTProcessor:
    """Process PowerPoint presentations by converting slides to images and generating descriptions"""
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize the PPT Processor
        
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
    
    def ppt_to_images(self, ppt_file: str, output_folder: str = "output_images") -> List[str]:
        """
        Convert PowerPoint slides to individual images
        
        Args:
            ppt_file: Path to the PowerPoint file
            output_folder: Directory to save slide images
            
        Returns:
            List of image file paths
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Load the presentation
        presentation = Presentation()
        presentation.LoadFromFile(ppt_file)
        
        image_paths = []
        
        # Iterate through each slide
        for i, slide in enumerate(presentation.Slides):
            # Convert slide to an image
            image = slide.SaveAsImage()
            
            # Generate unique filename
            ppt_basename = Path(ppt_file).stem
            output_path = os.path.join(output_folder, f"{ppt_basename}_slide_{i}.png")
            
            # Save image
            image.Save(output_path)
            image.Dispose()
            
            image_paths.append(output_path)
        
        presentation.Dispose()
        
        return image_paths
    
    def get_slide_description_gpt5(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
        """
        Get slide description using GPT-5 Vision
        
        Args:
            image_path: Path to the slide image
            custom_prompt: Optional custom prompt for description generation
            
        Returns:
            Generated description string
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                import base64
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Default prompt for PPT slides
            default_prompt = "Describe this PowerPoint slide in 2 to 3 lines. Include the main title/heading, key points, and any important visual elements (charts, diagrams, images)."
            
            prompt = custom_prompt if custom_prompt else default_prompt
            
            response = self.vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                # max_tokens=1000,
                # temperature=0.3
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return f"Error: Could not generate description - {str(e)}"
    
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
                }
        except Exception as e:
            print(f"Error extracting metadata for {image_path}: {e}")
            return {
                "width": 0,
                "height": 0,
                "resolution": "unknown",
                "format": "unknown",
                "size_bytes": 0,
            }
    
    def process_single_ppt(self, ppt_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single PowerPoint file
        
        Args:
            ppt_path: Path to the PowerPoint file
            custom_prompt: Optional custom prompt for slide descriptions
            
        Returns:
            Dictionary containing PPT data and slide information
        """
        ppt_name = Path(ppt_path).name
        ppt_unique_id = str(uuid.uuid4())
        
        # Convert PPT to images
        output_folder = os.path.join("output_images", f"{Path(ppt_path).stem}_{ppt_unique_id[:8]}")
        slide_image_paths = self.ppt_to_images(ppt_path, output_folder)
        
        # Process each slide's description IN PARALLEL
        slides_data = [None] * len(slide_image_paths)
        
        def _describe_slide(idx, slide_image_path):
            slide_description = self.get_slide_description_gpt5(slide_image_path, custom_prompt)
            img_metadata = self.get_image_metadata(slide_image_path)
            slide_data = SlideMetadata(
                ppt_name=ppt_name,
                ppt_path=ppt_path,
                ppt_unique_id=ppt_unique_id,
                slide_index=idx,
                slide_number=idx + 1,
                slide_image_path=slide_image_path,
                slide_description=slide_description,
                slide_image_width=img_metadata["width"],
                slide_image_height=img_metadata["height"],
                slide_image_resolution=img_metadata["resolution"],
                slide_image_format=img_metadata["format"],
                slide_image_size_bytes=img_metadata["size_bytes"]
            )
            return idx, asdict(slide_data)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_describe_slide, idx, path): idx
                for idx, path in enumerate(slide_image_paths)
            }
            for future in as_completed(futures):
                try:
                    idx, slide_dict = future.result()
                    slides_data[idx] = slide_dict
                except Exception as e:
                    s_idx = futures[future]
                    print(f"Error processing slide {s_idx + 1}: {e}")
        
        # Remove any None entries from failed slides
        slides_data = [s for s in slides_data if s is not None]
        
        return {
            "ppt_name": ppt_name,
            "ppt_path": ppt_path,
            "ppt_unique_id": ppt_unique_id,
            "total_slides": len(slides_data),
            "slides": slides_data
        }
    
    def process_multiple_ppts(
        self, 
        ppt_paths: List[str], 
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PowerPoint files in parallel
        
        Args:
            ppt_paths: List of PowerPoint file paths
            custom_prompt: Optional custom prompt for slide descriptions
            
        Returns:
            List of dictionaries containing PPT data
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ppt = {
                executor.submit(self.process_single_ppt, ppt_path, custom_prompt): ppt_path 
                for ppt_path in ppt_paths
            }
            
            for future in as_completed(future_to_ppt):
                ppt_path = future_to_ppt[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ Processed: {ppt_path}")
                except Exception as e:
                    print(f"✗ Error processing {ppt_path}: {e}")
        
        return results
    
    def create_documents(self, ppt_data: Dict[str, Any]) -> List[Document]:
        """
        Create LangChain documents from PPT slide data for vector store
        
        Args:
            ppt_data: Dictionary containing PPT and slide information
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for slide in ppt_data["slides"]:
            # Create document with slide description as content
            doc = Document(
                page_content=slide["slide_description"],
                metadata={
                    "source": "ppt",
                    "ppt_name": slide["ppt_name"],
                    "ppt_path": slide["ppt_path"],
                    "ppt_unique_id": slide["ppt_unique_id"],
                    "slide_index": slide["slide_index"],
                    "slide_number": slide["slide_number"],
                    "slide_image_path": slide["slide_image_path"],
                    "slide_image_resolution": slide["slide_image_resolution"],
                    "slide_image_format": slide["slide_image_format"],
                    "slide_image_size_bytes": slide["slide_image_size_bytes"],
                    "content_type": "ppt_slide"
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_and_create_documents(
        self, 
        ppt_paths: List[str],
        custom_prompt: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], List[Document]]:
        """
        Process PPTs and create documents in one go
        
        Args:
            ppt_paths: List of PowerPoint file paths
            custom_prompt: Optional custom prompt for slide descriptions
            
        Returns:
            Tuple of (ppt_data_list, documents_list)
        """
        # Process all PPTs
        ppt_data_list = self.process_multiple_ppts(ppt_paths, custom_prompt)
        
        # Create documents for all PPTs
        all_documents = []
        for ppt_data in ppt_data_list:
            documents = self.create_documents(ppt_data)
            all_documents.extend(documents)
        
        return ppt_data_list, all_documents


def supported_ppt_formats() -> List[str]:
    """Return list of supported PowerPoint file formats"""
    return ['ppt', 'pptx']
