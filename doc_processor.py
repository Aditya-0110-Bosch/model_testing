"""
DOC/DOCX Document Processor for PDF Retrieval Application

This module handles processing of .doc and .docx files by:
1. Converting documents to PDF using docx2pdf
2. Converting PDF pages to images using pdf2image
3. Generating descriptions using GPT-5 Vision
4. Creating LangChain documents with metadata
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from docx2pdf import convert
from pdf2image import convert_from_path
from langchain_core.documents import Document
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# GPT-5 Vision credentials from environment
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")


def supported_doc_formats() -> List[str]:
    """Return list of supported document formats"""
    return ['doc', 'docx']


class DocProcessor:
    """Process DOC/DOCX documents by converting to images and generating descriptions"""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize DOC processor
        
        Args:
            max_workers: Maximum number of parallel workers for processing
        """
        self.max_workers = max_workers
        
        # Initialize GPT-5 Vision client
        self.vision_client = AzureOpenAI(
            api_key=VISION_API_KEY,
            api_version=VISION_API_VERSION,
            azure_endpoint=VISION_ENDPOINT
        )
    
    def convert_doc_to_pdf(self, doc_path: str, output_dir: str = None) -> str:
        """
        Convert DOC/DOCX to PDF using docx2pdf
        
        Args:
            doc_path: Path to the .doc or .docx file
            output_dir: Directory to save the PDF (optional)
        
        Returns:
            Path to the generated PDF file
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output PDF path
        doc_name = Path(doc_path).stem
        pdf_path = os.path.join(output_dir, f"{doc_name}.pdf")
        
        # Convert doc to pdf
        convert(doc_path, pdf_path)
        
        return pdf_path
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str = None, dpi: int = 200) -> List[str]:
        """
        Convert PDF pages to images using pdf2image
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images (optional)
            dpi: DPI for image conversion (default 200)
        
        Returns:
            List of paths to generated images
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Save images and collect paths
        image_paths = []
        pdf_name = Path(pdf_path).stem
        
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        return image_paths
    
    def get_image_description(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
        """
        Generate description for an image using GPT-5 Vision
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom prompt for description generation
        
        Returns:
            Generated description text
        """
        import base64
        
        # Default prompt for document pages
        default_prompt = """Analyze this document page image and provide a comprehensive description including:
1. Main headings or titles
2. Key text content and paragraphs
3. Any tables, charts, or diagrams present
4. Lists or bullet points
5. Overall structure and layout
6. Any images or visual elements

Be detailed and capture all important information from the document page."""
        
        prompt = custom_prompt if custom_prompt else default_prompt
        
        # Read and encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Call GPT-5 Vision API
        try:
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
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                # max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
    def process_single_doc(
        self,
        doc_path: str,
        custom_prompt: Optional[str] = None,
        temp_dir: str = None
    ) -> Dict[str, Any]:
        """
        Process a single DOC/DOCX file
        
        Args:
            doc_path: Path to the document file
            custom_prompt: Optional custom prompt for descriptions
            temp_dir: Temporary directory for intermediate files
        
        Returns:
            Dictionary containing document data and page information
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        doc_name = Path(doc_path).name
        
        # Step 1: Convert DOC to PDF
        pdf_path = self.convert_doc_to_pdf(doc_path, temp_dir)
        
        # Step 2: Convert PDF pages to images
        image_dir = os.path.join(temp_dir, f"{Path(doc_path).stem}_images")
        os.makedirs(image_dir, exist_ok=True)
        image_paths = self.convert_pdf_to_images(pdf_path, image_dir)
        
        # Step 3: Generate descriptions for each page
        pages = []
        for i, image_path in enumerate(image_paths):
            description = self.get_image_description(image_path, custom_prompt)
            
            # Get image metadata
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
                file_size = os.path.getsize(image_path)
            
            page_data = {
                'page_number': i + 1,
                'image_path': image_path,
                'description': description,
                'image_width': width,
                'image_height': height,
                'image_format': format_name,
                'image_resolution': f"{width}x{height}",
                'image_size_bytes': file_size
            }
            
            pages.append(page_data)
        
        return {
            'doc_name': doc_name,
            'doc_path': doc_path,
            'total_pages': len(pages),
            'pages': pages,
            'pdf_path': pdf_path
        }
    
    def process_multiple_docs(
        self,
        doc_paths: List[str],
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple DOC/DOCX files in parallel
        
        Args:
            doc_paths: List of paths to document files
            custom_prompt: Optional custom prompt for descriptions
        
        Returns:
            List of dictionaries containing document data
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_single_doc, doc_path, custom_prompt): doc_path
                for doc_path in doc_paths
            }
            
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {doc_path}: {e}")
        
        return results
    
    def create_documents_from_doc_data(self, doc_data: Dict[str, Any]) -> List[Document]:
        """
        Create LangChain documents from processed DOC data
        
        Args:
            doc_data: Dictionary containing document data from process_single_doc
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for page in doc_data['pages']:
            # Create metadata
            metadata = {
                'source': 'doc',
                'content_type': 'doc_page',
                'doc_name': doc_data['doc_name'],
                'doc_path': doc_data['doc_path'],
                'page_number': page['page_number'],
                'total_pages': doc_data['total_pages'],
                'page_image_path': page['image_path'],
                'page_image_width': page['image_width'],
                'page_image_height': page['image_height'],
                'page_image_format': page['image_format'],
                'page_image_resolution': page['image_resolution'],
                'page_image_size_bytes': page['image_size_bytes']
            }
            
            # Create document with description as content
            doc = Document(
                page_content=page['description'],
                metadata=metadata
            )
            
            documents.append(doc)
        
        return documents
    
    def process_and_create_documents(
        self,
        doc_paths: List[str],
        custom_prompt: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], List[Document]]:
        """
        Process DOC files and create LangChain documents
        
        Args:
            doc_paths: List of paths to document files
            custom_prompt: Optional custom prompt for descriptions
        
        Returns:
            Tuple of (processed_data_list, langchain_documents)
        """
        # Process all docs
        doc_data_list = self.process_multiple_docs(doc_paths, custom_prompt)
        
        # Create LangChain documents
        all_documents = []
        for doc_data in doc_data_list:
            documents = self.create_documents_from_doc_data(doc_data)
            all_documents.extend(documents)
        
        return doc_data_list, all_documents


def save_uploaded_doc(uploaded_file, save_dir: str = "uploaded_documents") -> str:
    """
    Save an uploaded DOC/DOCX file to disk
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        save_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path
