"""
PDF Processing Module with Image/Table Extraction and GPT-5 Vision Integration
"""

import base64
import io
import os
import uuid
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
import pdfplumber
from PIL import Image
from openai import AzureOpenAI
from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_classic.schema import Document

# Docling imports for table extraction
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import PdfFormatOption

# Load environment variables
load_dotenv()

# GPT-5 Vision credentials from environment
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")


@dataclass
class ChunkMetadata:
    """Metadata structure for each document chunk"""
    pdf_name: str
    pdf_path: str
    semantic_topic: str
    source_pdf_unique_id: str
    chunk_type: str  # 'text', 'table', 'image'
    chunk_text: str
    source_file: str
    chunk_index: int
    table_structure: str = ""
    table_context: str = ""
    source_page: int = 0
    image_figure_name: str = ""
    image_figure_table_context: str = ""


class PDFProcessor:
    """Process PDFs with parallel extraction of text, images, and tables"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.vision_client = AzureOpenAI(
            api_key=VISION_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=VISION_ENDPOINT
        )
        
        # Initialize Docling converter for table extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_image_description_gpt5(self, image: Image.Image) -> str:
        """Get image description using GPT-5 Vision"""
        try:
            base64_image = self.encode_image(image)
            
            response = self.vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image/figure/table from a PDF document. Include key information, data, labels, and context."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                # max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def get_table_description_gpt5(self, table_df: pd.DataFrame, page_num: int) -> Tuple[str, str]:
        """Get table structure and context using GPT-5 with DataFrame"""
        try:
            # Convert DataFrame to a readable text format
            table_preview = table_df.head(10).to_string(index=False)
            table_info = f"Columns: {list(table_df.columns)}\nRows: {len(table_df)}\nPreview:\n{table_preview}"
            
            response = self.vision_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this table extracted from a PDF (Page {page_num}).

Table Information:
{table_info}

Provide:
1) A concise description of the table structure (what columns represent)
2) Key insights, patterns, or important information from the data

Format your response as:
STRUCTURE: <description>
INSIGHTS: <insights>"""
                    }
                ],
                # max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Parse the response
            structure = ""
            insights = ""
            
            if "STRUCTURE:" in content and "INSIGHTS:" in content:
                parts = content.split("INSIGHTS:")
                structure = parts[0].replace("STRUCTURE:", "").strip()
                insights = parts[1].strip()
            else:
                # Fallback parsing
                lines = content.strip().split("\n")
                structure = lines[0] if lines else content
                insights = " ".join(lines[1:]) if len(lines) > 1 else ""
            
            return structure, insights
        except Exception as e:
            return f"Error analyzing table: {str(e)}", ""
    
    def extract_images_from_page(self, pdf_path: str, page_num: int, page) -> List[Dict[str, Any]]:
        """Extract images from a PDF page"""
        images_data = []
        
        try:
            if hasattr(page, 'images'):
                for img_idx, img in enumerate(page.images):
                    try:
                        # Extract image
                        x0, y0, x1, y1 = img['x0'], img['top'], img['x1'], img['bottom']
                        cropped = page.within_bbox((x0, y0, x1, y1))
                        img_obj = cropped.to_image()
                        pil_image = img_obj.original
                        
                        # Get GPT-5 description
                        description = self.get_image_description_gpt5(pil_image)
                        
                        images_data.append({
                            'page': page_num,
                            'index': img_idx,
                            'name': f"page{page_num}_img{img_idx}",
                            'description': description,
                            'image': pil_image
                        })
                    except Exception as e:
                        print(f"Error extracting image {img_idx} from page {page_num}: {e}")
        except Exception as e:
            print(f"Error processing images on page {page_num}: {e}")
        
        return images_data
    
    def extract_tables_with_docling(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using Docling with GPT-5 descriptions"""
        tables_data = []
        
        try:
            print(f"🔍 Extracting tables from {pdf_path} using Docling...")
            
            # Convert the document using Docling
            conv_result = self.doc_converter.convert(pdf_path)
            doc = conv_result.document
            
            # Iterate through all tables identified by Docling
            for table_idx, table in enumerate(doc.tables):
                try:
                    # Convert Docling table to Pandas DataFrame
                    df = table.export_to_dataframe(doc)
                    
                    # Get page number from table provenance
                    page_num = table.prov[0].page_no if table.prov else 0
                    
                    # Convert DataFrame to text format
                    table_text = df.to_string(index=False)
                    
                    # Get GPT-5 description and insights
                    print(f"  📊 Analyzing Table {table_idx + 1} (Page {page_num}) with GPT-5...")
                    structure, context = self.get_table_description_gpt5(df, page_num)
                    
                    tables_data.append({
                        'page': page_num,
                        'index': table_idx,
                        'text': table_text,
                        'dataframe': df,  # Store DataFrame for later use
                        'structure': structure,
                        'context': context,
                        'columns': list(df.columns),
                        'row_count': len(df)
                    })
                    
                    print(f"  ✅ Table {table_idx + 1} extracted successfully")
                    
                except Exception as e:
                    print(f"  ❌ Error processing table {table_idx + 1}: {e}")
                    continue
            
            if not tables_data:
                print("  ℹ️  No tables found in the document")
            else:
                print(f"  ✨ Successfully extracted {len(tables_data)} tables")
                
        except Exception as e:
            print(f"❌ Error extracting tables with Docling: {e}")
        
        return tables_data
    
    def process_single_pdf(self, pdf_path: str, extract_tables_dir: str = None) -> Dict[str, Any]:
        """Process a single PDF and extract all content"""
        pdf_name = os.path.basename(pdf_path)
        pdf_id = str(uuid.uuid4())
        
        text_by_page = []
        all_images = []
        all_tables = []
        
        try:
            # Extract tables using Docling (separate from pdfplumber)
            print(f"\n📄 Processing PDF: {pdf_name}")
            all_tables = self.extract_tables_with_docling(pdf_path)
            
            # Optionally save tables to CSV
            if extract_tables_dir and all_tables:
                Path(extract_tables_dir).mkdir(parents=True, exist_ok=True)
                for table_data in all_tables:
                    df = table_data['dataframe']
                    csv_filename = Path(extract_tables_dir) / f"table_{table_data['index'] + 1}_page_{table_data['page']}.csv"
                    df.to_csv(csv_filename, index=False)
                    print(f"  💾 Saved table to {csv_filename}")
            
            # Extract text and images using pdfplumber
            print(f"📝 Extracting text and images...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    text = page.extract_text() or ""
                    text_by_page.append({
                        'page': page_num,
                        'text': text
                    })
                    
                    # Extract images
                    images = self.extract_images_from_page(pdf_path, page_num, page)
                    all_images.extend(images)
            
            print(f"✅ Completed processing {pdf_name}")
            print(f"  📊 Tables: {len(all_tables)}")
            print(f"  🖼️  Images: {len(all_images)}")
            print(f"  📄 Pages: {len(text_by_page)}")
        
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return None
        
        return {
            'pdf_name': pdf_name,
            'pdf_path': pdf_path,
            'pdf_id': pdf_id,
            'text_by_page': text_by_page,
            'images': all_images,
            'tables': all_tables
        }
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDFs in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {executor.submit(self.process_single_pdf, pdf_path): pdf_path 
                           for pdf_path in pdf_paths}
            
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
        
        return results
    
    def create_chunks(
        self,
        pdf_data: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive"
    ) -> List[Document]:
        """Create chunks from processed PDF data with metadata"""
        
        # Initialize text splitter
        if splitter_type == "character":
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        elif splitter_type == "token":
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:  # recursive (default)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        documents = []
        chunk_index = 0
        
        # Process text chunks
        for page_data in pdf_data['text_by_page']:
            if page_data['text'].strip():
                chunks = text_splitter.split_text(page_data['text'])
                
                for chunk in chunks:
                    metadata = ChunkMetadata(
                        pdf_name=pdf_data['pdf_name'],
                        pdf_path=pdf_data['pdf_path'],
                        semantic_topic="",  # Can be enriched later
                        source_pdf_unique_id=pdf_data['pdf_id'],
                        chunk_type="text",
                        chunk_text=chunk,
                        source_file=pdf_data['pdf_name'],
                        chunk_index=chunk_index,
                        source_page=page_data['page']
                    )
                    
                    doc = Document(
                        page_content=chunk,
                        metadata=asdict(metadata)
                    )
                    documents.append(doc)
                    chunk_index += 1
        
        # Process table chunks
        for table_data in pdf_data['tables']:
            metadata = ChunkMetadata(
                pdf_name=pdf_data['pdf_name'],
                pdf_path=pdf_data['pdf_path'],
                semantic_topic="",
                source_pdf_unique_id=pdf_data['pdf_id'],
                chunk_type="table",
                chunk_text=table_data['text'],
                source_file=pdf_data['pdf_name'],
                chunk_index=chunk_index,
                table_structure=table_data['structure'],
                table_context=table_data['context'],
                source_page=table_data['page'],
                image_figure_table_context=table_data['context']
            )
            
            # Combine table text with context for embedding
            combined_text = f"TABLE:\n{table_data['text']}\n\nContext: {table_data['context']}"
            
            doc = Document(
                page_content=combined_text,
                metadata=asdict(metadata)
            )
            documents.append(doc)
            chunk_index += 1
        
        # Process image chunks
        for image_data in pdf_data['images']:
            metadata = ChunkMetadata(
                pdf_name=pdf_data['pdf_name'],
                pdf_path=pdf_data['pdf_path'],
                semantic_topic="",
                source_pdf_unique_id=pdf_data['pdf_id'],
                chunk_type="image",
                chunk_text=image_data['description'],
                source_file=pdf_data['pdf_name'],
                chunk_index=chunk_index,
                source_page=image_data['page'],
                image_figure_name=image_data['name'],
                image_figure_table_context=image_data['description']
            )
            
            doc = Document(
                page_content=f"IMAGE: {image_data['name']}\n{image_data['description']}",
                metadata=asdict(metadata)
            )
            documents.append(doc)
            chunk_index += 1
        
        return documents
