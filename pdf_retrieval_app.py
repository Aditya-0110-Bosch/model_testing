"""
Streamlit PDF Embedding and Retrieval Experimentation Interface
"""

import os
import time
import logging
import traceback
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import AzureOpenAI

from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager, RetrievalConfig
from evaluation import RetrievalEvaluator
from image_processor import ImageProcessor, save_uploaded_image, supported_image_formats
from ppt_processor import PPTProcessor, supported_ppt_formats
from doc_processor import DocProcessor, save_uploaded_doc, supported_doc_formats

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPT-5 credentials for RAG generation
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")

# Validate required environment variables
if not all([VISION_API_KEY, VISION_ENDPOINT]):
    error_msg = "Missing required environment variables: VISION_API_KEY and/or VISION_ENDPOINT"
    logger.error(error_msg)
    raise ValueError(error_msg)

# ========== Production Utilities ==========

from functools import wraps
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

def retry_with_exponential_backoff(
    func=None,
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (RateLimitError, APIError, APIConnectionError, APITimeoutError)
):
    """Retry decorator with exponential backoff for API calls"""
    
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay
            
            while True:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {f.__name__}. Last error: {str(e)}")
                        raise
                    
                    import random
                    sleep_time = delay * (exponential_base ** (num_retries - 1)) * (0.5 + random.random())
                    logger.warning(f"Retry {num_retries}/{max_retries} for {f.__name__} after {sleep_time:.2f}s. Error: {str(e)}")
                    time.sleep(sleep_time)
                except Exception as e:
                    logger.error(f"Non-retryable error in {f.__name__}: {str(e)}", exc_info=True)
                    raise
        return wrapper
    
    return decorator if func is None else decorator(func)

# Page configuration
st.set_page_config(
    page_title="PDF Retrieval Experimentation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .doc-card {
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = None
if 'ppt_processor' not in st.session_state:
    st.session_state.ppt_processor = None
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = None
if 'vector_store_manager' not in st.session_state:
    st.session_state.vector_store_manager = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = RetrievalEvaluator()
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'processed_ppts' not in st.session_state:
    st.session_state.processed_ppts = []
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'pdf_documents' not in st.session_state:
    st.session_state.pdf_documents = []
if 'image_documents' not in st.session_state:
    st.session_state.image_documents = []
if 'ppt_documents' not in st.session_state:
    st.session_state.ppt_documents = []
if 'doc_documents' not in st.session_state:
    st.session_state.doc_documents = []
if 'retrieval_results' not in st.session_state:
    st.session_state.retrieval_results = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'rag_client' not in st.session_state:
    st.session_state.rag_client = AzureOpenAI(
        api_key=VISION_API_KEY,
        api_version=VISION_API_VERSION,
        azure_endpoint=VISION_ENDPOINT
    )


def render_sidebar():
    """Render sidebar with configuration options"""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Processing Configuration
        st.header("PDF Processing")
        max_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=8,
            value=8,
            help="Number of parallel threads for processing"
        )
        
        # Chunking Configuration
        st.header("Chunking Settings")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of each text chunk in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        splitter_type = st.selectbox(
            "Text Splitter",
            options=["recursive", "character", "token"],
            index=0,
            help="Strategy for splitting text into chunks"
        )
        
        # FAISS Configuration
        st.header("FAISS Settings")
        
        index_type = st.selectbox(
            "Distance Metric",
            options=["cosine", "l2", "ip"],
            index=0,
            help="Distance metric for similarity search"
        )
        
        k_neighbors = st.slider(
            "K (Top Results)",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of top results to retrieve"
        )
        
        # Store config in session state
        st.session_state.config = RetrievalConfig(
            k=k_neighbors,
            index_type=index_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type
        )
        st.session_state.max_workers = max_workers
        
        # Vector Store Management
        st.header("Vector Store Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save to DB", use_container_width=True):
                save_vector_store()
        
        with col2:
            if st.button("📂 Load from DB", use_container_width=True):
                load_vector_store()
        
        st.divider()
        
        if st.button("🗑️ Clear Vector Store", use_container_width=True, type="secondary"):
            clear_vector_store()


def save_vector_store():
    """Save combined vector store to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if st.session_state.vector_store_manager and st.session_state.vector_store_manager.vector_store:
            save_path = f"vector_stores/combined_{timestamp}"
            st.session_state.vector_store_manager.save_vector_store(save_path)
            st.success(f"✅ Vector store saved to {save_path}")
            st.info(f"📊 Saved {len(st.session_state.all_documents)} documents (All formats combined)")
        else:
            st.warning("⚠️ No vector store to save. Please process documents first.")
    
    except Exception as e:
        error_msg = f"Error saving vector store: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())


def load_vector_store():
    """Load combined vector store from disk"""
    
    # List available stores
    if os.path.exists("vector_stores"):
        all_stores = [d for d in os.listdir("vector_stores") if os.path.isdir(os.path.join("vector_stores", d))]
        stores = [s for s in all_stores if s.startswith("combined_") or s.startswith("store_")]
        
        if stores:
            selected_store = st.sidebar.selectbox("Select Vector Store", stores, key="load_store")
            
            try:
                if st.session_state.vector_store_manager is None:
                    st.session_state.vector_store_manager = VectorStoreManager(
                        max_workers=st.session_state.max_workers,
                        index_type=st.session_state.config.index_type
                    )
                
                st.session_state.vector_store_manager.load_vector_store(
                    os.path.join("vector_stores", selected_store)
                )
                st.success(f"✅ Vector store loaded from {selected_store}")
                st.info("📊 Loaded combined vector store with all document formats")
                    
            except Exception as e:
                error_msg = f"Error loading vector store: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(f"❌ {error_msg}")
                st.error(traceback.format_exc())
        else:
            st.warning("⚠️ No vector stores found.")
    else:
        st.warning("⚠️ No vector_stores directory found.")


def clear_vector_store():
    """Clear the current vector store from memory"""
    try:
        st.session_state.vector_store_manager = None
        st.session_state.all_documents = []
        st.session_state.pdf_documents = []
        st.session_state.image_documents = []
        st.session_state.ppt_documents = []
        st.session_state.doc_documents = []
        st.session_state.retrieval_results = []
        st.success("✅ Vector store cleared from memory")
        st.info("💡 You can now upload new documents or load a saved vector store")
    except Exception as e:
        st.error(f"❌ Error clearing vector store: {e}")


def render_unified_upload_tab():
    """Render unified upload tab for all file formats"""
    
    st.header("📤 Upload and Process Documents")
    st.markdown("*Upload PDF, Image, PowerPoint, or DOC/DOCX files - all formats supported*")
    
    # Single file uploader for all formats
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Images, PPT, DOC/DOCX)",
        type=['pdf', 'png', 'jpg', 'jpeg', 'ppt', 'pptx', 'doc', 'docx'],
        accept_multiple_files=True,
        help="Select one or more files of any supported format",
        key="unified_uploader"
    )
    
    if uploaded_files:
        # Categorize files by type
        pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
        image_files = [f for f in uploaded_files if f.name.lower().endswith(tuple(f'.{ext}' for ext in supported_image_formats()))]
        ppt_files = [f for f in uploaded_files if f.name.lower().endswith(tuple(f'.{ext}' for ext in supported_ppt_formats()))]
        doc_files = [f for f in uploaded_files if f.name.lower().endswith(tuple(f'.{ext}' for ext in supported_doc_formats()))]
        
        # Display file summary
        st.info(f"""
        **📁 {len(uploaded_files)} file(s) selected:**
        - 📄 PDFs: {len(pdf_files)}
        - 🖼️ Images: {len(image_files)}
        - 📊 PowerPoints: {len(ppt_files)}
        - 📝 DOC/DOCX: {len(doc_files)}
        """)
        
        # Show file list
        with st.expander("📋 View uploaded files"):
            for file in uploaded_files:
                file_type_icon = {
                    'pdf': '📄',
                    'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'tiff': '🖼️', 'webp': '🖼️',
                    'ppt': '📊', 'pptx': '📊',
                    'doc': '📝', 'docx': '📝'
                }.get(file.name.split('.')[-1].lower(), '📄')
                
                st.write(f"{file_type_icon} {file.name} ({file.size / 1024:.2f} KB)")
        
        # Custom prompt option for vision-based processing
        st.subheader("⚙️ Vision Description Settings (Optional)")
        
        use_custom_prompt = st.checkbox(
            "Use custom prompt for AI-generated descriptions",
            value=False,
            help="Provide a custom prompt for GPT-5 Vision to describe images, slides, and document pages"
        )
        
        custom_prompt = None
        if use_custom_prompt:
            custom_prompt = st.text_area(
                "Custom Prompt",
                value="""Analyze this image/slide/document page and provide a comprehensive description including:
1. Main title/heading or subject
2. Key text content and bullet points
3. Visual elements (charts, diagrams, images, tables)
4. Overall structure and layout
5. Purpose or message

Be detailed and capture all important information.""",
                height=200,
                help="This prompt will be used for images, PPT slides, and DOC pages"
            )
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button("🚀 Process All Files", type="primary", use_container_width=True)
        with col2:
            append_info = st.info("📌 Files will be added to existing vector store")
        
        if process_button:
            process_all_files(pdf_files, image_files, ppt_files, doc_files, custom_prompt)


def _save_files_to_disk(pdf_files, image_files, ppt_files, doc_files):
    """Save all uploaded files to disk in parallel (I/O bound). 
    Returns paths dict keyed by format type.
    Must be called from the main thread since it reads Streamlit UploadedFile buffers."""
    
    paths = {'pdf': [], 'image': [], 'ppt': [], 'doc': []}
    
    # Save PDFs
    if pdf_files:
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        for f in pdf_files:
            fp = os.path.join(temp_dir, f.name)
            with open(fp, "wb") as out:
                out.write(f.getbuffer())
            paths['pdf'].append(fp)
    
    # Save Images
    if image_files:
        temp_dir = "uploaded_images"
        os.makedirs(temp_dir, exist_ok=True)
        for f in image_files:
            fp = save_uploaded_image(f, temp_dir)
            paths['image'].append(fp)
    
    # Save PPTs
    if ppt_files:
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        for f in ppt_files:
            fp = os.path.join(temp_dir, f.name)
            with open(fp, "wb") as out:
                out.write(f.getbuffer())
            paths['ppt'].append(fp)
    
    # Save DOCs
    if doc_files:
        temp_dir = "uploaded_documents"
        os.makedirs(temp_dir, exist_ok=True)
        for f in doc_files:
            fp = save_uploaded_doc(f, temp_dir)
            paths['doc'].append(fp)
    
    return paths


def _ensure_processors_initialized():
    """Pre-initialize all processors so threads don't race on lazy init."""
    mw = st.session_state.max_workers
    if st.session_state.processor is None:
        st.session_state.processor = PDFProcessor(max_workers=mw)
    if st.session_state.image_processor is None:
        st.session_state.image_processor = ImageProcessor(max_workers=mw)
    if st.session_state.ppt_processor is None:
        st.session_state.ppt_processor = PPTProcessor(max_workers=mw)
    if st.session_state.doc_processor is None:
        st.session_state.doc_processor = DocProcessor(max_workers=mw)


def _process_pdfs_worker(pdf_paths, processor, config):
    """Thread-safe PDF processing worker (no Streamlit calls)."""
    processed_data = processor.process_multiple_pdfs(pdf_paths)
    all_documents = []
    for pdf_data in processed_data:
        documents = processor.create_chunks(
            pdf_data,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            splitter_type=config.splitter_type
        )
        all_documents.extend(documents)
    return processed_data, all_documents


def _process_images_worker(image_paths, image_processor, custom_prompt):
    """Thread-safe image processing worker (no Streamlit calls)."""
    image_data_list, image_documents = image_processor.process_and_create_documents(
        image_paths, custom_prompt=custom_prompt
    )
    return image_data_list, image_documents


def _process_ppts_worker(ppt_paths, ppt_processor, custom_prompt):
    """Thread-safe PPT processing worker (no Streamlit calls)."""
    ppt_data_list, ppt_documents = ppt_processor.process_and_create_documents(
        ppt_paths, custom_prompt=custom_prompt
    )
    return ppt_data_list, ppt_documents


def _process_docs_worker(doc_paths, doc_processor, custom_prompt):
    """Thread-safe DOC processing worker (no Streamlit calls)."""
    doc_data_list, doc_documents = doc_processor.process_and_create_documents(
        doc_paths, custom_prompt=custom_prompt
    )
    return doc_data_list, doc_documents


def process_all_files(pdf_files, image_files, ppt_files, doc_files, custom_prompt=None):
    """Process all uploaded files IN PARALLEL and add to vector store.
    
    Parallelism strategy:
    1. Save all files to disk first (main thread, reads Streamlit buffers)
    2. Pre-initialize all processors (avoids thread-unsafe lazy init)
    3. Launch one thread per format type — PDF, Image, PPT, DOC run concurrently
    4. Collect results and batch-create the vector store with parallel embeddings
    """
    
    all_new_documents = []
    total_files = len(pdf_files) + len(image_files) + len(ppt_files) + len(doc_files)
    
    if total_files == 0:
        st.warning("⚠️ No files to process")
        return
    
    logger.info(f"Starting processing of {total_files} files: {len(pdf_files)} PDFs, {len(image_files)} images, {len(ppt_files)} PPTs, {len(doc_files)} DOCs")
    
    # Create main progress tracking
    main_progress = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    try:
        # ── Step 1: Save all files to disk (main thread, I/O) ──
        status_text.text("💾 Saving uploaded files to disk...")
        logger.info("Step 1: Saving files to disk")
        save_start = time.time()
        file_paths = _save_files_to_disk(pdf_files, image_files, ppt_files, doc_files)
        logger.info(f"Files saved in {time.time() - save_start:.2f}s")
        main_progress.progress(0.05)
        
        # ── Step 2: Pre-initialize all processors (main thread) ──
        status_text.text("⚙️ Initializing processors...")
        logger.info("Step 2: Initializing processors")
        init_start = time.time()
        _ensure_processors_initialized()
        logger.info(f"Processors initialized in {time.time() - init_start:.2f}s")
        main_progress.progress(0.10)
        
        # Grab references for thread-safe access (read-only, no mutation)
        processor = st.session_state.processor
        image_processor = st.session_state.image_processor
        ppt_processor = st.session_state.ppt_processor
        doc_processor = st.session_state.doc_processor
        config = st.session_state.config
        
        # ── Step 3: Process ALL formats in parallel ──
        status_text.text("🚀 Processing all file formats in parallel...")
        logger.info("Step 3: Starting parallel processing of all formats")
        processing_start = time.time()
        
        format_futures = {}
        # Use one thread per format type for max concurrency
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="format") as executor:
            if file_paths['pdf']:
                logger.info(f"Submitting {len(file_paths['pdf'])} PDFs for processing")
                format_futures['pdf'] = executor.submit(
                    _process_pdfs_worker, file_paths['pdf'], processor, config
                )
            if file_paths['image']:
                logger.info(f"Submitting {len(file_paths['image'])} images for processing")
                format_futures['image'] = executor.submit(
                    _process_images_worker, file_paths['image'], image_processor, custom_prompt
                )
            if file_paths['ppt']:
                logger.info(f"Submitting {len(file_paths['ppt'])} PPTs for processing")
                format_futures['ppt'] = executor.submit(
                    _process_ppts_worker, file_paths['ppt'], ppt_processor, custom_prompt
                )
            if file_paths['doc']:
                logger.info(f"Submitting {len(file_paths['doc'])} DOCs for processing")
                format_futures['doc'] = executor.submit(
                    _process_docs_worker, file_paths['doc'], doc_processor, custom_prompt
                )
            
            # Track completion as formats finish
            completed = 0
            total_formats = len(format_futures)
            format_labels = {'pdf': '📄 PDF', 'image': '🖼️ Image', 'ppt': '📊 PPT', 'doc': '📝 DOC'}
            
            for future in as_completed(format_futures.values()):
                completed += 1
                # Find which format just completed
                fmt_name = next(k for k, v in format_futures.items() if v is future)
                elapsed = time.time() - start_time
                
                # Check for errors
                try:
                    future.result()
                    logger.info(f"Format {fmt_name} completed successfully")
                except Exception as e:
                    logger.error(f"Format {fmt_name} failed: {str(e)}", exc_info=True)
                    raise
                
                status_text.text(
                    f"✅ {format_labels[fmt_name]} done  |  "
                    f"{completed}/{total_formats} formats complete  |  "
                    f"{elapsed:.1f}s elapsed"
                )
                main_progress.progress(0.10 + 0.70 * (completed / total_formats))
        
        processing_time = time.time() - processing_start
        logger.info(f"All formats processed in {processing_time:.2f}s")
        
        # ── Step 4: Collect results from all futures ──
        status_text.text("📦 Collecting processed documents...")
        logger.info("Step 4: Collecting results from all format processors")
        
        if 'pdf' in format_futures:
            processed_data, pdf_documents = format_futures['pdf'].result()
            st.session_state.processed_pdfs.extend(processed_data)
            st.session_state.pdf_documents.extend(pdf_documents)
            all_new_documents.extend(pdf_documents)
            logger.info(f"Collected {len(pdf_documents)} documents from PDF processing")
        
        if 'image' in format_futures:
            image_data_list, image_documents = format_futures['image'].result()
            st.session_state.processed_images.extend(image_data_list)
            st.session_state.image_documents.extend(image_documents)
            all_new_documents.extend(image_documents)
            logger.info(f"Collected {len(image_documents)} documents from image processing")
        
        if 'ppt' in format_futures:
            ppt_data_list, ppt_documents = format_futures['ppt'].result()
            st.session_state.processed_ppts.extend(ppt_data_list)
            st.session_state.ppt_documents.extend(ppt_documents)
            all_new_documents.extend(ppt_documents)
            logger.info(f"Collected {len(ppt_documents)} documents from PPT processing")
        
        if 'doc' in format_futures:
            doc_data_list, doc_documents = format_futures['doc'].result()
            st.session_state.processed_docs.extend(doc_data_list)
            st.session_state.doc_documents.extend(doc_documents)
            all_new_documents.extend(doc_documents)
            logger.info(f"Collected {len(doc_documents)} documents from DOC processing")
        
        main_progress.progress(0.85)
        
        # ── Step 5: Add to all_documents ──
        st.session_state.all_documents.extend(all_new_documents)
        logger.info(f"Total documents in memory: {len(st.session_state.all_documents)}")
        
        # ── Step 6: Create or update vector store (parallel embeddings inside) ──
        status_text.text(f"🔢 Creating embeddings for {len(all_new_documents)} documents (parallel batches)...")
        logger.info(f"Step 6: Creating/updating vector store with {len(all_new_documents)} new documents")
        embedding_start = time.time()
        
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                max_workers=st.session_state.max_workers,
                index_type=st.session_state.config.index_type
            )
            st.session_state.vector_store_manager.create_vector_store(all_new_documents)
            logger.info("Created new vector store")
        else:
            st.session_state.vector_store_manager.add_documents(all_new_documents)
            logger.info("Added documents to existing vector store")
        
        embedding_time = time.time() - embedding_start
        logger.info(f"Vector store operations completed in {embedding_time:.2f}s")
        
        main_progress.progress(1.0)
        total_time = time.time() - start_time
        status_text.text(f"✅ Processing complete in {total_time:.1f}s!")
        
        logger.info(f"="*60)
        logger.info(f"PROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Processing time breakdown:")
        logger.info(f"  - File saving: {save_start and (time.time() - save_start):.2f}s")
        logger.info(f"  - Processor init: {init_start and (time.time() - init_start):.2f}s")
        logger.info(f"  - Format processing: {processing_time:.2f}s")
        logger.info(f"  - Vector store: {embedding_time:.2f}s")
        logger.info(f"Documents created: {len(all_new_documents)}")
        logger.info(f"="*60)
        
        # Show summary
        st.success(f"""
        **✅ Processing Summary (completed in {total_time:.1f}s):**
        - Total files processed: {total_files}
        - Documents created: {len(all_new_documents)}
        - PDFs: {len(pdf_files)} files
        - Images: {len(image_files)} files
        - PowerPoints: {len(ppt_files)} files
        - DOC/DOCX: {len(doc_files)} files
        
        **📊 Vector Store Status:**
        - Total documents in store: {len(st.session_state.all_documents)}
        - New documents added: {len(all_new_documents)}
        
        **⚡ Parallelism: All {len(format_futures)} format(s) processed concurrently**
        """)
        
        # Display statistics
        render_statistics()
        
    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())
    
    finally:
        main_progress.empty()
        status_text.empty()


def process_pdfs_unified(pdf_files):
    """Process PDF files and return documents (used by individual PDF tab)"""
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_paths = []
    for uploaded_file in pdf_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
    
    if st.session_state.processor is None:
        st.session_state.processor = PDFProcessor(max_workers=st.session_state.max_workers)
    
    processed_data, all_documents = _process_pdfs_worker(
        pdf_paths, st.session_state.processor, st.session_state.config
    )
    st.session_state.processed_pdfs.extend(processed_data)
    
    return all_documents


def process_images_unified(image_files, custom_prompt=None):
    """Process image files and return documents (used by individual Image tab)"""
    temp_dir = "uploaded_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    for uploaded_image in image_files:
        file_path = save_uploaded_image(uploaded_image, temp_dir)
        image_paths.append(file_path)
    
    if st.session_state.image_processor is None:
        st.session_state.image_processor = ImageProcessor(max_workers=st.session_state.max_workers)
    
    image_data_list, image_documents = _process_images_worker(
        image_paths, st.session_state.image_processor, custom_prompt
    )
    
    st.session_state.processed_images.extend(image_data_list)
    return image_documents


def process_ppts_unified(ppt_files, custom_prompt=None):
    """Process PowerPoint files and return documents (used by individual PPT tab)"""
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    ppt_paths = []
    for uploaded_ppt in ppt_files:
        file_path = os.path.join(temp_dir, uploaded_ppt.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_ppt.getbuffer())
        ppt_paths.append(file_path)
    
    if st.session_state.ppt_processor is None:
        st.session_state.ppt_processor = PPTProcessor(max_workers=st.session_state.max_workers)
    
    ppt_data_list, ppt_documents = _process_ppts_worker(
        ppt_paths, st.session_state.ppt_processor, custom_prompt
    )
    
    st.session_state.processed_ppts.extend(ppt_data_list)
    return ppt_documents


def process_docs_unified(doc_files, custom_prompt=None):
    """Process DOC/DOCX files and return documents (used by individual DOC tab)"""
    temp_dir = "uploaded_documents"
    os.makedirs(temp_dir, exist_ok=True)
    
    doc_paths = []
    for uploaded_doc in doc_files:
        file_path = save_uploaded_doc(uploaded_doc, temp_dir)
        doc_paths.append(file_path)
    
    if st.session_state.doc_processor is None:
        st.session_state.doc_processor = DocProcessor(max_workers=st.session_state.max_workers)
    
    doc_data_list, doc_documents = _process_docs_worker(
        doc_paths, st.session_state.doc_processor, custom_prompt
    )
    
    st.session_state.processed_docs.extend(doc_data_list)
    return doc_documents


def render_upload_tab():
    """Render PDF upload and processing tab"""
    
    st.header("📤 Upload and Process PDFs")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF files to process",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
        # Show file list
        with st.expander("📋 View uploaded files"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.2f} KB)")
        
        # Process button
        if st.button("🚀 Process PDFs", type="primary"):
            process_pdfs(uploaded_files)


def process_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    
    # Save uploaded files temporarily
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
    
    # Initialize processor
    if st.session_state.processor is None:
        st.session_state.processor = PDFProcessor(max_workers=st.session_state.max_workers)
    
    # Process PDFs with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📄 Extracting text, images, and tables from PDFs...")
    
    try:
        # Process PDFs
        processed_data = st.session_state.processor.process_multiple_pdfs(pdf_paths)
        st.session_state.processed_pdfs.extend(processed_data)
        
        progress_bar.progress(50)
        status_text.text("✂️ Creating chunks with ...")
        
        # Create chunks for all PDFs
        all_documents = []
        for pdf_data in processed_data:
            documents = st.session_state.processor.create_chunks(
                pdf_data,
                chunk_size=st.session_state.config.chunk_size,
                chunk_overlap=st.session_state.config.chunk_overlap,
                splitter_type=st.session_state.config.splitter_type
            )
            all_documents.extend(documents)
        
        st.session_state.all_documents.extend(all_documents)
        st.session_state.pdf_documents.extend(all_documents)  # Track PDF documents separately
        
        progress_bar.progress(75)
        status_text.text("🔢 Creating embeddings and building vector store...")
        
        # Create or update vector store
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                max_workers=st.session_state.max_workers,
                index_type=st.session_state.config.index_type
            )
            st.session_state.vector_store_manager.create_vector_store(all_documents)
        else:
            st.session_state.vector_store_manager.add_documents(all_documents)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Show summary
        st.success(f"""
        **Processing Summary:**
        - PDFs processed: {len(processed_data)}
        - Total chunks created: {len(all_documents)}
        - Text chunks: {sum(1 for doc in all_documents if doc.metadata.get('chunk_type') == 'text')}
        - Table chunks: {sum(1 for doc in all_documents if doc.metadata.get('chunk_type') == 'table')}
        - Image chunks: {sum(1 for doc in all_documents if doc.metadata.get('chunk_type') == 'image')}
        """)
        
        # Display statistics
        render_statistics()
        
    except Exception as e:
        error_msg = f"Error processing PDFs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()


def render_image_upload_tab():
    """Render image upload and processing tab"""
    
    st.header("🖼️ Upload and Process Images")
    
    # File uploader for images
    uploaded_images = st.file_uploader(
        "Upload image files",
        type=supported_image_formats(),
        accept_multiple_files=True,
        help="Select one or more image files to process",
        key="image_uploader"
    )
    
    if uploaded_images:
        st.info(f"🖼️ {len(uploaded_images)} image(s) selected")
        
        # Show image previews
        with st.expander("🖼️ View uploaded images", expanded=True):
            cols = st.columns(min(3, len(uploaded_images)))
            for idx, img_file in enumerate(uploaded_images):
                with cols[idx % 3]:
                    st.image(img_file, caption=img_file.name, 
                             width="stretch")
                    st.caption(f"Size: {img_file.size / 1024:.2f} KB")
        
        # Custom prompt option
        st.subheader("⚙️ Description Generation Settings")
        
        use_custom_prompt = st.checkbox(
            "Use custom prompt for image description",
            value=False,
            help="Provide a custom prompt for GPT-5 to generate descriptions"
        )
        
        custom_prompt = None
        if use_custom_prompt:
            custom_prompt = st.text_area(
                "Custom Prompt",
                value="""Analyze this image and provide a detailed description including:
1. Main subject/objects in the image
2. Visual characteristics (colors, composition, style)
3. Context and setting
4. Any text or labels visible
5. Overall purpose or message of the image

Be descriptive and specific.""",
                height=200,
                help="Customize how GPT-5 should describe your images"
            )
        
        # Process button
        if st.button("🚀 Process Images", type="primary", key="process_images_btn"):
            process_images(uploaded_images, custom_prompt)


def process_images(uploaded_images, custom_prompt=None):
    """Process uploaded image files"""
    
    # Save uploaded images temporarily
    temp_dir = "uploaded_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    for uploaded_image in uploaded_images:
        file_path = save_uploaded_image(uploaded_image, temp_dir)
        image_paths.append(file_path)
    
    # Initialize image processor
    if st.session_state.image_processor is None:
        st.session_state.image_processor = ImageProcessor(max_workers=st.session_state.max_workers)
    
    # Process images with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🖼️ Extracting image metadata and generating descriptions with GPT-5...")
    
    try:
        # Process images
        image_data_list, image_documents = st.session_state.image_processor.process_and_create_documents(
            image_paths,
            custom_prompt=custom_prompt
        )
        
        st.session_state.processed_images.extend(image_data_list)
        
        progress_bar.progress(60)
        status_text.text("🔢 Creating embeddings and updating vector store...")
        
        # Add to all_documents and track images separately
        st.session_state.all_documents.extend(image_documents)
        st.session_state.image_documents.extend(image_documents)  # Track image documents separately
        
        # Create or update vector store
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                max_workers=st.session_state.max_workers,
                index_type=st.session_state.config.index_type
            )
            st.session_state.vector_store_manager.create_vector_store(image_documents)
        else:
            st.session_state.vector_store_manager.add_documents(image_documents)
        
        progress_bar.progress(100)
        status_text.text("✅ Image processing complete!")
        
        # Show summary
        st.success(f"""
        **Image Processing Summary:**
        - Images processed: {len(image_data_list)}
        - Total documents created: {len(image_documents)}
        - Descriptions generated by: GPT-5 Vision
        """)
        
        # Display image details
        render_image_statistics(image_data_list)
        
    except Exception as e:
        error_msg = f"Error processing images: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()


def render_image_statistics(image_data_list):
    """Render image processing statistics"""
    
    if image_data_list:
        st.subheader("📊 Image Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", len(image_data_list))
        
        with col2:
            avg_width = sum(img['image_width'] for img in image_data_list) / len(image_data_list)
            st.metric("Avg Width", f"{avg_width:.0f}px")
        
        with col3:
            avg_height = sum(img['image_height'] for img in image_data_list) / len(image_data_list)
            st.metric("Avg Height", f"{avg_height:.0f}px")
        
        with col4:
            total_size_mb = sum(img['image_size_bytes'] for img in image_data_list) / (1024 * 1024)
            st.metric("Total Size", f"{total_size_mb:.2f} MB")
        
        # Display image details in expandable cards
        st.subheader("🖼️ Image Details")
        
        for img_data in image_data_list:
            with st.expander(f"📸 {img_data['image_name']}", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Resolution:** {img_data['image_resolution']}")
                    st.write(f"**Format:** {img_data['image_format']}")
                    st.write(f"**Size:** {img_data['image_size_bytes'] / 1024:.2f} KB")
                    st.write(f"**Path:** `{img_data['image_path']}`")
                
                with col2:
                    st.markdown("**GPT-5 Generated Description:**")
                    st.info(img_data['image_description'])


def render_ppt_upload_tab():
    """Render PPT upload and processing tab"""
    
    st.header("📊 Upload and Process PowerPoint Presentations")
    
    # File uploader for PPTs
    uploaded_ppts = st.file_uploader(
        "Upload PowerPoint files",
        type=supported_ppt_formats(),
        accept_multiple_files=True,
        help="Select one or more PowerPoint files to process",
        key="ppt_uploader"
    )
    
    if uploaded_ppts:
        st.info(f"📊 {len(uploaded_ppts)} PowerPoint file(s) selected")
        
        # Show file list
        with st.expander("📋 View uploaded files", expanded=True):
            for ppt_file in uploaded_ppts:
                st.write(f"• {ppt_file.name} ({ppt_file.size / 1024:.2f} KB)")
        
        # Custom prompt option
        st.subheader("⚙️ Slide Description Settings")
        
        use_custom_prompt = st.checkbox(
            "Use custom prompt for slide descriptions",
            value=False,
            help="Provide a custom prompt for GPT-5 to describe slides"
        )
        
        custom_prompt = None
        if use_custom_prompt:
            custom_prompt = st.text_area(
                "Custom Prompt",
                value="""Analyze this PowerPoint slide and provide a comprehensive description including:
1. Main title/heading
2. Key points and content
3. Visual elements (charts, diagrams, images)
4. Text content and bullet points
5. Overall message or purpose of the slide

Be detailed and capture all important information.""",
                height=200,
                help="Customize how GPT-5 should describe your slides"
            )
        
        # Process button
        if st.button("🚀 Process PowerPoints", type="primary", key="process_ppts_btn"):
            process_ppts(uploaded_ppts, custom_prompt)


def process_ppts(uploaded_ppts, custom_prompt=None):
    """Process uploaded PowerPoint files"""
    
    # Save uploaded PPTs temporarily
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    ppt_paths = []
    for uploaded_ppt in uploaded_ppts:
        file_path = os.path.join(temp_dir, uploaded_ppt.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_ppt.getbuffer())
        ppt_paths.append(file_path)
    
    # Initialize PPT processor
    if st.session_state.ppt_processor is None:
        st.session_state.ppt_processor = PPTProcessor(max_workers=st.session_state.max_workers)
    
    # Process PPTs with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📊 Converting slides to images and generating descriptions with GPT-5...")
    
    try:
        # Process PPTs
        ppt_data_list, ppt_documents = st.session_state.ppt_processor.process_and_create_documents(
            ppt_paths,
            custom_prompt=custom_prompt
        )
        
        st.session_state.processed_ppts.extend(ppt_data_list)
        
        progress_bar.progress(60)
        status_text.text("🔢 Creating embeddings and updating vector store...")
        
        # Add to all_documents and track PPTs separately
        st.session_state.all_documents.extend(ppt_documents)
        st.session_state.ppt_documents.extend(ppt_documents)
        
        # Create or update vector store
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                max_workers=st.session_state.max_workers,
                index_type=st.session_state.config.index_type
            )
            st.session_state.vector_store_manager.create_vector_store(ppt_documents)
        else:
            st.session_state.vector_store_manager.add_documents(ppt_documents)
        
        progress_bar.progress(100)
        status_text.text("✅ PPT processing complete!")
        
        # Show summary
        total_slides = sum(ppt['total_slides'] for ppt in ppt_data_list)
        st.success(f"""
        **PowerPoint Processing Summary:**
        - PPT files processed: {len(ppt_data_list)}
        - Total slides: {total_slides}
        - Total documents created: {len(ppt_documents)}
        - Descriptions generated by: GPT-5 Vision
        """)
        
        # Display PPT details
        render_ppt_statistics(ppt_data_list)
        
    except Exception as e:
        error_msg = f"Error processing PowerPoints: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()


def render_ppt_statistics(ppt_data_list):
    """Render PPT processing statistics"""
    
    if ppt_data_list:
        st.subheader("📊 PowerPoint Statistics")
        
        total_slides = sum(ppt['total_slides'] for ppt in ppt_data_list)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total PPT Files", len(ppt_data_list))
        
        with col2:
            st.metric("Total Slides", total_slides)
        
        with col3:
            avg_slides = total_slides / len(ppt_data_list)
            st.metric("Avg Slides per PPT", f"{avg_slides:.1f}")
        
        # Display PPT details in expandable cards
        st.subheader("📊 PowerPoint Details")
        
        for ppt_data in ppt_data_list:
            with st.expander(f"📊 {ppt_data['ppt_name']} ({ppt_data['total_slides']} slides)", expanded=False):
                
                # Display each slide
                for idx, slide in enumerate(ppt_data['slides']):
                    st.markdown(f"### Slide {slide['slide_number']}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Display slide image
                        if os.path.exists(slide['slide_image_path']):
                            st.image(slide['slide_image_path'], width="stretch")
                        st.write(f"**Resolution:** {slide['slide_image_resolution']}")
                        st.write(f"**Format:** {slide['slide_image_format']}")
                        st.write(f"**Size:** {slide['slide_image_size_bytes'] / 1024:.2f} KB")
                    
                    with col2:
                        st.markdown("**GPT-5 Generated Description:**")
                        st.info(slide['slide_description'])
                    
                    if idx < len(ppt_data['slides']) - 1:
                        st.divider()


def render_doc_upload_tab():
    """Render DOC/DOCX upload and processing tab"""
    
    st.header("📄 Upload and Process DOC/DOCX Files")
    
    # File uploader for DOC/DOCX
    uploaded_docs = st.file_uploader(
        "Upload DOC/DOCX files",
        type=supported_doc_formats(),
        accept_multiple_files=True,
        help="Select one or more DOC/DOCX files to process",
        key="doc_uploader"
    )
    
    if uploaded_docs:
        st.info(f"📄 {len(uploaded_docs)} DOC/DOCX file(s) selected")
        
        # Show file list
        with st.expander("📋 View uploaded files", expanded=True):
            for doc_file in uploaded_docs:
                st.write(f"• {doc_file.name} ({doc_file.size / 1024:.2f} KB)")
        
        # Custom prompt option
        st.subheader("⚙️ Page Description Settings")
        
        use_custom_prompt = st.checkbox(
            "Use custom prompt for page descriptions",
            value=False,
            help="Provide a custom prompt for GPT-5 to describe document pages"
        )
        
        custom_prompt = None
        if use_custom_prompt:
            custom_prompt = st.text_area(
                "Custom Prompt",
                value="""Analyze this document page image and provide a comprehensive description including:
1. Main headings or titles
2. Key text content and paragraphs
3. Any tables, charts, or diagrams present
4. Lists or bullet points
5. Overall structure and layout
6. Any images or visual elements

Be detailed and capture all important information from the document page.""",
                height=200,
                help="Customize how GPT-5 should describe your document pages"
            )
        
        # Process button
        if st.button("🚀 Process DOC/DOCX Files", type="primary", key="process_docs_btn"):
            process_docs(uploaded_docs, custom_prompt)


def process_docs(uploaded_docs, custom_prompt=None):
    """Process uploaded DOC/DOCX files"""
    
    # Save uploaded DOCs temporarily
    temp_dir = "uploaded_documents"
    os.makedirs(temp_dir, exist_ok=True)
    
    doc_paths = []
    for uploaded_doc in uploaded_docs:
        file_path = save_uploaded_doc(uploaded_doc, temp_dir)
        doc_paths.append(file_path)
    
    # Initialize DOC processor
    if st.session_state.doc_processor is None:
        st.session_state.doc_processor = DocProcessor(max_workers=st.session_state.max_workers)
    
    # Process DOCs with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📄 Converting DOC to PDF, then to images, and generating descriptions with GPT-5...")
    
    try:
        # Process DOCs
        doc_data_list, doc_documents = st.session_state.doc_processor.process_and_create_documents(
            doc_paths,
            custom_prompt=custom_prompt
        )
        
        st.session_state.processed_docs.extend(doc_data_list)
        
        progress_bar.progress(60)
        status_text.text("🔢 Creating embeddings and updating vector store...")
        
        # Add to all_documents and track DOCs separately
        st.session_state.all_documents.extend(doc_documents)
        st.session_state.doc_documents.extend(doc_documents)
        
        # Create or update vector store
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                max_workers=st.session_state.max_workers,
                index_type=st.session_state.config.index_type
            )
            st.session_state.vector_store_manager.create_vector_store(doc_documents)
        else:
            st.session_state.vector_store_manager.add_documents(doc_documents)
        
        progress_bar.progress(100)
        status_text.text("✅ DOC/DOCX processing complete!")
        
        # Show summary
        total_pages = sum(doc['total_pages'] for doc in doc_data_list)
        st.success(f"""
        **DOC/DOCX Processing Summary:**
        - DOC/DOCX files processed: {len(doc_data_list)}
        - Total pages: {total_pages}
        - Total documents created: {len(doc_documents)}
        - Descriptions generated by: GPT-5 Vision
        """)
        
        # Display DOC details
        render_doc_statistics(doc_data_list)
        
    except Exception as e:
        error_msg = f"Error processing DOC/DOCX files: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())
    
    finally:
        progress_bar.empty()
        status_text.empty()


def render_doc_statistics(doc_data_list):
    """Render DOC processing statistics"""
    
    if doc_data_list:
        st.subheader("📊 DOC/DOCX Statistics")
        
        total_pages = sum(doc['total_pages'] for doc in doc_data_list)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total DOC/DOCX Files", len(doc_data_list))
        
        with col2:
            st.metric("Total Pages", total_pages)
        
        with col3:
            avg_pages = total_pages / len(doc_data_list)
            st.metric("Avg Pages per DOC", f"{avg_pages:.1f}")
        
        # Display DOC details in expandable cards
        st.subheader("📄 DOC/DOCX Details")
        
        for doc_data in doc_data_list:
            with st.expander(f"📄 {doc_data['doc_name']} ({doc_data['total_pages']} pages)", expanded=False):
                
                # Display each page
                for idx, page in enumerate(doc_data['pages']):
                    st.markdown(f"### Page {page['page_number']}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Display page image
                        if os.path.exists(page['image_path']):
                            st.image(page['image_path'], width="stretch")
                        st.write(f"**Resolution:** {page['image_resolution']}")
                        st.write(f"**Format:** {page['image_format']}")
                        st.write(f"**Size:** {page['image_size_bytes'] / 1024:.2f} KB")
                    
                    with col2:
                        st.markdown("**GPT-5 Generated Description:**")
                        st.info(page['description'])
                    
                    if idx < len(doc_data['pages']) - 1:
                        st.divider()


def render_statistics():
    """Render processing statistics"""
    
    if st.session_state.all_documents:
        st.subheader("📊 Document Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", len(st.session_state.all_documents))
        
        with col2:
            text_chunks = sum(1 for doc in st.session_state.all_documents 
                            if doc.metadata.get('chunk_type') == 'text' or doc.metadata.get('content_type') != 'image')
            st.metric("Text Chunks", text_chunks)
        
        with col3:
            table_chunks = sum(1 for doc in st.session_state.all_documents 
                             if doc.metadata.get('chunk_type') == 'table')
            st.metric("Table Chunks", table_chunks)
        
        with col4:
            image_docs = sum(1 for doc in st.session_state.all_documents 
                           if doc.metadata.get('content_type') == 'image')
            st.metric("Image Documents", image_docs)
        
        # Document type distribution
        chunk_types = []
        for doc in st.session_state.all_documents:
            if doc.metadata.get('content_type') == 'image':
                chunk_types.append('image')
            else:
                chunk_types.append(doc.metadata.get('chunk_type', 'unknown'))
        
        type_df = pd.DataFrame({
            'Content Type': list(set(chunk_types)),
            'Count': [chunk_types.count(t) for t in set(chunk_types)]
        })
        
        st.bar_chart(type_df.set_index('Content Type'))


def render_retrieval_tab():
    """Render retrieval testing tab"""
    
    st.header("🔍 Retrieval Testing")
    
    if not st.session_state.vector_store_manager or not st.session_state.vector_store_manager.vector_store:
        st.warning("⚠️ Please upload and process PDFs first, or load a saved vector store.")
        return
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="Type your search query here..."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", width="stretch")
    
    if search_button and query:
        perform_search(query)
    
    # Display results
    if st.session_state.retrieval_results:
        render_results()


def perform_search(query: str):
    """Perform similarity search"""
    
    try:
        with st.spinner("🔍 Searching..."):
            # Perform search
            results = st.session_state.vector_store_manager.similarity_search(
                query=query,
                k=st.session_state.config.k,
                return_scores=True
            )
            
            # Evaluate results
            scored_results = st.session_state.evaluator.calculate_relevance_scores(results)
            diversity_metrics = st.session_state.evaluator.calculate_diversity(results)
            
            # Log query
            st.session_state.evaluator.log_query(
                query=query,
                results=results,
                config=st.session_state.config.to_dict()
            )
            
            # Store results
            st.session_state.retrieval_results = scored_results
            st.session_state.diversity_metrics = diversity_metrics
            st.session_state.current_query = query
            
            st.success(f"✅ Found {len(results)} results")
    
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.error(traceback.format_exc())


def render_results():
    """Render search results"""
    
    st.subheader(f"📊 Results for: '{st.session_state.current_query}'")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Results", len(st.session_state.retrieval_results))
    
    with col2:
        avg_score = sum(r['score'] for r in st.session_state.retrieval_results) / len(st.session_state.retrieval_results)
        st.metric("Avg Score", f"{avg_score:.4f}")
    
    with col3:
        st.metric("Unique Sources", st.session_state.diversity_metrics['unique_sources'])
    
    with col4:
        st.metric("Diversity", f"{st.session_state.diversity_metrics['diversity_score']:.2%}")
    
    # Chunk type distribution
    st.write("**Chunk Type Distribution:**")
    type_dist = st.session_state.diversity_metrics['chunk_type_distribution']
    col_types = st.columns(len(type_dist))
    for idx, (chunk_type, count) in enumerate(type_dist.items()):
        with col_types[idx]:
            st.metric(chunk_type.title(), count)
    
    st.divider()
    
    # Display individual results
    for result in st.session_state.retrieval_results:
        # Determine content type for display
        is_image = result['metadata'].get('content_type') == 'image'
        is_ppt = result['metadata'].get('source') == 'ppt'
        is_doc = result['metadata'].get('content_type') == 'doc_page'
        
        with st.expander(
            f"**Rank {result['rank']}** - {result['relevance_tier']} (Score: {result['score']:.4f})",
            expanded=(result['rank'] <= 3)
        ):
            # Metadata
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if is_image:
                    st.write(f"**Type:** 🖼️ Image")
                    st.write(f"**Image Name:** {result['metadata'].get('image_name', 'Unknown')}")
                    st.write(f"**Resolution:** {result['metadata'].get('image_resolution', 'N/A')}")
                    st.write(f"**Format:** {result['metadata'].get('image_format', 'N/A')}")
                elif is_doc:
                    st.write(f"**Type:** 📄 DOC/DOCX Page")
                    st.write(f"**Document Name:** {result['metadata'].get('doc_name', 'Unknown')}")
                    st.write(f"**Page:** {result['metadata'].get('page_number', 'N/A')} of {result['metadata'].get('total_pages', 'N/A')}")
                    st.write(f"**Resolution:** {result['metadata'].get('page_image_resolution', 'N/A')}")
                else:
                    st.write(f"**Source:** {result['metadata'].get('pdf_name', 'Unknown')}")
                    st.write(f"**Page:** {result['metadata'].get('source_page', 'N/A')}")
                    st.write(f"**Type:** {result['metadata'].get('chunk_type', 'text').title()}")
            
            with col2:
                if is_image:
                    st.write(f"**Size:** {result['metadata'].get('image_size_bytes', 0) / 1024:.2f} KB")
                elif is_doc:
                    st.write(f"**Size:** {result['metadata'].get('page_image_size_bytes', 0) / 1024:.2f} KB")
                else:
                    st.write(f"**Chunk Index:** {result['metadata'].get('chunk_index', 'N/A')}")
                st.write(f"**Normalized Score:** {result['normalized_score']:.4f}")
            
            # Display image if it's an image result
            if is_image:
                image_path = result['metadata'].get('image_path')
                if image_path and os.path.exists(image_path):
                    st.markdown("**Image Preview:**")
                    st.image(image_path, width="stretch")
            elif is_ppt:
                slide_image_path = result['metadata'].get('slide_image_path')
                if slide_image_path and os.path.exists(slide_image_path):
                    st.markdown("**Slide Preview:**")
                    st.image(slide_image_path, width="stretch")
                    st.caption(f"Slide {result['metadata'].get('slide_number', 'N/A')} from {result['metadata'].get('ppt_name', 'Unknown')}")
            elif is_doc:
                page_image_path = result['metadata'].get('page_image_path')
                if page_image_path and os.path.exists(page_image_path):
                    st.markdown("**Page Preview:**")
                    st.image(page_image_path, width="stretch")
                    st.caption(f"Page {result['metadata'].get('page_number', 'N/A')} from {result['metadata'].get('doc_name', 'Unknown')}")
            
            # Content / Description
            st.markdown("**Content/Description:**")
            st.text_area(
                "Content",
                value=result['content'],
                height=150,
                key=f"content_{result['rank']}",
                label_visibility="collapsed"
            )
            
            # Additional metadata for tables
            if result['metadata'].get('chunk_type') == 'table':
                if result['metadata'].get('table_structure'):
                    st.markdown("**Table Structure:**")
                    st.info(result['metadata']['table_structure'])
                
                if result['metadata'].get('table_context'):
                    st.markdown("**Table Context:**")
                    st.info(result['metadata']['table_context'])


def render_evaluation_tab():
    """Render evaluation and metrics tab"""
    
    st.header("📈 Evaluation & Metrics")
    
    if not st.session_state.evaluator.query_history:
        st.info("ℹ️ No queries executed yet. Perform some searches to see evaluation metrics.")
        return
    
    # Query statistics
    stats = st.session_state.evaluator.get_query_statistics()
    
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", stats['total_queries'])
    
    with col2:
        st.metric("Avg Results per Query", f"{stats['avg_results']:.1f}")
    
    with col3:
        st.metric("Avg Top Score", f"{stats['avg_top_score']:.4f}")
    
    st.divider()
    
    # Query history
    st.subheader("📜 Query History")
    
    history_df = pd.DataFrame(stats['queries'])
    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )
    
    # Export results
    st.divider()
    st.subheader("💾 Export Results")
    
    if st.button("📥 Download Evaluation Data (JSON)"):
        export_data = st.session_state.evaluator.export_results()
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def render_metadata_tab():
    """Render metadata exploration tab"""
    
    st.header("🗂️ Metadata Explorer")
    
    if not st.session_state.all_documents:
        st.info("ℹ️ No documents loaded. Please process PDFs or images first.")
        return
    
    # Create metadata DataFrame
    metadata_list = []
    for doc in st.session_state.all_documents:
        metadata_list.append(doc.metadata)
    
    df = pd.DataFrame(metadata_list)
    
    # Display options
    st.subheader("Filter and Explore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by content type
        content_types = []
        if 'content_type' in df.columns:
            content_types = df['content_type'].unique().tolist()
        if 'chunk_type' in df.columns:
            chunk_types = [ct for ct in df['chunk_type'].unique().tolist() if ct not in content_types]
            content_types.extend(chunk_types)
        if 'source' in df.columns:
            sources_col = df['source'].unique().tolist()
            for src in sources_col:
                if src == 'ppt' and 'ppt_slide' not in content_types:
                    content_types.append('ppt_slide')
                elif src == 'doc' and 'doc_page' not in content_types:
                    content_types.append('doc_page')
        
        if content_types:
            selected_types = st.multiselect(
                "Filter by Content Type",
                options=content_types,
                default=content_types
            )
        else:
            selected_types = []
    
    with col2:
        # Filter by source (PDF, Image, PPT, or DOC)
        sources = []
        if 'pdf_name' in df.columns:
            sources.extend(df['pdf_name'].dropna().unique().tolist())
        if 'image_name' in df.columns:
            sources.extend(df['image_name'].dropna().unique().tolist())
        if 'ppt_name' in df.columns:
            sources.extend(df['ppt_name'].dropna().unique().tolist())
        if 'doc_name' in df.columns:
            sources.extend(df['doc_name'].dropna().unique().tolist())
        
        if sources:
            selected_sources = st.multiselect(
                "Filter by Source",
                options=sources,
                default=sources
            )
        else:
            selected_sources = []
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_types:
        mask = pd.Series([False] * len(df))
        if 'content_type' in df.columns:
            mask |= df['content_type'].isin(selected_types)
        if 'chunk_type' in df.columns:
            mask |= df['chunk_type'].isin(selected_types)
        if 'source' in df.columns:
            if 'ppt_slide' in selected_types:
                mask |= df['source'] == 'ppt'
            if 'doc_page' in selected_types:
                mask |= df['source'] == 'doc'
        filtered_df = filtered_df[mask]
    
    if selected_sources:
        mask = pd.Series([False] * len(filtered_df))
        if 'pdf_name' in filtered_df.columns:
            mask |= filtered_df['pdf_name'].isin(selected_sources)
        if 'image_name' in filtered_df.columns:
            mask |= filtered_df['image_name'].isin(selected_sources)
        if 'ppt_name' in filtered_df.columns:
            mask |= filtered_df['ppt_name'].isin(selected_sources)
        if 'doc_name' in filtered_df.columns:
            mask |= filtered_df['doc_name'].isin(selected_sources)
        filtered_df = filtered_df[mask]
    
    st.write(f"**Showing {len(filtered_df)} of {len(df)} documents**")
    
    # Display dataframe
    st.dataframe(
        filtered_df,
        width="stretch",
        height=400
    )
    
    # Download metadata
    if st.button("📥 Download Metadata (CSV)"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Vision-Generated Descriptions Section
    st.divider()
    st.subheader("🔍 Vision-Generated Descriptions")
    
    # Filter documents to show only those with vision-generated descriptions
    vision_docs = [
        doc for doc in st.session_state.all_documents 
        if doc.metadata.get('content_type') in ['image', 'doc_page'] or doc.metadata.get('source') == 'ppt'
    ]
    
    if not vision_docs:
        st.info("ℹ️ No vision-generated descriptions available. Upload images, PPTs, or DOC files to see AI-generated descriptions.")
        return
    
    # Filter based on current filters
    filtered_vision_docs = []
    for doc in vision_docs:
        # Check content type filter
        if selected_types:
            doc_type = doc.metadata.get('content_type') or doc.metadata.get('chunk_type')
            if doc.metadata.get('source') == 'ppt' and 'ppt_slide' not in selected_types:
                continue
            if doc.metadata.get('source') == 'doc' and 'doc_page' not in selected_types:
                continue
            if doc_type and doc_type not in selected_types:
                continue
        
        # Check source filter
        if selected_sources:
            doc_source = (
                doc.metadata.get('pdf_name') or 
                doc.metadata.get('image_name') or 
                doc.metadata.get('ppt_name') or 
                doc.metadata.get('doc_name')
            )
            if doc_source not in selected_sources:
                continue
        
        filtered_vision_docs.append(doc)
    
    st.write(f"**{len(filtered_vision_docs)} vision-generated description(s) available**")
    
    # Display limit selector
    display_limit = st.selectbox(
        "Number of descriptions to display:",
        options=[10, 25, 50, 100, "All"],
        index=0
    )
    
    if display_limit == "All":
        display_limit = len(filtered_vision_docs)
    
    # Display vision descriptions in expandable cards
    for idx, doc in enumerate(filtered_vision_docs[:display_limit]):
        # Determine document type and create title
        metadata = doc.metadata
        
        if metadata.get('content_type') == 'image':
            title = f"🖼️ Image: {metadata.get('image_name', 'Unknown')}"
            subtitle = f"Resolution: {metadata.get('image_resolution', 'N/A')} | Format: {metadata.get('image_format', 'N/A')}"
            image_path = metadata.get('image_path')
        elif metadata.get('source') == 'ppt':
            title = f"📊 PPT Slide {metadata.get('slide_number', 'N/A')}: {metadata.get('ppt_name', 'Unknown')}"
            subtitle = f"Resolution: {metadata.get('slide_image_resolution', 'N/A')}"
            image_path = metadata.get('slide_image_path')
        elif metadata.get('content_type') == 'doc_page':
            title = f"📄 DOC Page {metadata.get('page_number', 'N/A')}: {metadata.get('doc_name', 'Unknown')}"
            subtitle = f"Page {metadata.get('page_number', 'N/A')} of {metadata.get('total_pages', 'N/A')} | Resolution: {metadata.get('page_image_resolution', 'N/A')}"
            image_path = metadata.get('page_image_path')
        else:
            title = f"📄 Document {idx + 1}"
            subtitle = "Unknown type"
            image_path = None
        
        with st.expander(f"{title}", expanded=False):
            st.caption(subtitle)
            
            # Display image preview if available
            if image_path and os.path.exists(image_path):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image_path, caption="Preview", width='stretch')
                
                with col2:
                    st.markdown("**GPT-5 Vision Description:**")
                    st.info(doc.page_content)
            else:
                st.markdown("**GPT-5 Vision Description:**")
                st.info(doc.page_content)
            
            # Show additional metadata
            with st.expander("📋 View Full Metadata"):
                st.json(metadata)


def render_rag_chat_tab():
    """Render RAG Chat tab — generation powered by retrieval context."""

    st.header("💬 RAG Chat — Ask Questions About Your Documents")

    if not st.session_state.vector_store_manager or not st.session_state.vector_store_manager.vector_store:
        st.warning("⚠️ Please upload and process documents first, or load a saved vector store.")
        return

    # ── Generation parameters (horizontal bar) ──
    st.subheader("⚙️ Generation Settings")
    param_col1, param_col2 = st.columns(2)

    with param_col1:
        rag_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Number of most-relevant chunks to feed as context to GPT-5",
            key="rag_k_slider"
        )

    with param_col2:
        show_sources = st.toggle(
            "Show Retrieved Sources",
            value=True,
            help="Display the retrieved chunks that were used as context",
            key="rag_show_sources"
        )

    st.divider()

    # ── Chat history display ──
    chat_container = st.container()
    with chat_container:
        for msg_idx, msg in enumerate(st.session_state.chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show sources for assistant messages
                if msg["role"] == "assistant" and msg.get("sources") and show_sources:
                    with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} chunks)", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            _render_source_card(i, src, msg_idx)

    # ── Chat input ──
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving relevant context & generating answer..."):
                sources, answer = _rag_generate(user_query, rag_k)

            st.markdown(answer)

            if sources and show_sources:
                with st.expander(f"📚 Retrieved Sources ({len(sources)} chunks)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        _render_source_card(i, src, len(st.session_state.chat_messages))

        # Store assistant message
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources if show_sources else []
        })

    # ── Clear chat button ──
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


def _rag_generate(query: str, k: int):
    """Retrieve relevant chunks and generate an answer with GPT-5.
    Includes retry logic for robustness.

    Returns:
        (sources_list, answer_text)
    """
    @retry_with_exponential_backoff(max_retries=3)
    def call_gpt5_with_retry(messages):
        return st.session_state.rag_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
        )
    
    # Step 1 — Retrieve
    try:
        results = st.session_state.vector_store_manager.similarity_search(
            query=query,
            k=k,
            return_scores=True
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
        return [], f"Retrieval error: {str(e)}"

    # Build context + source cards
    context_parts = []
    sources = []
    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata

        # Determine human-friendly source label
        source_label = (
            meta.get("pdf_name")
            or meta.get("image_name")
            or meta.get("ppt_name")
            or meta.get("doc_name")
            or "Unknown"
        )
        content_type = (
            meta.get("content_type")
            or meta.get("chunk_type")
            or "text"
        )

        context_parts.append(
            f"[Source {rank} | {source_label} | {content_type}]\n{doc.page_content}"
        )
        sources.append({
            "rank": rank,
            "score": float(score),
            "source": source_label,
            "content_type": content_type,
            "page": meta.get("source_page") or meta.get("page_number") or meta.get("slide_number"),
            "content": doc.page_content[:500],
            "metadata": meta
        })

    context_block = "\n\n---\n\n".join(context_parts)

    # Step 2 — Build messages
    system_prompt = (
        "You are an expert document assistant. Answer the user's question accurately "
        "based ONLY on the retrieved context below. "
        "If the context does not contain enough information, say so honestly. "
        "Cite the source numbers (e.g. [Source 1]) when you use information from a specific chunk.\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context_block}\n--- END CONTEXT ---"
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Include recent chat history for conversational continuity (last 10 turns)
    history_window = st.session_state.chat_messages[-10:]
    for msg in history_window:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Current user query (always last)
    messages.append({"role": "user", "content": query})

    # Step 3 — Call GPT-5
    try:
        response = st.session_state.rag_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"❌ Generation error: {e}"

    return sources, answer


def _render_source_card(rank: int, src: dict, msg_idx: int):
    """Render a single retrieved-source card inside an expander."""
    type_icons = {
        "text": "📄", "table": "📊", "image": "🖼️",
        "ppt_slide": "📊", "doc_page": "📝"
    }
    icon = type_icons.get(src["content_type"], "📄")
    page_info = f" | Page/Slide {src['page']}" if src.get("page") else ""

    st.markdown(
        f"**{icon} Source {rank}** — *{src['source']}*{page_info}  "
        f"  Score: `{src['score']:.4f}`"
    )
    st.text_area(
        f"Content (Source {rank})",
        value=src["content"],
        height=100,
        key=f"rag_src_msg{msg_idx}_rank{rank}_{hash(src['content'][:50])}",
        label_visibility="collapsed",
        disabled=True
    )

    # Show image/slide/page preview if available
    meta = src.get("metadata", {})
    preview_path = (
        meta.get("image_path")
        or meta.get("slide_image_path")
        or meta.get("page_image_path")
    )
    if preview_path and os.path.exists(preview_path):
        st.image(preview_path, width=300, caption=f"Source {rank} preview")


def main():
    """Main application"""
    
    # Title
    st.title("📚 Multi-Format Document Embedding & Retrieval System")
    st.markdown("*Powered by Azure OpenAI (text-embedding-3-large) and GPT-5-Saarathi*")
    st.markdown("*Supports: PDF, Images (PNG/JPG/etc), PowerPoint (PPT/PPTX), Word Documents (DOC/DOCX)*")
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tabs = st.tabs([
        " Upload Documents",
        " 💬 RAG Chat",
        " Retrieval",
        " Evaluation",
        " Metadata"
    ])
    
    with tabs[0]:
        render_unified_upload_tab()
    
    with tabs[1]:
        render_rag_chat_tab()
    
    with tabs[2]:
        render_retrieval_tab()
    
    with tabs[3]:
        render_evaluation_tab()
    
    with tabs[4]:
        render_metadata_tab()
    
    # Footer
    st.divider()
    st.caption("Multi-Format Document Retrieval System | Built with Streamlit, LangChain, FAISS, Spire.Presentation, docx2pdf, pdf2image, and GPT-5 Vision")


if __name__ == "__main__":
    main()
