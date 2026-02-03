"""
Streamlit PDF Embedding and Retrieval Experimentation Interface
"""

import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any

from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager, RetrievalConfig
from evaluation import RetrievalEvaluator
from image_processor import ImageProcessor, save_uploaded_image, supported_image_formats

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
if 'vector_store_manager' not in st.session_state:
    st.session_state.vector_store_manager = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = RetrievalEvaluator()
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'pdf_documents' not in st.session_state:
    st.session_state.pdf_documents = []
if 'image_documents' not in st.session_state:
    st.session_state.image_documents = []
if 'retrieval_results' not in st.session_state:
    st.session_state.retrieval_results = []


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
            value=4,
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
        
        # Save/Load Vector Store
        st.header("Vector Store Persistence")
        
        store_type = st.radio(
            "Store Type",
            options=["Combined (PDFs + Images)", "PDFs Only", "Images Only"],
            index=0,
            help="Choose which documents to save/load"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Store", width="stretch"):
                save_vector_store(store_type)
        
        with col2:
            if st.button("📂 Load Store", width="stretch"):
                load_vector_store(store_type)


def save_vector_store(store_type: str):
    """Save vector store to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if store_type == "Combined (PDFs + Images)":
            if st.session_state.vector_store_manager and st.session_state.vector_store_manager.vector_store:
                save_path = f"vector_stores/combined_{timestamp}"
                st.session_state.vector_store_manager.save_vector_store(save_path)
                st.success(f"✅ Combined vector store saved to {save_path}")
                st.info(f"📊 Saved {len(st.session_state.all_documents)} documents (PDFs + Images)")
            else:
                st.warning("⚠️ No combined vector store to save. Please process documents first.")
        
        elif store_type == "PDFs Only":
            if st.session_state.pdf_documents:
                save_path = f"vector_stores/pdf_only_{timestamp}"
                # Create temporary vector store for PDFs only
                temp_manager = VectorStoreManager(
                    max_workers=st.session_state.max_workers,
                    index_type=st.session_state.config.index_type
                )
                temp_manager.create_vector_store(st.session_state.pdf_documents)
                temp_manager.save_vector_store(save_path)
                st.success(f"✅ PDF-only vector store saved to {save_path}")
                st.info(f"📄 Saved {len(st.session_state.pdf_documents)} PDF documents")
            else:
                st.warning("⚠️ No PDF documents to save. Please process PDFs first.")
        
        elif store_type == "Images Only":
            if st.session_state.image_documents:
                save_path = f"vector_stores/image_only_{timestamp}"
                # Create temporary vector store for images only
                temp_manager = VectorStoreManager(
                    max_workers=st.session_state.max_workers,
                    index_type=st.session_state.config.index_type
                )
                temp_manager.create_vector_store(st.session_state.image_documents)
                temp_manager.save_vector_store(save_path)
                st.success(f"✅ Image-only vector store saved to {save_path}")
                st.info(f"🖼️ Saved {len(st.session_state.image_documents)} image documents")
            else:
                st.warning("⚠️ No image documents to save. Please process images first.")
    
    except Exception as e:
        st.error(f"❌ Error saving vector store: {e}")
        import traceback
        st.error(traceback.format_exc())


def load_vector_store(store_type: str):
    """Load vector store from disk"""
    
    # List available stores
    if os.path.exists("vector_stores"):
        all_stores = [d for d in os.listdir("vector_stores") if os.path.isdir(os.path.join("vector_stores", d))]
        
        # Filter stores based on type
        if store_type == "Combined (PDFs + Images)":
            stores = [s for s in all_stores if s.startswith("combined_") or s.startswith("store_")]
        elif store_type == "PDFs Only":
            stores = [s for s in all_stores if s.startswith("pdf_only_")]
        elif store_type == "Images Only":
            stores = [s for s in all_stores if s.startswith("image_only_")]
        
        if stores:
            selected_store = st.sidebar.selectbox(f"Select {store_type} Store", stores, key=f"load_{store_type}")
            
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
                
                # Provide info about what was loaded
                if "combined" in selected_store or "store" in selected_store:
                    st.info("📊 Loaded combined PDFs + Images store")
                elif "pdf_only" in selected_store:
                    st.info("📄 Loaded PDF-only store")
                elif "image_only" in selected_store:
                    st.info("🖼️ Loaded image-only store")
                    
            except Exception as e:
                st.error(f"❌ Error loading vector store: {e}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.warning(f"⚠️ No {store_type} vector stores found.")
    else:
        st.warning("⚠️ No vector_stores directory found.")


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
        st.error(f"❌ Error processing PDFs: {e}")
        import traceback
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
        st.error(f"❌ Error processing images: {e}")
        import traceback
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
        st.error(f"❌ Search error: {e}")
        import traceback
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
                else:
                    st.write(f"**Source:** {result['metadata'].get('pdf_name', 'Unknown')}")
                    st.write(f"**Page:** {result['metadata'].get('source_page', 'N/A')}")
                    st.write(f"**Type:** {result['metadata'].get('chunk_type', 'text').title()}")
            
            with col2:
                if is_image:
                    st.write(f"**Size:** {result['metadata'].get('image_size_bytes', 0) / 1024:.2f} KB")
                else:
                    st.write(f"**Chunk Index:** {result['metadata'].get('chunk_index', 'N/A')}")
                st.write(f"**Normalized Score:** {result['normalized_score']:.4f}")
            
            # Display image if it's an image result
            if is_image:
                image_path = result['metadata'].get('image_path')
                if image_path and os.path.exists(image_path):
                    st.markdown("**Image Preview:**")
                    st.image(image_path, width="stretch")
            
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
        
        if content_types:
            selected_types = st.multiselect(
                "Filter by Content Type",
                options=content_types,
                default=content_types
            )
        else:
            selected_types = []
    
    with col2:
        # Filter by source (PDF or Image)
        sources = []
        if 'pdf_name' in df.columns:
            sources.extend(df['pdf_name'].dropna().unique().tolist())
        if 'image_name' in df.columns:
            sources.extend(df['image_name'].dropna().unique().tolist())
        
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
        filtered_df = filtered_df[mask]
    
    if selected_sources:
        mask = pd.Series([False] * len(filtered_df))
        if 'pdf_name' in filtered_df.columns:
            mask |= filtered_df['pdf_name'].isin(selected_sources)
        if 'image_name' in filtered_df.columns:
            mask |= filtered_df['image_name'].isin(selected_sources)
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


def main():
    """Main application"""
    
    # Title
    st.title("📚 PDF & Image Embedding & Retrieval Experimentation")
    st.markdown("*Powered by Azure OpenAI (text-embedding-3-large) and GPT-5-Saarathi*")
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tabs = st.tabs(["📤 Upload PDFs", "🖼️ Upload Images", "🔍 Retrieval", "📈 Evaluation", "🗂️ Metadata"])
    
    with tabs[0]:
        render_upload_tab()
    
    with tabs[1]:
        render_image_upload_tab()
    
    with tabs[2]:
        render_retrieval_tab()
    
    with tabs[3]:
        render_evaluation_tab()
    
    with tabs[4]:
        render_metadata_tab()
    
    # Footer
    st.divider()
    st.caption("PDF & Image Retrieval Experimentation Interface | Built with Streamlit, LangChain, FAISS, and GPT-5 Vision")


if __name__ == "__main__":
    main()
