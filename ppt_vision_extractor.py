"""
PowerPoint Vision Extractor - Convert slides to images and analyze with GPT-5 Vision
"""

import os
import base64
from io import BytesIO
from pathlib import Path
from pptx import Presentation
from PIL import Image
from openai import AzureOpenAI
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Vision API Configuration from .env
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")


def ppt_to_images(ppt_path, output_dir="ppt_images", dpi=300):
    """
    Convert PowerPoint slides to images.
    
    Args:
        ppt_path: Path to PPT/PPTX file
        output_dir: Directory to save slide images
        dpi: Resolution for image conversion
    
    Returns:
        List of image paths and metadata
    """
    print(f"\n{'='*80}")
    print(f"{'Converting PowerPoint to Images':^80}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load presentation
    prs = Presentation(ppt_path)
    
    # Use win32com for Windows
    try:
        import comtypes.client
        
        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        powerpoint.Visible = 1
        
        ppt_abs_path = os.path.abspath(ppt_path)
        deck = powerpoint.Presentations.Open(ppt_abs_path)
        
        slide_images = []
        
        for i, slide in enumerate(deck.Slides, start=1):
            image_path = os.path.join(output_dir, f"slide_{i}.png")
            
            # Export slide as image
            slide.Export(image_path, "PNG", 1920, 1080)
            
            slide_info = {
                'slide_number': i,
                'image_path': image_path,
                'width': 1920,
                'height': 1080
            }
            slide_images.append(slide_info)
            
            print(f"✅ Exported Slide {i} -> {image_path}")
        
        deck.Close()
        powerpoint.Quit()
        
        print(f"\n✅ Successfully converted {len(slide_images)} slides to images\n")
        return slide_images
    
    except Exception as e:
        print(f"❌ Error with COM automation: {e}")
        print("Falling back to manual screenshot method...")
        
        # Fallback: Manual method using pptx library metadata
        slide_images = []
        prs = Presentation(ppt_path)
        
        for i in range(len(prs.slides)):
            slide_info = {
                'slide_number': i + 1,
                'image_path': None,  # Will need manual screenshot
                'width': prs.slide_width,
                'height': prs.slide_height,
                'note': 'Manual screenshot required'
            }
            slide_images.append(slide_info)
        
        return slide_images


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_slide_description_with_vision(image_path, slide_number, api_key, endpoint, model, api_version):
    """
    Get detailed description of slide using GPT-5 Vision.
    
    Args:
        image_path: Path to slide image
        slide_number: Slide number
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint
        model: Model deployment name
        api_version: API version
    
    Returns:
        Dictionary with description and metadata
    """
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Create vision request with detailed prompt
    prompt = f"""Analyze this PowerPoint slide (Slide {slide_number}) and provide:

1. **Title/Heading**: Main title or heading of the slide
2. **Content Summary**: Brief summary of the slide content
3. **Key Points**: List all bullet points, text elements, and key information
4. **Visual Elements**: Describe any charts, graphs, images, diagrams, or visual elements
5. **Layout**: Describe the slide layout and structure
6. **Data/Numbers**: Extract any numerical data, statistics, or metrics shown
7. **Design Elements**: Note colors, themes, or branding elements

Please be detailed and comprehensive."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
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
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000
    )
    
    description = response.choices[0].message.content
    
    return {
        'slide_number': slide_number,
        'description': description,
        'model': model,
        'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
    }


def extract_ppt_metadata(ppt_path):
    """Extract PowerPoint metadata."""
    prs = Presentation(ppt_path)
    
    metadata = {
        'file_name': Path(ppt_path).name,
        'total_slides': len(prs.slides),
        'slide_width': prs.slide_width,
        'slide_height': prs.slide_height,
        'core_properties': {}
    }
    
    # Extract core properties
    core_props = prs.core_properties
    if core_props:
        metadata['core_properties'] = {
            'title': core_props.title,
            'author': core_props.author,
            'subject': core_props.subject,
            'created': str(core_props.created) if core_props.created else None,
            'modified': str(core_props.modified) if core_props.modified else None,
        }
    
    return metadata


def analyze_ppt_with_vision(ppt_path, output_dir="ppt_vision_analysis"):
    """
    Complete PowerPoint analysis using Vision API.
    
    Args:
        ppt_path: Path to PPT/PPTX file
        output_dir: Directory to save results
    
    Returns:
        Complete analysis results
    """
    print(f"\n{'='*80}")
    print(f"{'GPT-5 Vision PowerPoint Analysis':^80}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract metadata
    print("📋 Extracting PowerPoint metadata...")
    metadata = extract_ppt_metadata(ppt_path)
    print(f"✅ Found {metadata['total_slides']} slides\n")
    
    # Step 2: Convert slides to images
    print("🖼️  Converting slides to images...")
    images_dir = os.path.join(output_dir, "slide_images")
    slide_images = ppt_to_images(ppt_path, images_dir)
    
    # Step 3: Analyze each slide with Vision API
    print(f"\n{'='*80}")
    print("🔍 Analyzing slides with GPT-5 Vision...")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for slide_info in slide_images:
        if slide_info['image_path'] and os.path.exists(slide_info['image_path']):
            slide_num = slide_info['slide_number']
            print(f"Analyzing Slide {slide_num}...")
            
            # Get vision description
            vision_result = get_slide_description_with_vision(
                slide_info['image_path'],
                slide_num,
                VISION_API_KEY,
                VISION_ENDPOINT,
                VISION_MODEL,
                VISION_API_VERSION
            )
            
            # Combine with slide info
            result = {
                **slide_info,
                **vision_result
            }
            all_results.append(result)
            
            print(f"✅ Completed Slide {slide_num}\n")
    
    # Step 4: Save results
    print(f"{'='*80}")
    print("💾 Saving results...")
    print(f"{'='*80}\n")
    
    # Save JSON
    json_path = os.path.join(output_dir, "ppt_vision_analysis.json")
    complete_results = {
        'metadata': metadata,
        'slides': all_results,
        'total_analyzed': len(all_results)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved JSON analysis to: {json_path}")
    
    # Save readable text report
    txt_path = os.path.join(output_dir, "ppt_vision_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"PowerPoint Vision Analysis Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"File: {metadata['file_name']}\n")
        f.write(f"Total Slides: {metadata['total_slides']}\n")
        f.write(f"Model: {VISION_MODEL}\n\n")
        
        for result in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"SLIDE {result['slide_number']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(result['description'])
            f.write(f"\n\n")
    
    print(f"✅ Saved text report to: {txt_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    print(f"  • Total Slides Analyzed: {len(all_results)}")
    print(f"  • Output Directory: {output_dir}")
    print(f"  • Files Created:")
    print(f"    - {json_path}")
    print(f"    - {txt_path}")
    print(f"\n{'='*80}\n")
    
    return complete_results


# =====================================================
# USAGE EXAMPLE
# =====================================================

if __name__ == "__main__":
    # Example usage
    # ppt_path = "C:\\Users\\DQO3KOR\\Downloads\\your_presentation.pptx"
    # results = analyze_ppt_with_vision(ppt_path)
    
    # Access results
    # if results:
    #     print("\n🎯 Sample Output:")
    #     print("="*60)
    #     if results['slides']:
    #         first_slide = results['slides'][0]
    #         print(f"\nSlide 1 Description Preview:")
    #         print(first_slide['description'][:300] + "...")
    
    print("PowerPoint Vision Extractor loaded successfully!")
    print("Use: analyze_ppt_with_vision(ppt_path) to analyze a presentation")
