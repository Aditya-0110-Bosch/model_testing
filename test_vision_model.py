"""
Test script to verify GPT-5 Vision model connectivity and functionality
"""

import os
import sys
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# GPT-5 Vision credentials
VISION_API_KEY = os.getenv("VISION_API_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-Saarathi")
VISION_API_VERSION = os.getenv("VISION_API_VERSION", "2025-01-01-preview")

def create_test_image():
    """Create a simple test image with text"""
    # Create a 400x300 image with white background
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes and text
    draw.rectangle([50, 50, 350, 100], fill='blue', outline='black', width=2)
    draw.ellipse([100, 150, 300, 250], fill='red', outline='black', width=2)
    draw.text((120, 60), "Test Image", fill='white')
    draw.text((150, 180), "VISION TEST", fill='white')
    
    return img

def encode_image_to_base64(image):
    """Encode PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_vision_api():
    """Test the GPT-5 Vision API"""
    
    print("=" * 70)
    print("GPT-5 VISION MODEL TEST")
    print("=" * 70)
    print()
    
    # Step 1: Check environment variables
    print("Step 1: Checking environment variables...")
    if not VISION_API_KEY:
        print("❌ VISION_API_KEY not found in environment")
        return False
    if not VISION_ENDPOINT:
        print("❌ VISION_ENDPOINT not found in environment")
        return False
    
    print(f"✅ VISION_API_KEY: {'*' * 10}{VISION_API_KEY[-4:]}")
    print(f"✅ VISION_ENDPOINT: {VISION_ENDPOINT}")
    print(f"✅ VISION_MODEL: {VISION_MODEL}")
    print(f"✅ VISION_API_VERSION: {VISION_API_VERSION}")
    print()
    
    # Step 2: Initialize client
    print("Step 2: Initializing Azure OpenAI client...")
    try:
        client = AzureOpenAI(
            api_key=VISION_API_KEY,
            api_version=VISION_API_VERSION,
            azure_endpoint=VISION_ENDPOINT
        )
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return False
    print()
    
    # Step 3: Create test image
    print("Step 3: Creating test image...")
    try:
        test_image = create_test_image()
        print("✅ Test image created (400x300 with shapes and text)")
    except Exception as e:
        print(f"❌ Image creation failed: {e}")
        return False
    print()
    
    # Step 4: Encode image to base64
    print("Step 4: Encoding image to base64...")
    try:
        base64_image = encode_image_to_base64(test_image)
        print(f"✅ Image encoded ({len(base64_image)} bytes)")
    except Exception as e:
        print(f"❌ Image encoding failed: {e}")
        return False
    print()
    
    # Step 5: Test simple text completion (without vision)
    print("Step 5: Testing basic text completion...")
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": "Say 'Hello, Vision Model Test!' if you can read this."}
            ],
            max_tokens=50
        )
        answer = response.choices[0].message.content.strip()
        print(f"✅ Text completion successful")
        print(f"   Response: {answer}")
    except Exception as e:
        print(f"❌ Text completion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Step 6: Test vision capabilities with test image
    print("Step 6: Testing vision capabilities with test image...")
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this image. What shapes, colors, and text are visible?"
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
            max_tokens=200
        )
        
        description = response.choices[0].message.content.strip()
        print(f"✅ Vision test successful!")
        print(f"   Model description:")
        print(f"   {description}")
        
        # Check usage stats if available
        if hasattr(response, 'usage'):
            print(f"\n   Token usage:")
            print(f"   - Prompt tokens: {response.usage.prompt_tokens}")
            print(f"   - Completion tokens: {response.usage.completion_tokens}")
            print(f"   - Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ Vision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Summary
    print("=" * 70)
    print("✅ ALL TESTS PASSED - GPT-5 Vision Model is working correctly!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_vision_api()
    sys.exit(0 if success else 1)
