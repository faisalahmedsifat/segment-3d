#!/usr/bin/env python3
"""
Test script for Modal SAM integration
Run this before testing the full PyQt5 application
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_modal_connection():
    """Test if we can connect to Modal deployment"""
    print("=" * 50)
    print("Testing Modal Connection")
    print("=" * 50)
    
    try:
        from modal_client import get_modal_sam_client
        
        client = get_modal_sam_client()
        print("✓ Modal client created successfully")
        
        # Test health check
        if client.health_check():
            print("✓ Modal deployment is healthy")
            return True
        else:
            print("✗ Modal deployment health check failed")
            return False
            
    except Exception as e:
        print(f"✗ Failed to connect to Modal: {e}")
        print("\nTroubleshooting:")
        print("1. Install Modal: pip install modal")
        print("2. Setup Modal: python -m modal setup")
        print("3. Deploy SAM: modal deploy sam_modal.py")
        return False

def create_test_image():
    """Create a simple test image for segmentation"""
    print("\n" + "=" * 50)
    print("Creating Test Image")
    print("=" * 50)
    
    # Create a simple test image with some shapes
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # White background
    img.fill(255)
    
    # Add some colored shapes
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (400, 150), 80, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(img, (300, 250), (550, 350), (0, 0, 255), -1)  # Red rectangle
    
    # Save test image
    test_image_path = "test_image.jpg"
    cv2.imwrite(test_image_path, img)
    print(f"✓ Test image created: {test_image_path}")
    
    return test_image_path

def test_modal_segmentation():
    """Test Modal SAM segmentation with a simple test case"""
    print("\n" + "=" * 50)
    print("Testing Modal SAM Segmentation")
    print("=" * 50)
    
    try:
        from modal_client import get_modal_sam_client
        
        # Create test image
        test_image_path = create_test_image()
        
        # Initialize client
        client = get_modal_sam_client()
        
        # Test segmentation with click on the blue rectangle
        click_pos = np.array([[125, 125]])  # Click in the middle of blue rectangle
        labels = np.array([1])  # Positive click
        
        print(f"Testing segmentation with click at {click_pos[0]}")
        print("Sending request to Modal...")
        
        # Perform segmentation
        masks, scores, logits = client.predict(test_image_path, click_pos, labels)
        
        print(f"✓ Segmentation successful!")
        print(f"  - Received {len(masks)} masks")
        print(f"  - Scores: {scores}")
        print(f"  - Best mask shape: {masks[0].shape}")
        print(f"  - Best score: {max(scores):.4f}")
        
        # Save result for visual inspection
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Save mask as image
        mask_img = (best_mask * 255).astype(np.uint8)
        cv2.imwrite("test_result_mask.jpg", mask_img)
        print("✓ Result mask saved as: test_result_mask.jpg")
        
        # Create overlay visualization
        original_img = cv2.imread(test_image_path)
        overlay = original_img.copy()
        overlay[best_mask > 0] = [0, 255, 255]  # Yellow overlay
        
        # Blend original and overlay
        result = cv2.addWeighted(original_img, 0.7, overlay, 0.3, 0)
        cv2.imwrite("test_result_overlay.jpg", result)
        print("✓ Result overlay saved as: test_result_overlay.jpg")
        
        return True
        
    except Exception as e:
        print(f"✗ Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "test_image.jpg",
        "test_result_mask.jpg", 
        "test_result_overlay.jpg"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def main():
    """Run all tests"""
    print("Modal SAM Integration Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Modal connection
    if not test_modal_connection():
        success = False
        print("\n❌ Modal connection test FAILED")
        print("Please fix Modal deployment before proceeding")
        return
    
    # Test 2: Segmentation functionality
    if test_modal_segmentation():
        print("\n✅ All tests PASSED!")
        print("\nYou can now run your PyQt5 application:")
        print("python src/main.py")
        print("\nInspect the generated test images to verify segmentation quality.")
    else:
        success = False
        print("\n❌ Segmentation test FAILED")
    
    # Cleanup
    print("\n" + "=" * 50)
    print("Cleanup")
    print("=" * 50)
    cleanup_test_files()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 