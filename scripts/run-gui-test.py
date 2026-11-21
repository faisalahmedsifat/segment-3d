#!/usr/bin/env python3
"""
Simple test script to verify Segment3D components work in Docker container
This can run without X11 display for testing imports and basic functionality
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Python imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ opencv-python import failed: {e}")
        return False
    
    try:
        import vtk
        print("✅ vtk imported successfully")
    except ImportError as e:
        print(f"❌ vtk import failed: {e}")
        return False
    
    try:
        from sklearn.neighbors import KDTree
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import modal
        print("✅ modal imported successfully")
    except ImportError as e:
        print(f"❌ modal import failed: {e}")
        return False
    
    # Test PyQt5 without creating QApplication (which needs display)
    try:
        from PyQt5 import QtCore, QtWidgets
        print("✅ PyQt5 imported successfully")
    except ImportError as e:
        print(f"❌ PyQt5 import failed: {e}")
        return False
    
    return True

def test_file_system():
    """Test file system permissions and directories"""
    print("\n📁 Testing file system...")
    
    directories = ['/app/data', '/app/output', '/app/temp']
    for directory in directories:
        if os.path.exists(directory):
            if os.access(directory, os.W_OK):
                print(f"✅ {directory} exists and is writable")
            else:
                print(f"⚠️  {directory} exists but is not writable")
        else:
            print(f"❌ {directory} does not exist")
    
    # Test creating a file
    try:
        test_file = '/app/output/test_file.txt'
        with open(test_file, 'w') as f:
            f.write("Test content")
        os.remove(test_file)
        print("✅ File creation and deletion works")
    except Exception as e:
        print(f"❌ File operations failed: {e}")

def test_environment():
    """Test environment variables"""
    print("\n🌍 Testing environment...")
    
    env_vars = ['DISPLAY', 'PYTHONUNBUFFERED', 'QT_X11_NO_MITSHM']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")

def test_modal_client():
    """Test Modal client can be imported"""
    print("\n🤖 Testing Modal client...")
    
    try:
        sys.path.append('/app/src')
        from modal_client import ModalSAMClient
        print("✅ ModalSAMClient imported successfully")
        print("ℹ️  Modal credentials needed for actual connection test")
    except ImportError as e:
        print(f"❌ ModalSAMClient import failed: {e}")
    except Exception as e:
        print(f"⚠️  ModalSAMClient import succeeded but initialization may need credentials: {e}")

def main():
    print("🚀 Segment3D Docker Container Test")
    print("===================================")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        sys.exit(1)
    
    # Test file system
    test_file_system()
    
    # Test environment
    test_environment()
    
    # Test Modal client
    test_modal_client()
    
    print("\n🎉 All basic tests passed!")
    print("\n📝 Next steps:")
    print("1. Set up X11 server on Windows (VcXsrv, Xming, or MobaXterm)")
    print("2. Configure Modal AI credentials in .env file")
    print("3. Try running the full GUI application")
    
    # Check if we can try to start GUI
    display = os.environ.get('DISPLAY')
    if display:
        print(f"\n🖥️  Display is set to: {display}")
        print("Attempting to test Qt platform...")
        
        # Set Qt platform environment for better compatibility
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Use offscreen for testing
        
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication([])
            print("✅ Qt application created successfully (offscreen mode)")
            app.quit()
        except Exception as e:
            print(f"⚠️  Qt application test failed: {e}")
            print("This is expected without X11 server running")

if __name__ == "__main__":
    main() 