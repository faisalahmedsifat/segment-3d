#!/bin/bash

# Segment3D X11 Display Setup Script
# This script helps configure X11 display forwarding for running GUI applications in Docker

set -e

echo "🖥️  Setting up X11 display forwarding for Segment3D..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect the operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to setup X11 on Linux
setup_linux_x11() {
    echo "📋 Setting up X11 on Linux..."
    
    # Check if X11 is running
    if [ -z "$DISPLAY" ]; then
        echo "❌ DISPLAY variable is not set. Make sure you're running this in a graphical session."
        exit 1
    fi
    
    echo "✅ DISPLAY is set to: $DISPLAY"
    
    # Check if xauth exists
    if ! command_exists xauth; then
        echo "📦 Installing xauth..."
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y xauth
        elif command_exists yum; then
            sudo yum install -y xorg-x11-xauth
        elif command_exists dnf; then
            sudo dnf install -y xorg-x11-xauth
        elif command_exists pacman; then
            sudo pacman -S xorg-xauth
        else
            echo "❌ Could not install xauth. Please install it manually."
            exit 1
        fi
    fi
    
    # Allow local connections
    echo "🔓 Allowing local X11 connections..."
    xhost +local: 2>/dev/null || echo "⚠️  xhost command failed, but this might be OK"
    
    # Check if .Xauthority exists
    if [ ! -f "$HOME/.Xauthority" ]; then
        echo "🔐 Creating .Xauthority file..."
        touch "$HOME/.Xauthority"
        xauth generate "$DISPLAY" . trusted 2>/dev/null || echo "⚠️  Could not generate xauth entry"
    fi
    
    echo "✅ X11 setup complete for Linux!"
}

# Function to setup X11 on macOS
setup_macos_x11() {
    echo "📋 Setting up X11 on macOS..."
    
    # Check if XQuartz is installed
    if [ ! -d "/Applications/Utilities/XQuartz.app" ] && [ ! -d "/opt/X11" ]; then
        echo "❌ XQuartz is not installed. Please install it from:"
        echo "   https://www.xquartz.org/"
        echo "   or run: brew install --cask xquartz"
        exit 1
    fi
    
    echo "✅ XQuartz is installed"
    
    # Set DISPLAY if not set
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=host.docker.internal:0
        echo "📺 Set DISPLAY to: $DISPLAY"
    fi
    
    # Allow connections from localhost
    echo "🔓 Allowing X11 connections from localhost..."
    /opt/X11/bin/xhost +localhost 2>/dev/null || echo "⚠️  xhost command failed"
    
    echo "✅ X11 setup complete for macOS!"
    echo "ℹ️  Make sure XQuartz is running before starting Docker containers"
}

# Function to setup X11 on Windows
setup_windows_x11() {
    echo "📋 Setting up X11 on Windows..."
    echo "ℹ️  For Windows, you need to install an X11 server like:"
    echo "   - VcXsrv: https://sourceforge.net/projects/vcxsrv/"
    echo "   - Xming: https://sourceforge.net/projects/xming/"
    echo "   - MobaXterm: https://mobaxterm.mobatek.net/"
    echo ""
    echo "📝 After installing, configure the X11 server to:"
    echo "   - Allow connections from localhost"
    echo "   - Disable access control"
    echo "   - Set DISPLAY=host.docker.internal:0 in your environment"
    echo ""
    echo "🚨 Note: GUI applications in Docker on Windows require WSL2 backend"
}

# Main setup function
main() {
    echo "🚀 Segment3D X11 Setup Script"
    echo "================================"
    
    OS=$(detect_os)
    echo "🔍 Detected OS: $OS"
    
    case $OS in
        "linux")
            setup_linux_x11
            ;;
        "macos")
            setup_macos_x11
            ;;
        "windows")
            setup_windows_x11
            return 0
            ;;
        *)
            echo "❌ Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    echo ""
    echo "🎉 X11 setup completed!"
    echo ""
    echo "🔧 Next steps:"
    echo "   1. Run: docker compose up"
    echo "   2. The GUI should appear on your desktop"
    echo ""
    echo "🐛 Troubleshooting:"
    echo "   - If GUI doesn't appear, check Docker logs: docker compose logs segment3d"
    echo "   - Ensure your user has access to X11: groups \$USER"
    echo "   - Try running: xhost +local:docker"
    echo ""
    echo "📚 For more help, see: https://github.com/your-repo/segment-3d/blob/main/docs/DOCKER.md"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 