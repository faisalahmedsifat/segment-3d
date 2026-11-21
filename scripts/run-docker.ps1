# Segment3D Docker Setup Script for Windows
# PowerShell script to help Windows users run Segment3D with Docker

param(
    [string]$Action = "start",
    [switch]$Development,
    [switch]$Build,
    [switch]$Help
)

# Configuration
$ProjectName = "segment-3d"
$DockerComposeFile = "docker-compose.yml"
$DevComposeFile = "docker-compose.dev.yml"

function Show-Help {
    Write-Host @"
🐳 Segment3D Docker Setup Script for Windows

USAGE:
    .\scripts\run-docker.ps1 [ACTION] [OPTIONS]

ACTIONS:
    start       Start the Docker containers (default)
    stop        Stop the Docker containers
    restart     Restart the Docker containers
    build       Build the Docker images
    logs        Show container logs
    clean       Clean up containers and images
    setup       Initial setup and environment check

OPTIONS:
    -Development    Use development configuration
    -Build          Force rebuild images
    -Help           Show this help message

EXAMPLES:
    .\scripts\run-docker.ps1                    # Start production containers
    .\scripts\run-docker.ps1 -Development      # Start development environment
    .\scripts\run-docker.ps1 build -Build      # Force rebuild images
    .\scripts\run-docker.ps1 logs              # View container logs
    .\scripts\run-docker.ps1 clean             # Clean up everything

REQUIREMENTS:
    - Docker Desktop for Windows
    - X11 Server (VcXsrv, Xming, or MobaXterm)
    - WSL2 enabled

For more information, see DOCKER.md
"@
}

function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
        Write-Host "   Please install Docker Desktop from https://docker.com/products/docker-desktop" -ForegroundColor Red
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker compose version
        Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker Compose is not available" -ForegroundColor Red
        Write-Host "   Please update Docker Desktop to the latest version" -ForegroundColor Red
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Host "✅ Docker daemon is running" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker daemon is not running" -ForegroundColor Red
        Write-Host "   Please start Docker Desktop" -ForegroundColor Red
        exit 1
    }
    
    # Check for .env file
    if (Test-Path ".env") {
        Write-Host "✅ Environment file found" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  No .env file found" -ForegroundColor Yellow
        if (Test-Path ".env.example") {
            Write-Host "   Creating .env from .env.example..." -ForegroundColor Yellow
            Copy-Item ".env.example" ".env"
            Write-Host "   Please edit .env with your Modal credentials" -ForegroundColor Yellow
        }
        else {
            Write-Host "❌ No .env.example found" -ForegroundColor Red
            exit 1
        }
    }
}

function Show-X11Setup {
    Write-Host @"
🖥️  X11 Server Setup for Windows

To run the GUI application, you need an X11 server. Choose one:

1. VcXsrv (Recommended):
   - Download: https://sourceforge.net/projects/vcxsrv/
   - Install and run with these settings:
     ✓ Multiple windows
     ✓ Start no client
     ✓ Disable access control
     ✓ Native opengl

2. Xming:
   - Download: https://sourceforge.net/projects/xming/
   - Run: Xming :0 -multiwindow -clipboard -ac

3. MobaXterm:
   - Download: https://mobaxterm.mobatek.net/
   - X11 server starts automatically

After starting your X11 server, set DISPLAY in your .env file:
DISPLAY=host.docker.internal:0

"@ -ForegroundColor Cyan
}

function Start-Containers {
    Write-Host "🚀 Starting Segment3D containers..." -ForegroundColor Green
    
    $composeArgs = @("up", "-d")
    
    if ($Build) {
        $composeArgs += "--build"
    }
    
    if ($Development) {
        $composeArgs = @("-f", $DockerComposeFile, "-f", $DevComposeFile) + $composeArgs
        Write-Host "📝 Using development configuration" -ForegroundColor Yellow
    }
    
    & docker compose @composeArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Containers started successfully!" -ForegroundColor Green
        Show-ServiceUrls
    }
    else {
        Write-Host "❌ Failed to start containers" -ForegroundColor Red
        exit 1
    }
}

function Stop-Containers {
    Write-Host "⏹️  Stopping Segment3D containers..." -ForegroundColor Yellow
    
    $composeArgs = @("down")
    
    if ($Development) {
        $composeArgs = @("-f", $DockerComposeFile, "-f", $DevComposeFile) + $composeArgs
    }
    
    & docker compose @composeArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Containers stopped successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Failed to stop containers" -ForegroundColor Red
    }
}

function Build-Images {
    Write-Host "🏗️  Building Segment3D images..." -ForegroundColor Green
    
    $composeArgs = @("build")
    
    if ($Development) {
        $composeArgs = @("-f", $DockerComposeFile, "-f", $DevComposeFile) + $composeArgs
    }
    
    & docker compose @composeArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Images built successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Failed to build images" -ForegroundColor Red
        exit 1
    }
}

function Show-Logs {
    Write-Host "📋 Showing container logs..." -ForegroundColor Green
    
    $composeArgs = @("logs", "-f")
    
    if ($Development) {
        $composeArgs = @("-f", $DockerComposeFile, "-f", $DevComposeFile) + $composeArgs
    }
    
    & docker compose @composeArgs
}

function Clean-Up {
    Write-Host "🧹 Cleaning up Docker resources..." -ForegroundColor Yellow
    
    # Stop and remove containers
    & docker compose down -v --remove-orphans
    
    # Remove images
    $images = docker images "${ProjectName}*" -q
    if ($images) {
        Write-Host "Removing images..." -ForegroundColor Yellow
        & docker rmi $images -f
    }
    
    # Prune system
    & docker system prune -f
    
    Write-Host "✅ Cleanup completed!" -ForegroundColor Green
}

function Show-ServiceUrls {
    Write-Host @"

🌐 Service URLs:
   Main Application: GUI window on your desktop
   
"@ -ForegroundColor Cyan
    
    if ($Development) {
        Write-Host @"
   Development Services:
   📊 pgAdmin:         http://localhost:5050
   📊 Redis Commander: http://localhost:8081  
   📓 Jupyter Lab:     http://localhost:8888
   🔍 Database:        localhost:5432
   🔍 Redis:           localhost:6379

"@ -ForegroundColor Cyan
    }
}

function Setup-Environment {
    Write-Host "⚙️  Setting up Segment3D environment..." -ForegroundColor Green
    
    Test-Prerequisites
    Show-X11Setup
    
    Write-Host @"

📁 Directory Structure:
   ./data      → Input 3D models
   ./output    → Generated segmentations  
   ./temp      → Temporary files

🔑 Next Steps:
   1. Install and start an X11 server (see above)
   2. Edit .env file with your Modal credentials
   3. Run: .\scripts\run-docker.ps1 start

"@ -ForegroundColor Green
}

# Main script logic
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "🐳 Segment3D Docker Management Script" -ForegroundColor Blue
Write-Host "=====================================" -ForegroundColor Blue

switch ($Action.ToLower()) {
    "start" {
        Test-Prerequisites
        Start-Containers
    }
    "stop" {
        Stop-Containers
    }
    "restart" {
        Stop-Containers
        Start-Sleep -Seconds 2
        Start-Containers
    }
    "build" {
        Test-Prerequisites
        Build-Images
    }
    "logs" {
        Show-Logs
    }
    "clean" {
        Clean-Up
    }
    "setup" {
        Setup-Environment
    }
    default {
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Write-Host "Use -Help to see available actions" -ForegroundColor Yellow
        exit 1
    }
} 