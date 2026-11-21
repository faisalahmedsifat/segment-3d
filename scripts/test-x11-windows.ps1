# Test X11 Setup for Windows
# This script helps test and set up X11 display forwarding for Windows

Write-Host "🖥️  Testing X11 Setup for Segment3D on Windows..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Function to test if a port is listening
function Test-Port {
    param([string]$Host, [int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient($Host, $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if process is running
function Test-Process {
    param([string]$ProcessName)
    return (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue) -ne $null
}

Write-Host "`n🔍 Checking X11 Server Status..." -ForegroundColor Yellow

# Check for common X11 servers on Windows
$x11Servers = @(
    @{Name="VcXsrv"; ProcessName="vcxsrv"; Port=6000},
    @{Name="Xming"; ProcessName="Xming"; Port=6000},
    @{Name="MobaXterm"; ProcessName="MobaXterm"; Port=6000}
)

$x11Running = $false
foreach ($server in $x11Servers) {
    if (Test-Process -ProcessName $server.ProcessName) {
        Write-Host "✅ $($server.Name) is running" -ForegroundColor Green
        $x11Running = $true
        
        # Test if X11 port is listening
        if (Test-Port -Host "localhost" -Port $server.Port) {
            Write-Host "✅ X11 server is listening on port $($server.Port)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  X11 server process found but port $($server.Port) not accessible" -ForegroundColor Yellow
        }
    }
}

if (-not $x11Running) {
    Write-Host "`n❌ No X11 server detected. You need to install and start one:" -ForegroundColor Red
    Write-Host "`n📥 Installation Options:" -ForegroundColor Cyan
    Write-Host "1. VcXsrv (Recommended):" -ForegroundColor White
    Write-Host "   Download: https://sourceforge.net/projects/vcxsrv/" -ForegroundColor Gray
    Write-Host "   Setup: Run XLaunch with these settings:" -ForegroundColor Gray
    Write-Host "   ✓ Multiple windows" -ForegroundColor Gray
    Write-Host "   ✓ Start no client" -ForegroundColor Gray
    Write-Host "   ✓ Disable access control" -ForegroundColor Gray
    Write-Host "   ✓ Native opengl" -ForegroundColor Gray
    
    Write-Host "`n2. Xming:" -ForegroundColor White
    Write-Host "   Download: https://sourceforge.net/projects/xming/" -ForegroundColor Gray
    Write-Host "   Run: Xming :0 -multiwindow -clipboard -ac" -ForegroundColor Gray
    
    Write-Host "`n3. MobaXterm:" -ForegroundColor White
    Write-Host "   Download: https://mobaxterm.mobatek.net/" -ForegroundColor Gray
    Write-Host "   X11 server starts automatically" -ForegroundColor Gray
    
    Write-Host "`n🔧 After installing, restart this script to test the connection." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n🐳 Testing Docker Setup..." -ForegroundColor Yellow

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found or not running" -ForegroundColor Red
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker compose version
    Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose not available" -ForegroundColor Red
    exit 1
}

Write-Host "`n🧪 Testing Docker X11 Connection..." -ForegroundColor Yellow

# Test X11 connection with a simple container
Write-Host "Running X11 test container..." -ForegroundColor Gray
$testResult = docker run --rm -e DISPLAY=host.docker.internal:0 `
    --add-host=host.docker.internal:host-gateway `
    alpine:latest sh -c "echo 'Docker can resolve host.docker.internal'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker can connect to host" -ForegroundColor Green
} else {
    Write-Host "❌ Docker host connection failed" -ForegroundColor Red
}

Write-Host "`n📋 Environment Check..." -ForegroundColor Yellow

# Check for required files
$requiredFiles = @("Dockerfile", "docker-compose.yml", "docker-compose.windows.yml")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
    }
}

# Check for data directories
$requiredDirs = @("data", "output", "temp")
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "✅ Directory exists: $dir" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Creating directory: $dir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

Write-Host "`n🎯 Ready to Launch!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Your system appears ready to run Segment3D." -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "1. Make sure your X11 server is running" -ForegroundColor Gray
Write-Host "2. Set MODAL credentials in .env file (if using Modal AI)" -ForegroundColor Gray
Write-Host "3. Run: docker compose -f docker-compose.yml -f docker-compose.windows.yml up --build" -ForegroundColor Gray

Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Cyan
Write-Host "- If GUI doesn't appear: Check X11 server settings" -ForegroundColor Gray
Write-Host "- If build fails: Check Docker Desktop memory allocation (recommend 4GB+)" -ForegroundColor Gray
Write-Host "- If connection errors: Restart Docker Desktop" -ForegroundColor Gray

Write-Host "`n✨ Test completed!" -ForegroundColor Green 