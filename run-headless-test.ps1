# Quick headless test of Segment3D Docker container
Write-Host "🧪 Testing Segment3D in headless mode..." -ForegroundColor Cyan

# Set environment for headless testing
$env:QT_QPA_PLATFORM = "offscreen"
$env:DISPLAY = ""

Write-Host "Running comprehensive container tests..." -ForegroundColor Yellow

# Run our test script
docker compose run --rm segment3d python /app/scripts/run-gui-test.py

Write-Host "`n✅ Headless test completed!" -ForegroundColor Green
Write-Host "💡 Install VcXsrv for full GUI experience:" -ForegroundColor Cyan  
Write-Host "   1. Download: https://sourceforge.net/projects/vcxsrv/" -ForegroundColor White
Write-Host "   2. Configure with: scripts/vcxsrv-config.md" -ForegroundColor White
Write-Host "   3. Run: docker compose -f docker-compose.yml -f docker-compose.windows.yml up" -ForegroundColor White 