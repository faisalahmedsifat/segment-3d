# 🚀 Segment3D Docker Quick Start Guide

**Your Docker setup is ready!** All dependencies are working correctly. You just need to set up GUI display forwarding to see the application window.

## ✅ What's Already Working

- ✅ **All Python dependencies** (numpy, opencv, vtk, scikit-learn, modal, PyQt5)
- ✅ **File system permissions** 
- ✅ **Database and Redis services**
- ✅ **Docker container networking**
- ✅ **Application can run in offscreen mode**

## 🎯 What You Need To Do

### Windows Users (Current Platform)

#### 1. Install X11 Server (Choose One)

**Option A: VcXsrv (Recommended)**
1. Download: https://sourceforge.net/projects/vcxsrv/
2. Install and run **XLaunch**
3. Configure with these settings:
   - ✅ **Multiple windows**
   - ✅ **Start no client** 
   - ✅ **Disable access control**
   - ✅ **Native opengl**

**Option B: Xming**
1. Download: https://sourceforge.net/projects/xming/
2. Run: `Xming :0 -multiwindow -clipboard -ac`

**Option C: MobaXterm**
1. Download: https://mobaxterm.mobatek.net/
2. X11 server starts automatically

#### 2. Start the Application

```powershell
# Option 1: Using our PowerShell script
.\scripts\run-docker.ps1 start

# Option 2: Using Docker Compose directly  
docker compose -f docker-compose.yml -f docker-compose.windows.yml up

# Option 3: Using Makefile
make run
```

### Linux Users

```bash
# 1. Setup X11 (automated)
./scripts/setup-x11.sh

# 2. Start application
make run
# or
docker compose up
```

### macOS Users

```bash
# 1. Install XQuartz
brew install --cask xquartz
# or download from https://www.xquartz.org/

# 2. Setup X11
./scripts/setup-x11.sh

# 3. Start application
make run
```

## 🔧 Available Commands

```bash
# First-time setup and start
make quick-start

# Development environment with tools
make quick-dev

# Production deployment  
make prod

# View logs
make logs

# Stop application
make stop

# Clean up
make clean

# Get help
make help
```

## 🧪 Test Results Summary

**✅ Container Test Results:**
- All Python imports: **PASSED**
- File system permissions: **PASSED** 
- Environment variables: **PASSED**
- Modal client import: **PASSED**
- Qt offscreen mode: **PASSED**

**❌ GUI Display:**
- X11 connection: **NEEDS SETUP** (install X11 server)

## 🎯 Expected Result

After setting up X11 server, you should see:

1. **Container starts successfully** ✅
2. **All tests pass** ✅  
3. **GUI window appears** (after X11 setup)
4. **File browser opens** for selecting STL files
5. **3D model viewer loads** when STL file is selected

## 🐛 Troubleshooting

### GUI Doesn't Appear
```bash
# Check if X11 server is running
# Windows: Check Task Manager for vcxsrv.exe, Xming.exe, or MobaXterm.exe
# Linux: echo $DISPLAY  
# macOS: Check if XQuartz is running

# Test X11 connection
docker run --rm -e DISPLAY=host.docker.internal:0 --add-host=host.docker.internal:host-gateway alpine:latest echo "Connection test"
```

### Container Build Issues
```bash
# Clean rebuild
make clean
make build-no-cache
```

### Performance Issues  
```bash
# Check Docker Desktop memory allocation (recommended: 4GB+)
# Windows: Docker Desktop → Settings → Resources → Advanced
```

## 📊 Service URLs (Development Mode)

When running development environment (`make dev`):

- **Main Application**: GUI window on desktop
- **pgAdmin**: http://localhost:5050 (admin@segment3d.local / admin)
- **Redis Commander**: http://localhost:8081
- **Jupyter Lab**: http://localhost:8888 (token: segment3d)
- **Database**: localhost:5432 (segment3d / segment3d_password)

## 🔑 Modal AI Setup (Optional)

To use the AI segmentation features:

1. Sign up at https://modal.com
2. Get your API credentials
3. Add to `.env` file:
   ```
   MODAL_TOKEN_ID=your_token_id_here
   MODAL_TOKEN_SECRET=your_token_secret_here
   ```

## 🎉 Success Indicators

You'll know everything is working when:

1. **No container errors** in logs
2. **GUI window appears** on your desktop  
3. **Can browse and select STL files**
4. **3D model renders** in the viewer
5. **Right-click triggers** AI segmentation (with Modal setup)

## 💡 Pro Tips

- **First run**: Use `make quick-start` for complete setup
- **Development**: Use `make dev` for additional tools
- **Debugging**: Use `make logs` to see what's happening
- **Performance**: Allocate 4GB+ RAM to Docker Desktop
- **Updates**: Use `make build` to rebuild after code changes

---

**Need more help?** See `DOCKER.md` for comprehensive documentation or run `make docs` for detailed usage information.

**Ready to start?** Run `make quick-start` and enjoy your containerized 3D segmentation tool! 🎉 