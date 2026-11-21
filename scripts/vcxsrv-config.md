# VcXsrv Configuration Guide for Segment3D

## 🖥️ Step-by-Step VcXsrv Setup

### 1. Launch XLaunch
- Press **Windows Key + R**
- Type: `xlaunch`
- Press **Enter**

### 2. Configuration Screens

#### Screen 1: Select Display Settings
```
✅ Multiple windows
❌ One large window  
❌ One window without titlebar
❌ Fullscreen

Click "Next >"
```

#### Screen 2: Select How to Start Clients
```
✅ Start no client
❌ Start a program
❌ Open session via XDMCP

Click "Next >"
```

#### Screen 3: Extra Settings
```
✅ Clipboard (checked)
✅ Primary Selection (checked)  
✅ Native opengl (checked)
✅ Disable access control (IMPORTANT!)
❌ Additional parameters for VcXsrv

Click "Next >"
```

#### Screen 4: Finish Configuration
```
✅ Save configuration (optional - saves as config.xlaunch)

Click "Finish"
```

### 3. Verify VcXsrv is Running
- Check system tray for VcXsrv icon (X icon)
- If you see the X icon, VcXsrv is running correctly

### 4. Test Connection
Run this command to test X11 connection:
```bash
docker run --rm -e DISPLAY=host.docker.internal:0 --add-host=host.docker.internal:host-gateway alpine:latest echo "X11 test connection"
```

### 5. Windows Firewall (If Needed)
If Windows Firewall blocks VcXsrv:
1. Go to **Windows Security** → **Firewall & network protection**
2. Click **"Allow an app through firewall"**
3. Find **VcXsrv** and check both **Private** and **Public** networks
4. Click **OK**

## 🚀 Ready to Run Segment3D!

Once VcXsrv is configured and running, start the application:

```powershell
# Start Segment3D with GUI
docker compose -f docker-compose.yml -f docker-compose.windows.yml up
```

## 🐛 Troubleshooting

### VcXsrv Icon Missing from System Tray
- VcXsrv might not be running
- Launch XLaunch again and run through configuration

### Docker Container Starts But No GUI
- Check that VcXsrv is running (X icon in system tray)
- Verify "Disable access control" was checked in VcXsrv config
- Try restarting VcXsrv

### GUI Appears But Is Corrupted
- Enable "Native opengl" in VcXsrv configuration
- Restart VcXsrv after changing settings

### Connection Refused Errors
- Check Windows Firewall settings
- Ensure Docker Desktop is running with WSL2 backend
- Verify host.docker.internal resolves correctly

## 💡 Pro Tips

- **Save Configuration**: Check "Save configuration" in final step to avoid reconfiguring each time
- **Auto-start**: Add saved config to Windows startup folder for automatic launch
- **Multiple Displays**: VcXsrv handles multiple monitors automatically with "Multiple windows" setting
- **Performance**: "Native opengl" provides better 3D rendering performance for VTK/OpenGL 