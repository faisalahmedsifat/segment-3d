# 🐳 Docker Setup for Segment3D

This document provides comprehensive instructions for running Segment3D using Docker, following current best practices for containerized GUI applications.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🚀 Prerequisites

### System Requirements

- **Docker Engine**: 24.0+ with BuildKit support
- **Docker Compose**: v2.20+ (comes with Docker Desktop)
- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Memory**: Minimum 4GB RAM, 8GB+ recommended
- **Storage**: At least 10GB free space for images and data

### Platform-Specific Requirements

#### Linux
- X11 server running (for GUI display)
- `xauth` package installed
- User access to Docker daemon

#### macOS
- XQuartz installed ([download here](https://www.xquartz.org/))
- Docker Desktop with sufficient memory allocation

#### Windows
- Docker Desktop with WSL2 backend
- X11 server (VcXsrv, Xming, or MobaXterm)
- Windows Subsystem for Linux 2 (WSL2)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd segment-3d

# Set up environment variables
cp .env.example .env
# Edit .env with your Modal credentials and preferences
```

### 2. Configure X11 Display (Linux/macOS)

Run the automated setup script:

```bash
./scripts/setup-x11.sh
```

Or manually configure:

**Linux:**
```bash
# Allow local X11 connections
xhost +local:docker

# Verify DISPLAY is set
echo $DISPLAY
```

**macOS:**
```bash
# Start XQuartz
open -a XQuartz

# Allow localhost connections
xhost +localhost
```

### 3. Start the Application

```bash
# Build and start all services
docker compose up --build

# Or run in detached mode
docker compose up -d --build
```

The GUI application should appear on your desktop within 30-60 seconds.

### 4. Access Additional Services

When running the development stack:

- **Main Application**: GUI window on your desktop
- **Database Admin (pgAdmin)**: http://localhost:5050
- **Redis Admin**: http://localhost:8081
- **Jupyter Lab**: http://localhost:8888

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Modal AI credentials (required)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# Display settings
DISPLAY=:0  # Linux
# DISPLAY=host.docker.internal:0  # macOS/Windows

# Resource limits
MAX_MEMORY=4g
MAX_CPUS=2

# Database credentials
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
```

### Volume Mounts

The Docker setup provides persistent storage:

```yaml
# Host paths -> Container paths
./data -> /app/data          # Input 3D models
./output -> /app/output      # Generated segmentations
./temp -> /app/temp          # Temporary processing files
```

## 🛠️ Development Setup

### Using Development Override

```bash
# Start development environment
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# This provides:
# - Source code hot reloading
# - Development tools (Jupyter, pgAdmin, Redis Commander)
# - Exposed database ports
# - Debug-friendly settings
```

### Development Services

1. **Jupyter Lab** (`http://localhost:8888`)
   - Token: `segment3d` (or set `JUPYTER_TOKEN` in .env)
   - Useful for data analysis and algorithm development

2. **pgAdmin** (`http://localhost:5050`)
   - Email: `admin@segment3d.local`
   - Password: `admin`

3. **Redis Commander** (`http://localhost:8081`)
   - View and manage Redis cache data

### Code Development

```bash
# Enter the development container
docker compose exec segment3d bash

# Install additional packages
pip install new-package

# Run tests
python -m pytest

# Format code
black src/
```

## 🚀 Production Deployment

### Environment Preparation

1. **Create production environment file:**
   ```bash
   cp .env.example .env.prod
   # Set secure passwords and production settings
   ```

2. **Set up secrets (recommended):**
   ```bash
   # Create Docker secrets
   echo "your_modal_token" | docker secret create modal_token -
   echo "secure_db_password" | docker secret create postgres_password -
   ```

### Production Deployment

```bash
# Build production images
docker compose build --target runtime

# Start production stack
docker compose --env-file .env.prod up -d

# Monitor logs
docker compose logs -f
```

### Production Checklist

- [ ] Secure passwords in `.env.prod`
- [ ] Enable Docker secrets for sensitive data
- [ ] Configure proper backup strategy for volumes
- [ ] Set up log rotation
- [ ] Configure monitoring and alerting
- [ ] Enable container auto-restart policies
- [ ] Review resource limits and scaling needs

## 🔧 Troubleshooting

### Common Issues

#### GUI Application Not Appearing

**Linux:**
```bash
# Check DISPLAY variable
echo $DISPLAY

# Allow Docker X11 access
xhost +local:docker

# Check if X11 socket is accessible
ls -la /tmp/.X11-unix/

# Verify container can access X11
docker compose exec segment3d echo $DISPLAY
```

**macOS:**
```bash
# Ensure XQuartz is running
ps aux | grep XQuartz

# Check X11 forwarding
xhost +localhost

# Set correct DISPLAY in .env
DISPLAY=host.docker.internal:0
```

**Windows:**
```bash
# Ensure X11 server is running (VcXsrv, Xming, etc.)
# Set DISPLAY in .env
DISPLAY=host.docker.internal:0
```

#### Modal Connection Issues

```bash
# Check Modal credentials
docker compose exec segment3d python -c "
from src.modal_client import ModalSAMClient
client = ModalSAMClient()
print('Modal connection successful')
"

# Check Modal app deployment
modal app list
```

#### Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data ./output ./temp

# Set correct user ID in docker-compose.yml
user: "$(id -u):$(id -g)"
```

#### Database Connection Issues

```bash
# Check database logs
docker compose logs segment3d-db

# Test database connection
docker compose exec segment3d-db psql -U segment3d -d segment3d -c "SELECT version();"

# Reset database
docker compose down -v
docker compose up -d segment3d-db
```

### Performance Troubleshooting

```bash
# Monitor resource usage
docker stats

# Check application logs
docker compose logs segment3d

# Profile container performance
docker compose exec segment3d top
```

## 🔍 Advanced Usage

### Custom Build Arguments

```bash
# Build with custom Python version
docker compose build --build-arg PYTHON_VERSION=3.11

# Build for different architecture
docker compose build --platform linux/amd64
```

### Multi-Architecture Support

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t segment3d:latest .
```

### Scaling Services

```bash
# Scale specific services
docker compose up -d --scale segment3d-redis=2

# Use Docker Swarm for production scaling
docker swarm init
docker stack deploy -c docker-compose.yml segment3d-stack
```

### Custom Networks

```bash
# Create custom network
docker network create segment3d-external

# Connect to external services
docker compose --profile external up
```

### Backup and Restore

```bash
# Backup volumes
docker run --rm -v segment3d_postgres_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v segment3d_postgres_data:/data -v $(pwd):/backup alpine \
    tar xzf /backup/postgres_backup.tar.gz -C /data
```

## 📊 Monitoring and Logging

### Health Checks

```bash
# Check service health
docker compose ps

# View health check logs
docker inspect segment3d-app | jq '.[0].State.Health'
```

### Log Management

```bash
# View logs with timestamps
docker compose logs -t segment3d

# Follow logs in real-time
docker compose logs -f

# Limit log output
docker compose logs --tail=100 segment3d

# Export logs
docker compose logs segment3d > segment3d.log
```

## 🛡️ Security Best Practices

1. **Never run containers as root in production**
2. **Use Docker secrets for sensitive data**
3. **Regularly update base images**
4. **Scan images for vulnerabilities**
5. **Limit container capabilities**
6. **Use read-only file systems where possible**
7. **Implement proper network segmentation**

### Security Commands

```bash
# Scan for vulnerabilities
docker scout cves segment3d:latest

# Check for security issues
docker bench-security

# Audit container configuration
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image segment3d:latest
```

## 🤝 Contributing

When contributing Docker-related changes:

1. Test on multiple platforms (Linux, macOS, Windows)
2. Update documentation for any new services or configurations
3. Follow Docker best practices and security guidelines
4. Ensure backward compatibility with existing setups
5. Update version numbers appropriately

## 📚 Additional Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Specification](https://docs.docker.com/compose/compose-file/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [X11 Forwarding Guide](https://wiki.archlinux.org/title/Xorg#Using_xhost)
- [Modal Documentation](https://modal.com/docs)

---

**Need help?** Open an issue on GitHub or check the troubleshooting section above. 