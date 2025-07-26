# Segment3D Platform - Setup Guide

## 🚀 Complete Setup Instructions

### Prerequisites

Before starting, ensure you have:

- **Node.js 18+** and **npm 9+**
- **Python 3.10+** with **pip**
- **PostgreSQL 14+** (for production) or **SQLite** (for development)
- **Modal Account** (free tier: $30/month credits)

### 1. Repository Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd segment3d-platform

# Install all dependencies
npm run setup
```

### 2. Backend Configuration

#### Environment Variables
```bash
# Create environment file
cd services/api
cp .env.example .env

# Edit .env with your settings:
DATABASE_URL="postgresql://user:password@localhost:5432/segment3d"
SECRET_KEY="your-super-secret-jwt-key-here"
MODAL_TOKEN_ID="your-modal-token-id"
MODAL_TOKEN_SECRET="your-modal-token-secret"
```

#### Database Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Generate Prisma client
prisma generate

# Run database migrations
prisma db push

# Optional: Seed with sample data
python scripts/seed_data.py
```

### 3. Modal SAM Deployment

#### Setup Modal
```bash
# Install Modal CLI
pip install modal

# Set up Modal auth (follow prompts)
modal setup

# Deploy SAM service
cd services/sam
modal deploy sam_service.py
```

#### Verify Deployment
```bash
# Test SAM service
modal run sam_service.py::health_check
```

### 4. Frontend Configuration

```bash
cd apps/web

# Create environment file
cp .env.example .env.local

# Edit .env.local:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 5. Start Development

#### Option A: Start Everything
```bash
# From repository root
npm run start:dev
```

#### Option B: Start Individually
```bash
# Terminal 1: Backend
cd services/api
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend  
cd apps/web
npm run dev
```

### 6. Access the Platform

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

---

## 🧪 Testing the 3D Annotation

### Quick Test Setup

1. **Copy Test STL File**:
   ```bash
   # Copy your existing test STL
   cp data/test_model.stl apps/web/public/test-model.stl
   ```

2. **Access Annotation Workspace**:
   ```
   http://localhost:3000/annotate?model=/test-model.stl&project=demo&file=test1
   ```

3. **Test Workflow**:
   - Select "Segment" tool
   - Choose a label (create demo labels first)
   - Click on 3D model to trigger SAM segmentation
   - View results in annotation history
   - Save and export annotations

### Create Demo Data

```bash
# Run demo data script
cd services/api
python scripts/create_demo_data.py
```

This creates:
- Demo user account
- Sample project with labels
- Test file entries

---

## 🔧 Development Commands

### Frontend (Next.js)
```bash
cd apps/web

npm run dev          # Development server
npm run build        # Production build
npm run lint         # Lint code
npm run type-check   # TypeScript checking
```

### Backend (FastAPI)
```bash
cd services/api

# Development
python -m uvicorn main:app --reload

# Production
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Database
prisma db push       # Apply schema changes
prisma generate      # Regenerate client
prisma studio        # Database GUI
```

### SAM Service (Modal)
```bash
cd services/sam

modal deploy sam_service.py      # Deploy to Modal
modal run sam_service.py::benchmark_models  # Benchmark
modal logs                       # View logs
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Modal Authentication
```bash
# If Modal auth fails
modal setup --force
modal token set <your-token-id> <your-token-secret>
```

#### 2. Database Connection
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Reset database
prisma db push --force-reset
```

#### 3. Three.js Loading Issues
- Ensure STL files are in `apps/web/public/` directory
- Check browser console for CORS errors
- Verify file paths are correct

#### 4. SAM Segmentation Fails
- Check Modal deployment status: `modal app list`
- Verify Modal tokens in environment
- Check Modal logs: `modal logs segment3d-sam-platform`

### Debug Mode

#### Enable Verbose Logging
```bash
# Backend debug mode
export LOG_LEVEL=DEBUG
export DEBUG=true

# Frontend debug mode
export NEXT_PUBLIC_DEBUG=true
```

#### Monitor API Calls
- Use browser dev tools Network tab
- Check FastAPI logs in terminal
- Use Modal dashboard for SAM monitoring

---

## 🚀 Production Deployment

### Environment Setup
1. Set production environment variables
2. Configure PostgreSQL database
3. Deploy Modal SAM service to production
4. Build and deploy Next.js app
5. Configure reverse proxy (nginx)

### Security Checklist
- [ ] Change default JWT secret
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up database backups
- [ ] Monitor Modal usage/costs

---

## 📊 Platform Status

### ✅ Implemented Features
- [x] 3D STL viewer with Three.js
- [x] Point-click SAM segmentation
- [x] Annotation tools and workflow
- [x] Zustand state management
- [x] FastAPI backend with Prisma
- [x] Modal SAM integration
- [x] Project and user management
- [x] Clean architecture implementation

### 🚧 Next Development Phase
- [ ] Authentication UI (login/register)
- [ ] File upload system
- [ ] Real-time collaboration
- [ ] Export to ML formats (COCO, YOLO)
- [ ] Mobile responsive optimization

---

## 🎯 Success Criteria

Your platform is ready when:

1. ✅ **3D Model Loads**: STL file displays in Three.js viewer
2. ✅ **SAM Integration**: Click triggers segmentation via Modal
3. ✅ **Annotation Flow**: Complete workflow from click to save
4. ✅ **State Management**: Zustand stores work correctly
5. ✅ **API Communication**: Frontend ↔ Backend ↔ Modal chain works

**🎉 Congratulations!** You've successfully transformed your desktop PyQt5 app into a modern, scalable web annotation platform!

---

## Need Help?

- **Frontend Issues**: Check browser console and Next.js terminal
- **Backend Issues**: Check FastAPI terminal and `/docs` endpoint
- **Modal Issues**: Check Modal dashboard and logs
- **Database Issues**: Use Prisma Studio to inspect data

Your platform now combines the power of the original desktop app with modern web technologies and collaborative features! 🚀 