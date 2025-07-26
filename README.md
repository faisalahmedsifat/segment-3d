# Segment3D Platform - SAM-Powered Annotation Tool

A comprehensive **3D/2D annotation platform** powered by Meta's Segment Anything Model (SAM) for intelligent region selection and annotation. Built as a modern web application with microservices architecture for scalable, collaborative annotation workflows.

![Platform Architecture](https://img.shields.io/badge/Architecture-Microservices-blue) ![Frontend](https://img.shields.io/badge/Frontend-Next.js-black) ![Backend](https://img.shields.io/badge/Backend-FastAPI-green) ![AI](https://img.shields.io/badge/AI-SAM%20%2B%20Modal-orange)

## 🚀 Features

### **Core Annotation Capabilities**
- **3D Model Annotation**: Interactive Three.js-based 3D viewer with intelligent segmentation
- **2D Image Annotation**: SAM-powered click-to-segment with real-time feedback
- **Cloud-Powered AI**: SAM runs on Modal cloud GPUs - no local GPU required
- **Multi-format Support**: STL, PLY, OBJ for 3D; PNG, JPG, TIFF for 2D
- **Real-time Collaboration**: Multiple users can annotate simultaneously
- **Version Control**: Git-like versioning for annotation datasets

### **Professional Workflow**
- **Project Management**: Organize datasets, assign tasks, track progress
- **Team Collaboration**: Role-based access (Admin, Annotator, Reviewer)
- **Quality Control**: Inter-annotator agreement, validation workflows
- **Export System**: COCO, YOLO, Pascal VOC, custom 3D formats
- **API Integration**: RESTful APIs for external ML pipelines

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Segment3D Platform                            │
├─────────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)          │  Backend Services (FastAPI)        │
│  ┌─────────────────────────┐  │  ┌─────────────────────────────────┐  │
│  │ • 3D Viewer (Three.js)  │  │  │ • Auth Service                 │  │
│  │ • 2D Annotation Editor  │  │  │ • Project Management          │  │
│  │ • Project Dashboard     │  │  │ • SAM Integration Service     │  │
│  │ • Team Collaboration    │  │  │ • Export Service              │  │
│  │ • Quality Control       │  │  │ • File Processing             │  │
│  └─────────────────────────┘  │  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                   Shared Infrastructure                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   PostgreSQL    │  │   Redis Cache   │  │  Modal SAM Cloud   │  │
│  │   (Metadata)    │  │   (Sessions)    │  │   (GPU Inference)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │  
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Monorepo Structure

```
segment3d-platform/
├── apps/
│   └── web/                    # Next.js frontend application
│       ├── src/
│       │   ├── components/     # React components
│       │   ├── pages/          # Next.js pages
│       │   ├── hooks/          # Custom React hooks
│       │   ├── lib/            # Utilities and configurations
│       │   └── types/          # TypeScript type definitions
│       └── public/             # Static assets
├── packages/
│   ├── ui/                     # Shared UI components
│   ├── config/                 # Shared configurations
│   ├── types/                  # Shared TypeScript types
│   └── utils/                  # Shared utilities
├── services/
│   ├── api/                    # Main FastAPI backend
│   │   ├── routers/            # API route handlers
│   │   ├── models/             # Database models
│   │   ├── services/           # Business logic
│   │   └── main.py             # FastAPI app entry point
│   ├── sam/                    # Modal SAM service
│   │   └── sam_service.py      # SAM model deployment
│   └── worker/                 # Background task processing
└── docker/                     # Docker configurations
```

## 🛠️ Quick Start

### Prerequisites
- **Node.js 18+** and **npm 9+**
- **Python 3.10+** with **pip**
- **PostgreSQL 14+** (for production)
- **Modal Account** (free tier includes $30/month credits)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd segment3d-platform
npm run setup  # Installs dependencies and builds packages
```

### 2. Environment Configuration
```bash
# Copy environment templates
cp apps/web/.env.example apps/web/.env.local
cp services/api/.env.example services/api/.env

# Configure your environment variables:
# - Database connections
# - Modal API tokens
# - JWT secrets
# - Cloud storage credentials
```

### 3. Database Setup
```bash
cd services/api
# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_data.py
```

### 4. Deploy SAM Service
```bash
# Deploy SAM model to Modal cloud
npm run deploy:modal
```

### 5. Start Development
```bash
# Start both frontend and backend
npm run start:dev

# Or start individually:
npm run dev:frontend  # Frontend on http://localhost:3000
npm run dev:backend   # Backend on http://localhost:8000
```

### 6. Access the Platform
- **Web App**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 🔧 Development

### Available Commands
```bash
npm run dev          # Start all services in development
npm run build        # Build all applications
npm run test         # Run all tests
npm run lint         # Lint all code
npm run type-check   # TypeScript type checking
npm run format       # Format code with Prettier
```

### Key Technologies
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Three.js
- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic
- **Database**: PostgreSQL, Redis
- **AI/ML**: Segment Anything Model (SAM) via Modal
- **DevOps**: Docker, Turbo (monorepo), GitHub Actions

## 📊 Platform Capabilities

### **Enhanced from Original Desktop App**
✅ **Web-based** - No installation required, works everywhere  
✅ **Multi-user** - Team collaboration and real-time sharing  
✅ **Scalable** - Handle large teams and massive datasets  
✅ **2D + 3D** - Expanded to support both image and model annotation  
✅ **Professional** - Project management, quality control, exports  
✅ **Modern UI** - Beautiful, responsive interface with great UX  

### **Preserved Core Functionality**
✅ **3D STL Support** - Full STL model visualization and segmentation  
✅ **SAM Integration** - Same intelligent segmentation powered by SAM  
✅ **Modal Cloud** - GPU-accelerated processing without local hardware  
✅ **Point-Click Interface** - Intuitive interaction for region selection  
✅ **Export Formats** - Generate segmented STL files and other formats  

## 🎯 Use Cases

- **Research Labs**: Annotate 3D medical scans, geological samples
- **Manufacturing**: Quality control, defect annotation on 3D models  
- **Computer Vision**: Create training datasets for 2D/3D ML models
- **Education**: Interactive learning with annotated 3D content
- **Content Creation**: Segment and label 3D assets for games/VR

## 📈 Roadmap

- [x] **Phase 1**: Core platform architecture and 3D annotation
- [ ] **Phase 2**: 2D image annotation system integration  
- [ ] **Phase 3**: Advanced collaboration and workflow features
- [ ] **Phase 4**: Mobile support and offline capabilities
- [ ] **Phase 5**: Custom model training and advanced AI features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**From Desktop to Cloud** - Transforming a powerful desktop 3D annotation tool into a comprehensive, collaborative web platform for the future of ML dataset creation.