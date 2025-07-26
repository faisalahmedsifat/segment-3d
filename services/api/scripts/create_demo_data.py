#!/usr/bin/env python3
"""
Demo data creation script for Segment3D Platform
Creates sample users, projects, and labels for testing
"""

import asyncio
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prisma import Prisma
from core.auth import get_password_hash
from datetime import datetime


async def create_demo_data():
    """Create demo data for testing the platform"""
    
    db = Prisma()
    await db.connect()
    
    try:
        print("🚀 Creating demo data for Segment3D Platform...")
        
        # 1. Create demo users
        print("\n👥 Creating demo users...")
        
        # Admin user
        admin_user = await db.user.create(
            data={
                "email": "admin@segment3d.demo",
                "username": "admin",
                "firstName": "Admin",
                "lastName": "User",
                "role": "ADMIN",
                "isActive": True
            }
        )
        print(f"✅ Created admin user: {admin_user.email}")
        
        # Annotator user
        annotator_user = await db.user.create(
            data={
                "email": "annotator@segment3d.demo", 
                "username": "annotator",
                "firstName": "Demo",
                "lastName": "Annotator",
                "role": "ANNOTATOR",
                "isActive": True
            }
        )
        print(f"✅ Created annotator user: {annotator_user.email}")
        
        # 2. Create demo project
        print("\n📁 Creating demo project...")
        
        demo_project = await db.project.create(
            data={
                "name": "3D Model Annotation Demo",
                "description": "Sample project for testing 3D annotation workflow with SAM integration",
                "type": "MODEL_3D",
                "ownerId": admin_user.id,
                "settings": {
                    "sam_model_default": "vit_b",
                    "auto_save": True,
                    "quality_threshold": 0.8
                }
            }
        )
        print(f"✅ Created demo project: {demo_project.name}")
        
        # 3. Create label schemas
        print("\n🏷️  Creating label schemas...")
        
        labels = [
            {
                "name": "Component A",
                "color": "#FF5733",
                "description": "Main structural component",
                "type": "SEGMENTATION"
            },
            {
                "name": "Component B", 
                "color": "#33FF57",
                "description": "Secondary component",
                "type": "SEGMENTATION"
            },
            {
                "name": "Surface Defect",
                "color": "#3357FF",
                "description": "Surface anomalies or defects",
                "type": "SEGMENTATION"
            },
            {
                "name": "Connection Point",
                "color": "#FF33F5",
                "description": "Assembly connection points",
                "type": "KEYPOINT"
            },
            {
                "name": "Quality Check",
                "color": "#F5FF33",
                "description": "Areas requiring quality inspection",
                "type": "BOUNDING_BOX"
            }
        ]
        
        created_labels = []
        for label_data in labels:
            label = await db.labelschema.create(
                data={
                    **label_data,
                    "projectId": demo_project.id,
                    "isActive": True
                }
            )
            created_labels.append(label)
            print(f"✅ Created label: {label.name} ({label.color})")
        
        # 4. Create demo dataset
        print("\n📦 Creating demo dataset...")
        
        demo_dataset = await db.dataset.create(
            data={
                "name": "Sample STL Models",
                "description": "Collection of 3D models for annotation testing",
                "type": "MODELS_3D",
                "projectId": demo_project.id,
                "fileCount": 1,
                "totalSize": 1024 * 1024  # 1MB
            }
        )
        print(f"✅ Created demo dataset: {demo_dataset.name}")
        
        # 5. Create demo file entry
        print("\n📄 Creating demo file entry...")
        
        demo_file = await db.file.create(
            data={
                "filename": "test-model.stl",
                "originalName": "Test Model.stl",
                "mimeType": "application/sla",
                "size": 1024 * 1024,
                "checksum": "demo-checksum-123",
                "storageUrl": "/test-model.stl",
                "datasetId": demo_dataset.id,
                "metadata": {
                    "source": "demo",
                    "created_for": "testing",
                    "model_type": "mechanical_part"
                }
            }
        )
        print(f"✅ Created demo file: {demo_file.originalName}")
        
        # 6. Create demo task
        print("\n📋 Creating demo task...")
        
        demo_task = await db.task.create(
            data={
                "name": "Annotate Test Model",
                "status": "IN_PROGRESS",
                "priority": "MEDIUM",
                "instructions": "Please annotate the main components and any surface defects in this 3D model using the SAM segmentation tool.",
                "projectId": demo_project.id,
                "datasetId": demo_dataset.id,
                "assigneeId": annotator_user.id,
                "metadata": {
                    "estimated_time": "30 minutes",
                    "difficulty": "beginner"
                }
            }
        )
        print(f"✅ Created demo task: {demo_task.name}")
        
        # 7. Create project stats
        print("\n📊 Creating project statistics...")
        
        project_stats = await db.projectstats.create(
            data={
                "projectId": demo_project.id,
                "totalFiles": 1,
                "totalAnnotations": 0,
                "completedTasks": 0,
                "pendingTasks": 1,
                "avgQualityScore": None
            }
        )
        print(f"✅ Created project statistics")
        
        # 8. Add project member
        print("\n👥 Adding project member...")
        
        project_member = await db.projectmember.create(
            data={
                "projectId": demo_project.id,
                "userId": annotator_user.id,
                "role": "ANNOTATOR",
                "isActive": True
            }
        )
        print(f"✅ Added annotator as project member")
        
        print("\n🎉 Demo data creation completed successfully!")
        print("\n📋 Summary:")
        print(f"   👥 Users: 2 (admin, annotator)")
        print(f"   📁 Projects: 1 ({demo_project.name})")
        print(f"   🏷️  Labels: {len(created_labels)}")
        print(f"   📦 Datasets: 1 ({demo_dataset.name})")
        print(f"   📄 Files: 1 ({demo_file.originalName})")
        print(f"   📋 Tasks: 1 ({demo_task.name})")
        
        print("\n🔗 Test URLs:")
        print(f"   Frontend: http://localhost:3000")
        print(f"   Annotation: http://localhost:3000/annotate?model=/test-model.stl&project={demo_project.id}&file={demo_file.id}")
        print(f"   API Docs: http://localhost:8000/docs")
        
        print("\n👤 Demo Login (when auth is implemented):")
        print(f"   Admin: admin@segment3d.demo")
        print(f"   Annotator: annotator@segment3d.demo")
        
    except Exception as e:
        print(f"❌ Error creating demo data: {e}")
        raise
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(create_demo_data()) 