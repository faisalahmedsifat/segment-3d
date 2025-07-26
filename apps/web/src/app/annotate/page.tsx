'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { useAnnotationStore } from '@/stores/annotation.store'
import { useProjectStore } from '@/stores/project.store'
import { AnnotationWorkspace } from '@/components/3d/AnnotationWorkspace'
import { FileText, ArrowLeft, Save, Download, Settings } from 'lucide-react'
import Link from 'next/link'
import { toast } from 'sonner'

export default function AnnotatePage() {
  const searchParams = useSearchParams()
  const modelUrl = searchParams.get('model') || '/api/files/default-model.stl'
  const projectId = searchParams.get('project')
  const fileId = searchParams.get('file')
  
  const [isSaving, setIsSaving] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  
  const {
    currentFile,
    currentAnnotations,
    setCurrentFile,
    resetSession
  } = useAnnotationStore()
  
  const {
    currentProject,
    labelSchemas,
    setCurrentProject,
    setLabelSchemas
  } = useProjectStore()

  // Initialize annotation session
  useEffect(() => {
    if (projectId && fileId) {
      // Load project and file data
      loadProjectData(projectId)
      loadFileData(fileId)
    }
  }, [projectId, fileId])

  const loadProjectData = async (id: string) => {
    try {
      const response = await fetch(`/api/v1/projects/${id}`)
      if (response.ok) {
        const project = await response.json()
        setCurrentProject(project)
        
        // Load label schemas for this project
        if (project.labelSchema) {
          setLabelSchemas(project.labelSchema)
        }
      }
    } catch (error) {
      console.error('Failed to load project:', error)
      toast.error('Failed to load project data')
    }
  }

  const loadFileData = async (id: string) => {
    try {
      // This would load file data from API
      // For now, create a mock file object
      const mockFile = {
        id,
        filename: 'model.stl',
        originalName: 'Sample Model.stl',
        mimeType: 'application/sla',
        size: 1024 * 1024,
        storageUrl: modelUrl,
        metadata: {},
        annotations: []
      }
      
      setCurrentFile(mockFile)
    } catch (error) {
      console.error('Failed to load file:', error)
      toast.error('Failed to load file data')
    }
  }

  const handleSave = async () => {
    if (!currentFile || currentAnnotations.length === 0) {
      toast.error('No annotations to save')
      return
    }

    setIsSaving(true)
    
    try {
      // Save annotations to backend
      const response = await fetch('/api/v1/annotations/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileId: currentFile.id,
          annotations: currentAnnotations.map(annotation => ({
            ...annotation,
            status: 'SUBMITTED' // Auto-submit when saving
          }))
        })
      })

      if (response.ok) {
        toast.success(`Saved ${currentAnnotations.length} annotations`)
      } else {
        throw new Error('Failed to save annotations')
      }
    } catch (error) {
      console.error('Save error:', error)
      toast.error('Failed to save annotations') 
    } finally {
      setIsSaving(false)
    }
  }

  const handleExport = async () => {
    if (!currentFile || currentAnnotations.length === 0) {
      toast.error('No annotations to export')
      return
    }

    setIsExporting(true)
    
    try {
      // Export annotations in various formats
      const response = await fetch('/api/v1/export/annotations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fileId: currentFile.id,
          format: 'stl', // Export segmented STL
          annotations: currentAnnotations.filter(a => a.status === 'APPROVED')
        })
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${currentFile.filename}_segmented.stl`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
        
        toast.success('Annotations exported successfully')
      } else {
        throw new Error('Failed to export annotations')
      }
    } catch (error) {
      console.error('Export error:', error)
      toast.error('Failed to export annotations')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Left side */}
            <div className="flex items-center space-x-4">
              <Link
                href="/dashboard"
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Dashboard
              </Link>
              
              <div className="flex items-center">
                <FileText className="h-5 w-5 text-gray-400 mr-2" />
                <div>
                  <h1 className="text-lg font-semibold text-gray-900">
                    {currentFile?.originalName || 'Annotation Workspace'}
                  </h1>
                  {currentProject && (
                    <p className="text-sm text-gray-500">
                      Project: {currentProject.name}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Right side */}
            <div className="flex items-center space-x-3">
              {/* Annotation count */}
              <div className="text-sm text-gray-600">
                {currentAnnotations.length} annotation{currentAnnotations.length !== 1 ? 's' : ''}
              </div>

              {/* Action buttons */}
              <button
                onClick={handleSave}
                disabled={isSaving || currentAnnotations.length === 0}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSaving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Save
                  </>
                )}
              </button>

              <button
                onClick={handleExport}
                disabled={isExporting || currentAnnotations.length === 0}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isExporting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export
                  </>
                )}
              </button>

              <button className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
                <Settings className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1" style={{ height: 'calc(100vh - 64px)' }}>
        {currentFile ? (
          <AnnotationWorkspace 
            modelUrl={currentFile.storageUrl}
            className="h-full"
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h2 className="text-lg font-medium text-gray-900 mb-2">
                Loading 3D Model...
              </h2>
              <p className="text-gray-600">
                Please wait while we prepare your annotation workspace
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
} 