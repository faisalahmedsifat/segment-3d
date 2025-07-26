'use client'

import React, { useState, useCallback } from 'react'
import * as THREE from 'three'
import { useAnnotationStore } from '@/stores/annotation.store'
import { STLViewer } from './STLViewer'
import { AnnotationTools } from './AnnotationTools'
import { AnnotationHistory } from './AnnotationHistory'
import { toast } from 'sonner'

interface AnnotationWorkspaceProps {
  modelUrl: string
  className?: string
}

export function AnnotationWorkspace({ modelUrl, className = '' }: AnnotationWorkspaceProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  
  const {
    currentFile,
    activeTool,
    activeLabel,
    clickPoints,
    setSamProcessing,
    setSamMasks,
    addAnnotation
  } = useAnnotationStore()

  // Handle point clicks from 3D viewer - trigger SAM segmentation
  const handlePointClick = useCallback(async (worldPoint: THREE.Vector3, screenPosition: THREE.Vector2) => {
    if (activeTool !== 'segment' || !activeLabel) {
      toast.error('Please select the segment tool and a label first')
      return
    }

    if (!currentFile) {
      toast.error('No file loaded for annotation')
      return
    }

    setIsProcessing(true)
    setSamProcessing(true)

    try {
      // Take screenshot of current 3D view for SAM processing
      const screenshot = await captureViewerScreenshot()
      
      if (!screenshot) {
        throw new Error('Failed to capture 3D view')
      }

      // Prepare SAM request
      const samRequest = {
        image_base64: screenshot,
        points: [[screenPosition.x, screenPosition.y]],
        labels: [1], // Positive click
        model_size: 'vit_b', // Default to fast model
        multimask_output: true
      }

      // Call SAM API
      const response = await fetch('/api/v1/sam/segment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(samRequest)
      })

      if (!response.ok) {
        throw new Error(`SAM API error: ${response.statusText}`)
      }

      const result = await response.json()

      if (result.success) {
        // Process SAM results
        const masks = result.masks.map((mask: number[][], index: number) => ({
          id: `mask-${Date.now()}-${index}`,
          data: mask,
          label: activeLabel,
          confidence: result.scores[index]
        }))

        setSamMasks(masks)

        // Create annotation from best mask
        if (masks.length > 0) {
          const bestMask = masks.reduce((prev: any, current: any) => 
            prev.confidence > current.confidence ? prev : current
          )

          const annotation = {
            id: `annotation-${Date.now()}`,
            type: 'SEGMENTATION_3D' as const,
            data: {
              worldPoint: {
                x: worldPoint.x,
                y: worldPoint.y,
                z: worldPoint.z
              },
              screenPoint: {
                x: screenPosition.x,
                y: screenPosition.y
              },
              mask: bestMask.data,
              samResult: result
            },
            confidence: bestMask.confidence,
            status: 'DRAFT' as const,
            labelId: activeLabel,
            fileId: currentFile.id,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          }

          addAnnotation(annotation)
          toast.success(`Segmentation completed with ${(bestMask.confidence * 100).toFixed(1)}% confidence`)
        }

      } else {
        throw new Error(result.error || 'SAM segmentation failed')
      }

    } catch (error) {
      console.error('SAM segmentation error:', error)
      toast.error(`Segmentation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsProcessing(false)
      setSamProcessing(false)
    }
  }, [activeTool, activeLabel, currentFile, setSamProcessing, setSamMasks, addAnnotation])

  // Capture screenshot from 3D viewer for SAM processing
  const captureViewerScreenshot = useCallback(async (): Promise<string | null> => {
    try {
      // This would capture the current WebGL canvas
      // For now, we'll return a placeholder - this needs to be implemented
      // with proper canvas capture from the Three.js renderer
      
      // Find the canvas element
      const canvas = document.querySelector('canvas')
      if (!canvas) {
        throw new Error('3D canvas not found')
      }

      // Convert canvas to base64
      const dataURL = canvas.toDataURL('image/png')
      const base64 = dataURL.split(',')[1]
      
      return base64
    } catch (error) {
      console.error('Screenshot capture error:', error)
      return null
    }
  }, [])

  return (
    <div className={`flex h-full ${className}`}>
      {/* Main 3D Viewer */}
      <div className="flex-1 relative">
        <STLViewer 
          modelUrl={modelUrl}
          onPointClick={handlePointClick}
          className="w-full h-full"
        />
        
        {/* Processing Overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-30">
            <div className="bg-white p-6 rounded-lg shadow-lg flex items-center space-x-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <div className="text-lg font-medium">Processing with SAM...</div>
            </div>
          </div>
        )}
      </div>

      {/* Right Sidebar */}
      <div className="w-80 bg-gray-50 border-l border-gray-200 flex flex-col">
        {/* Annotation Tools */}
        <div className="p-4">
          <AnnotationTools />
        </div>

        {/* Annotation History */}
        <div className="flex-1 p-4 border-t border-gray-200">
          <AnnotationHistory />
        </div>
      </div>
    </div>
  )
}

export default AnnotationWorkspace 