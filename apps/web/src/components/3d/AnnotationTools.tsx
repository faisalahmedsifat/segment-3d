'use client'

import React from 'react'
import { useAnnotationStore } from '@/stores/annotation.store'
import { useProjectStore } from '@/stores/project.store'
import { 
  MousePointer, 
  Scissors, 
  Square, 
  Polygon, 
  Target,
  Eye,
  EyeOff,
  Layers,
  Settings
} from 'lucide-react'

interface AnnotationToolsProps {
  className?: string
}

export function AnnotationTools({ className = '' }: AnnotationToolsProps) {
  const {
    activeTool,
    activeLabel,
    showMesh,
    showAnnotations,
    samProcessing,
    clickPoints,
    setActiveTool,
    setActiveLabel,
    toggleMesh,
    toggleAnnotations,
    clearClickPoints,
    setShowSettings
  } = useAnnotationStore()

  const { labelSchemas } = useProjectStore()

  const tools = [
    {
      id: 'select' as const,
      name: 'Select',
      icon: MousePointer,
      description: 'Select and move annotations'
    },
    {
      id: 'segment' as const,
      name: 'Segment',
      icon: Scissors,
      description: 'Click to segment with SAM'
    },
    {
      id: 'bbox' as const,
      name: 'Bounding Box',
      icon: Square,
      description: 'Draw 3D bounding boxes'
    },
    {
      id: 'polygon' as const,
      name: 'Polygon',
      icon: Polygon,
      description: 'Draw polygon annotations'
    },
    {
      id: 'keypoint' as const,
      name: 'Keypoint',
      icon: Target,
      description: 'Add keypoint annotations'
    }
  ]

  return (
    <div className={`bg-white shadow-lg rounded-lg p-4 space-y-4 ${className}`}>
      {/* Tool Selection */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Annotation Tools</h3>
        <div className="grid grid-cols-2 gap-2">
          {tools.map((tool) => {
            const Icon = tool.icon
            const isActive = activeTool === tool.id
            
            return (
              <button
                key={tool.id}
                onClick={() => setActiveTool(isActive ? null : tool.id)}
                className={`
                  flex flex-col items-center p-3 rounded-lg border-2 transition-all
                  ${isActive 
                    ? 'border-blue-500 bg-blue-50 text-blue-700' 
                    : 'border-gray-200 hover:border-gray-300 text-gray-600 hover:text-gray-700'
                  }
                  ${samProcessing && tool.id === 'segment' ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                disabled={samProcessing && tool.id === 'segment'}
                title={tool.description}
              >
                <Icon className="h-5 w-5 mb-1" />
                <span className="text-xs font-medium">{tool.name}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Label Selection */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Labels</h3>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {labelSchemas.map((label) => (
            <button
              key={label.id}
              onClick={() => setActiveLabel(activeLabel === label.id ? null : label.id)}
              className={`
                w-full flex items-center p-2 rounded text-left transition-colors
                ${activeLabel === label.id
                  ? 'bg-blue-100 text-blue-700 border border-blue-300'
                  : 'hover:bg-gray-50 text-gray-600'
                }
              `}
            >
              <div 
                className="w-3 h-3 rounded-full mr-2 border"
                style={{ backgroundColor: label.color }}
              />
              <span className="text-sm font-medium">{label.name}</span>
            </button>
          ))}
          
          {labelSchemas.length === 0 && (
            <div className="text-sm text-gray-400 italic p-2">
              No labels defined for this project
            </div>
          )}
        </div>
      </div>

      {/* Visibility Controls */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Display</h3>
        <div className="space-y-2">
          <button
            onClick={toggleMesh}
            className={`
              w-full flex items-center justify-between p-2 rounded transition-colors
              ${showMesh ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-600'}
            `}
          >
            <span className="flex items-center text-sm">
              {showMesh ? <Eye className="h-4 w-4 mr-2" /> : <EyeOff className="h-4 w-4 mr-2" />}
              3D Model
            </span>
          </button>
          
          <button
            onClick={toggleAnnotations}
            className={`
              w-full flex items-center justify-between p-2 rounded transition-colors
              ${showAnnotations ? 'bg-green-50 text-green-700' : 'bg-gray-50 text-gray-600'}
            `}
          >
            <span className="flex items-center text-sm">
              {showAnnotations ? <Layers className="h-4 w-4 mr-2" /> : <EyeOff className="h-4 w-4 mr-2" />}
              Annotations
            </span>
          </button>
        </div>
      </div>

      {/* SAM Status */}
      {activeTool === 'segment' && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">SAM Segmentation</h3>
          <div className="bg-blue-50 p-3 rounded-lg">
            {samProcessing ? (
              <div className="flex items-center text-blue-700">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 mr-2"></div>
                <span className="text-sm">Processing...</span>
              </div>
            ) : (
              <div className="text-sm text-blue-700">
                Click on the 3D model to segment
                {clickPoints.length > 0 && (
                  <div className="mt-1 text-xs">
                    {clickPoints.length} point{clickPoints.length !== 1 ? 's' : ''} selected
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-2 pt-2 border-t">
        {clickPoints.length > 0 && (
          <button
            onClick={clearClickPoints}
            className="w-full px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Clear Points ({clickPoints.length})
          </button>
        )}
        
        <button
          onClick={() => setShowSettings(true)}
          className="w-full flex items-center justify-center px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
        >
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </button>
      </div>
    </div>
  )
}

export default AnnotationTools 