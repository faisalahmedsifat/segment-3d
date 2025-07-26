'use client'

import React from 'react'
import { useAnnotationStore } from '@/stores/annotation.store'
import { useProjectStore } from '@/stores/project.store'
import { 
  Trash2, 
  Eye, 
  EyeOff, 
  Edit3, 
  Check, 
  X, 
  Clock,
  Target
} from 'lucide-react'
import { toast } from 'sonner'

interface AnnotationHistoryProps {
  className?: string
}

export function AnnotationHistory({ className = '' }: AnnotationHistoryProps) {
  const {
    currentAnnotations,
    selectedAnnotation,
    annotationHistory,
    setSelectedAnnotation,
    removeAnnotation,
    updateAnnotation
  } = useAnnotationStore()

  const { labelSchemas } = useProjectStore()

  // Get label by ID
  const getLabelById = (labelId: string) => {
    return labelSchemas.find(label => label.id === labelId)
  }

  // Handle annotation selection
  const handleSelectAnnotation = (annotation: any) => {
    setSelectedAnnotation(
      selectedAnnotation?.id === annotation.id ? null : annotation
    )
  }

  // Handle annotation deletion
  const handleDeleteAnnotation = (annotationId: string) => {
    if (confirm('Are you sure you want to delete this annotation?')) {
      removeAnnotation(annotationId)
      toast.success('Annotation deleted')
    }
  }

  // Handle annotation status change
  const handleStatusChange = (annotationId: string, status: string) => {
    updateAnnotation(annotationId, { status, updatedAt: new Date().toISOString() })
    toast.success(`Annotation marked as ${status.toLowerCase()}`)
  }

  // Format confidence percentage
  const formatConfidence = (confidence?: number) => {
    if (!confidence) return 'N/A'
    return `${(confidence * 100).toFixed(1)}%`
  }

  // Format date
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString()
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Current Annotations */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">
            Current Annotations ({currentAnnotations.length})
          </h3>
        </div>

        <div className="space-y-2 max-h-64 overflow-y-auto">
          {currentAnnotations.map((annotation) => {
            const label = getLabelById(annotation.labelId)
            const isSelected = selectedAnnotation?.id === annotation.id
            
            return (
              <div
                key={annotation.id}
                className={`
                  p-3 rounded-lg border cursor-pointer transition-all
                  ${isSelected 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300 bg-white'
                  }
                `}
                onClick={() => handleSelectAnnotation(annotation)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      {label && (
                        <div 
                          className="w-3 h-3 rounded-full border"
                          style={{ backgroundColor: label.color }}
                        />
                      )}
                      <span className="text-sm font-medium text-gray-700">
                        {label?.name || 'Unknown Label'}
                      </span>
                    </div>
                    
                    <div className="mt-1 text-xs text-gray-500 space-y-1">
                      <div>Type: {annotation.type}</div>
                      <div>Confidence: {formatConfidence(annotation.confidence)}</div>
                      <div>Status: {annotation.status}</div>
                      <div>Time: {formatDate(annotation.createdAt)}</div>
                    </div>
                  </div>

                  <div className="flex flex-col space-y-1">
                    {/* Status Actions */}
                    {annotation.status === 'DRAFT' && (
                      <div className="flex space-x-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleStatusChange(annotation.id, 'SUBMITTED')
                          }}
                          className="p-1 text-green-600 hover:bg-green-100 rounded"
                          title="Submit annotation"
                        >
                          <Check className="h-3 w-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteAnnotation(annotation.id)
                          }}
                          className="p-1 text-red-600 hover:bg-red-100 rounded"
                          title="Delete annotation"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    )}

                    {annotation.status === 'SUBMITTED' && (
                      <div className="flex space-x-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleStatusChange(annotation.id, 'APPROVED')
                          }}
                          className="p-1 text-green-600 hover:bg-green-100 rounded"
                          title="Approve annotation"
                        >
                          <Check className="h-3 w-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleStatusChange(annotation.id, 'NEEDS_REVISION')
                          }}
                          className="p-1 text-orange-600 hover:bg-orange-100 rounded"
                          title="Request revision"
                        >
                          <Edit3 className="h-3 w-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleStatusChange(annotation.id, 'REJECTED')
                          }}
                          className="p-1 text-red-600 hover:bg-red-100 rounded"
                          title="Reject annotation"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    )}

                    {annotation.status === 'APPROVED' && (
                      <div className="text-green-600 text-xs font-medium">
                        ✓ Approved
                      </div>
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                {isSelected && (
                  <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-600">
                    <div className="space-y-1">
                      <div>ID: {annotation.id}</div>
                      {annotation.data?.worldPoint && (
                        <div>
                          World Position: 
                          ({annotation.data.worldPoint.x.toFixed(2)}, 
                           {annotation.data.worldPoint.y.toFixed(2)}, 
                           {annotation.data.worldPoint.z.toFixed(2)})
                        </div>
                      )}
                      {annotation.data?.screenPoint && (
                        <div>
                          Screen Position: 
                          ({Math.round(annotation.data.screenPoint.x)}, 
                           {Math.round(annotation.data.screenPoint.y)})
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}

          {currentAnnotations.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              <Target className="h-8 w-8 mx-auto mb-2 text-gray-300" />
              <div className="text-sm">No annotations yet</div>
              <div className="text-xs">Click on the 3D model to start annotating</div>
            </div>
          )}
        </div>
      </div>

      {/* Recent History */}
      {annotationHistory.length > 0 && (
        <div className="space-y-2 border-t border-gray-200 pt-4">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center">
            <Clock className="h-4 w-4 mr-1" />
            Recent History
          </h3>
          
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {annotationHistory.slice(0, 5).map((annotation, index) => {
              const label = getLabelById(annotation.labelId)
              
              return (
                <div key={`history-${annotation.id}-${index}`} className="p-2 bg-gray-50 rounded text-xs">
                  <div className="flex items-center space-x-2">
                    {label && (
                      <div 
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: label.color }}
                      />
                    )}
                    <span className="text-gray-600">
                      {label?.name} - {formatDate(annotation.updatedAt)}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default AnnotationHistory 