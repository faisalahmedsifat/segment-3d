import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

export interface AnnotationPoint {
  id: string
  x: number
  y: number
  z?: number // For 3D annotations
  label: string
  confidence?: number
}

export interface AnnotationMask {
  id: string
  data: number[][] // 2D array representing the mask
  label: string
  confidence?: number
}

export interface Annotation {
  id: string
  type: 'SEGMENTATION_2D' | 'SEGMENTATION_3D' | 'BOUNDING_BOX_2D' | 'BOUNDING_BOX_3D' | 'POLYGON_2D' | 'KEYPOINT_2D' | 'KEYPOINT_3D' | 'CLASSIFICATION'
  data: any // Flexible data structure for different annotation types
  confidence?: number
  status: 'DRAFT' | 'SUBMITTED' | 'APPROVED' | 'REJECTED' | 'NEEDS_REVISION'
  labelId: string
  fileId: string
  createdAt: string
  updatedAt: string
}

export interface FileItem {
  id: string
  filename: string
  originalName: string
  mimeType: string
  size: number
  storageUrl: string
  metadata: Record<string, any>
  annotations: Annotation[]
}

interface AnnotationState {
  // Current file being annotated
  currentFile: FileItem | null
  files: FileItem[]
  
  // Current annotation session
  currentAnnotations: Annotation[]
  selectedAnnotation: Annotation | null
  
  // Annotation tools state
  activeTool: 'select' | 'segment' | 'bbox' | 'polygon' | 'keypoint' | null
  activeLabel: string | null
  
  // 3D viewer state
  cameraPosition: [number, number, number]
  cameraTarget: [number, number, number]
  showMesh: boolean
  showAnnotations: boolean
  
  // SAM integration state
  samProcessing: boolean
  samMasks: AnnotationMask[]
  clickPoints: AnnotationPoint[]
  
  // UI state
  isAnnotating: boolean
  showSettings: boolean
  annotationHistory: Annotation[]
}

interface AnnotationActions {
  // File management
  setCurrentFile: (file: FileItem | null) => void
  setFiles: (files: FileItem[]) => void
  addFile: (file: FileItem) => void
  updateFile: (id: string, updates: Partial<FileItem>) => void
  
  // Annotation management
  setCurrentAnnotations: (annotations: Annotation[]) => void
  addAnnotation: (annotation: Annotation) => void
  updateAnnotation: (id: string, updates: Partial<Annotation>) => void
  removeAnnotation: (id: string) => void
  setSelectedAnnotation: (annotation: Annotation | null) => void
  
  // Tool management
  setActiveTool: (tool: 'select' | 'segment' | 'bbox' | 'polygon' | 'keypoint' | null) => void
  setActiveLabel: (labelId: string | null) => void
  
  // 3D viewer actions
  setCameraPosition: (position: [number, number, number]) => void
  setCameraTarget: (target: [number, number, number]) => void
  toggleMesh: () => void
  toggleAnnotations: () => void
  
  // SAM integration
  setSamProcessing: (processing: boolean) => void
  addClickPoint: (point: AnnotationPoint) => void
  clearClickPoints: () => void
  setSamMasks: (masks: AnnotationMask[]) => void
  
  // UI actions
  setIsAnnotating: (annotating: boolean) => void
  setShowSettings: (show: boolean) => void
  addToHistory: (annotation: Annotation) => void
  
  // Reset actions
  reset: () => void
  resetSession: () => void
}

export const useAnnotationStore = create<AnnotationState & AnnotationActions>()(
  immer((set) => ({
    // Initial state
    currentFile: null,
    files: [],
    currentAnnotations: [],
    selectedAnnotation: null,
    activeTool: null,
    activeLabel: null,
    cameraPosition: [0, 0, 10],
    cameraTarget: [0, 0, 0],
    showMesh: true,
    showAnnotations: true,
    samProcessing: false,
    samMasks: [],
    clickPoints: [],
    isAnnotating: false,
    showSettings: false,
    annotationHistory: [],

    // File actions
    setCurrentFile: (file) => {
      set((state) => {
        state.currentFile = file
        state.currentAnnotations = file?.annotations || []
      })
    },

    setFiles: (files) => {
      set((state) => {
        state.files = files
      })
    },

    addFile: (file) => {
      set((state) => {
        state.files.push(file)
      })
    },

    updateFile: (id, updates) => {
      set((state) => {
        const index = state.files.findIndex(f => f.id === id)
        if (index !== -1) {
          Object.assign(state.files[index], updates)
        }
        if (state.currentFile?.id === id) {
          Object.assign(state.currentFile, updates)
        }
      })
    },

    // Annotation actions
    setCurrentAnnotations: (annotations) => {
      set((state) => {
        state.currentAnnotations = annotations
      })
    },

    addAnnotation: (annotation) => {
      set((state) => {
        state.currentAnnotations.push(annotation)
        state.addToHistory(annotation)
        
        // Update current file's annotations
        if (state.currentFile) {
          state.currentFile.annotations.push(annotation)
        }
      })
    },

    updateAnnotation: (id, updates) => {
      set((state) => {
        const index = state.currentAnnotations.findIndex(a => a.id === id)
        if (index !== -1) {
          Object.assign(state.currentAnnotations[index], updates)
          state.addToHistory(state.currentAnnotations[index])
        }
        
        // Update in current file
        if (state.currentFile) {
          const fileIndex = state.currentFile.annotations.findIndex(a => a.id === id)
          if (fileIndex !== -1) {
            Object.assign(state.currentFile.annotations[fileIndex], updates)
          }
        }
      })
    },

    removeAnnotation: (id) => {
      set((state) => {
        state.currentAnnotations = state.currentAnnotations.filter(a => a.id !== id)
        
        if (state.selectedAnnotation?.id === id) {
          state.selectedAnnotation = null
        }
        
        // Update current file
        if (state.currentFile) {
          state.currentFile.annotations = state.currentFile.annotations.filter(a => a.id !== id)
        }
      })
    },

    setSelectedAnnotation: (annotation) => {
      set((state) => {
        state.selectedAnnotation = annotation
      })
    },

    // Tool actions
    setActiveTool: (tool) => {
      set((state) => {
        state.activeTool = tool
        state.isAnnotating = tool !== null
      })
    },

    setActiveLabel: (labelId) => {
      set((state) => {
        state.activeLabel = labelId
      })
    },

    // 3D viewer actions
    setCameraPosition: (position) => {
      set((state) => {
        state.cameraPosition = position
      })
    },

    setCameraTarget: (target) => {
      set((state) => {
        state.cameraTarget = target
      })
    },

    toggleMesh: () => {
      set((state) => {
        state.showMesh = !state.showMesh
      })
    },

    toggleAnnotations: () => {
      set((state) => {
        state.showAnnotations = !state.showAnnotations
      })
    },

    // SAM actions
    setSamProcessing: (processing) => {
      set((state) => {
        state.samProcessing = processing
      })
    },

    addClickPoint: (point) => {
      set((state) => {
        state.clickPoints.push(point)
      })
    },

    clearClickPoints: () => {
      set((state) => {
        state.clickPoints = []
      })
    },

    setSamMasks: (masks) => {
      set((state) => {
        state.samMasks = masks
      })
    },

    // UI actions
    setIsAnnotating: (annotating) => {
      set((state) => {
        state.isAnnotating = annotating
      })
    },

    setShowSettings: (show) => {
      set((state) => {
        state.showSettings = show
      })
    },

    addToHistory: (annotation) => {
      set((state) => {
        state.annotationHistory.unshift(annotation)
        // Keep only last 50 items in history
        if (state.annotationHistory.length > 50) {
          state.annotationHistory = state.annotationHistory.slice(0, 50)
        }
      })
    },

    // Reset actions
    reset: () => {
      set((state) => {
        state.currentFile = null
        state.files = []
        state.currentAnnotations = []
        state.selectedAnnotation = null
        state.activeTool = null
        state.activeLabel = null
        state.samProcessing = false
        state.samMasks = []
        state.clickPoints = []
        state.isAnnotating = false
        state.annotationHistory = []
      })
    },

    resetSession: () => {
      set((state) => {
        state.currentAnnotations = []
        state.selectedAnnotation = null
        state.activeTool = null
        state.samMasks = []
        state.clickPoints = []
        state.isAnnotating = false
      })
    },
  }))
) 