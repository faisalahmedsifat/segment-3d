import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

export interface Project {
  id: string
  name: string
  description?: string
  type: 'MIXED' | 'IMAGE_2D' | 'MODEL_3D'
  settings: Record<string, any>
  isArchived: boolean
  createdAt: string
  updatedAt: string
  owner: {
    id: string
    username: string
    email: string
  }
  stats?: {
    totalFiles: number
    totalAnnotations: number
    completedTasks: number
    pendingTasks: number
    avgQualityScore?: number
  }
}

export interface Dataset {
  id: string
  name: string
  description?: string
  type: 'IMAGES' | 'MODELS_3D' | 'MIXED'
  fileCount: number
  totalSize: number
  createdAt: string
  updatedAt: string
}

export interface LabelSchema {
  id: string
  name: string
  color: string
  description?: string
  type: 'SEGMENTATION' | 'BOUNDING_BOX' | 'POLYGON' | 'KEYPOINT' | 'CLASSIFICATION'
  isActive: boolean
}

interface ProjectState {
  // Current project
  currentProject: Project | null
  projects: Project[]
  
  // Datasets for current project
  datasets: Dataset[]
  currentDataset: Dataset | null
  
  // Label schemas for current project
  labelSchemas: LabelSchema[]
  
  // Loading states
  isLoading: boolean
  isProjectsLoading: boolean
  isDatasetsLoading: boolean
}

interface ProjectActions {
  // Project management
  setCurrentProject: (project: Project | null) => void
  setProjects: (projects: Project[]) => void
  addProject: (project: Project) => void
  updateProject: (id: string, updates: Partial<Project>) => void
  removeProject: (id: string) => void
  
  // Dataset management
  setDatasets: (datasets: Dataset[]) => void
  setCurrentDataset: (dataset: Dataset | null) => void
  addDataset: (dataset: Dataset) => void
  updateDataset: (id: string, updates: Partial<Dataset>) => void
  removeDataset: (id: string) => void
  
  // Label schema management
  setLabelSchemas: (schemas: LabelSchema[]) => void
  addLabelSchema: (schema: LabelSchema) => void
  updateLabelSchema: (id: string, updates: Partial<LabelSchema>) => void
  removeLabelSchema: (id: string) => void
  
  // Loading states
  setLoading: (loading: boolean) => void
  setProjectsLoading: (loading: boolean) => void
  setDatasetsLoading: (loading: boolean) => void
  
  // Reset actions
  reset: () => void
}

export const useProjectStore = create<ProjectState & ProjectActions>()(
  immer((set) => ({
    // Initial state
    currentProject: null,
    projects: [],
    datasets: [],
    currentDataset: null,
    labelSchemas: [],
    isLoading: false,
    isProjectsLoading: false,
    isDatasetsLoading: false,

    // Project actions
    setCurrentProject: (project) => {
      set((state) => {
        state.currentProject = project
      })
    },

    setProjects: (projects) => {
      set((state) => {
        state.projects = projects
      })
    },

    addProject: (project) => {
      set((state) => {
        state.projects.push(project)
      })
    },

    updateProject: (id, updates) => {
      set((state) => {
        const index = state.projects.findIndex(p => p.id === id)
        if (index !== -1) {
          Object.assign(state.projects[index], updates)
        }
        if (state.currentProject?.id === id) {
          Object.assign(state.currentProject, updates)
        }
      })
    },

    removeProject: (id) => {
      set((state) => {
        state.projects = state.projects.filter(p => p.id !== id)
        if (state.currentProject?.id === id) {
          state.currentProject = null
        }
      })
    },

    // Dataset actions
    setDatasets: (datasets) => {
      set((state) => {
        state.datasets = datasets
      })
    },

    setCurrentDataset: (dataset) => {
      set((state) => {
        state.currentDataset = dataset
      })
    },

    addDataset: (dataset) => {
      set((state) => {
        state.datasets.push(dataset)
      })
    },

    updateDataset: (id, updates) => {
      set((state) => {
        const index = state.datasets.findIndex(d => d.id === id)
        if (index !== -1) {
          Object.assign(state.datasets[index], updates)
        }
        if (state.currentDataset?.id === id) {
          Object.assign(state.currentDataset, updates)
        }
      })
    },

    removeDataset: (id) => {
      set((state) => {
        state.datasets = state.datasets.filter(d => d.id !== id)
        if (state.currentDataset?.id === id) {
          state.currentDataset = null
        }
      })
    },

    // Label schema actions
    setLabelSchemas: (schemas) => {
      set((state) => {
        state.labelSchemas = schemas
      })
    },

    addLabelSchema: (schema) => {
      set((state) => {
        state.labelSchemas.push(schema)
      })
    },

    updateLabelSchema: (id, updates) => {
      set((state) => {
        const index = state.labelSchemas.findIndex(s => s.id === id)
        if (index !== -1) {
          Object.assign(state.labelSchemas[index], updates)
        }
      })
    },

    removeLabelSchema: (id) => {
      set((state) => {
        state.labelSchemas = state.labelSchemas.filter(s => s.id !== id)
      })
    },

    // Loading actions
    setLoading: (loading) => {
      set((state) => {
        state.isLoading = loading
      })
    },

    setProjectsLoading: (loading) => {
      set((state) => {
        state.isProjectsLoading = loading
      })
    },

    setDatasetsLoading: (loading) => {
      set((state) => {
        state.isDatasetsLoading = loading
      })
    },

    // Reset
    reset: () => {
      set((state) => {
        state.currentProject = null
        state.projects = []
        state.datasets = []
        state.currentDataset = null
        state.labelSchemas = []
        state.isLoading = false
        state.isProjectsLoading = false
        state.isDatasetsLoading = false
      })
    },
  }))
) 