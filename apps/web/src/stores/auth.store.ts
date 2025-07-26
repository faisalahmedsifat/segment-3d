import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { persist } from 'zustand/middleware'

export interface User {
  id: string
  email: string
  username: string
  firstName?: string
  lastName?: string
  avatar?: string
  role: 'ADMIN' | 'PROJECT_MANAGER' | 'ANNOTATOR' | 'REVIEWER' | 'VIEWER'
  isActive: boolean
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
}

interface AuthActions {
  login: (token: string, user: User) => void
  logout: () => void
  updateUser: (updates: Partial<User>) => void
  setLoading: (loading: boolean) => void
}

export const useAuthStore = create<AuthState & AuthActions>()(
  persist(
    immer((set) => ({
      // State
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      // Actions
      login: (token: string, user: User) => {
        set((state) => {
          state.token = token
          state.user = user
          state.isAuthenticated = true
          state.isLoading = false
        })
      },

      logout: () => {
        set((state) => {
          state.token = null
          state.user = null
          state.isAuthenticated = false
          state.isLoading = false
        })
      },

      updateUser: (updates: Partial<User>) => {
        set((state) => {
          if (state.user) {
            Object.assign(state.user, updates)
          }
        })
      },

      setLoading: (loading: boolean) => {
        set((state) => {
          state.isLoading = loading
        })
      },
    })),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
) 