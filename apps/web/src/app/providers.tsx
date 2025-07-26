'use client'

interface ProvidersProps {
  children: React.ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <>
      {children}
      {/* Toast notifications will be added when we install sonner */}
    </>
  )
} 