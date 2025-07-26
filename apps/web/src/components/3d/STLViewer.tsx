'use client'

import React, { useRef, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useLoader, ThreeEvent, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader'
import * as THREE from 'three'
import { useAnnotationStore } from '@/stores/annotation.store'

interface STLViewerProps {
  modelUrl: string
  onPointClick?: (point: THREE.Vector3, screenPosition: THREE.Vector2) => void
  className?: string
}

// STL Model Component
function STLModel({ url, onPointClick }: { url: string; onPointClick?: (point: THREE.Vector3, screenPosition: THREE.Vector2) => void }) {
  const geometry = useLoader(STLLoader, url)
  const meshRef = useRef<THREE.Mesh>(null)
  const { camera, gl, size } = useThree()
  
  const { showMesh, showAnnotations, activeTool } = useAnnotationStore()

  // Handle click events on the 3D model
  const handleClick = useCallback((event: ThreeEvent<MouseEvent>) => {
    if (!onPointClick || activeTool !== 'segment') return
    
    event.stopPropagation()
    
    // Get the 3D world position
    const point = event.point
    
    // Convert 3D world position to screen coordinates
    const screenPosition = point.clone().project(camera)
    screenPosition.x = (screenPosition.x + 1) * size.width / 2
    screenPosition.y = (-screenPosition.y + 1) * size.height / 2
    
    onPointClick(point, new THREE.Vector2(screenPosition.x, screenPosition.y))
  }, [onPointClick, camera, size.width, size.height, activeTool])

  // Create material based on visibility settings
  const material = new THREE.MeshPhongMaterial({
    color: showMesh ? 0x888888 : 0x333333,
    shininess: 100,
    transparent: !showMesh,
    opacity: showMesh ? 1.0 : 0.3,
    side: THREE.DoubleSide
  })

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      material={material}
      onClick={handleClick}
      castShadow
      receiveShadow
    >
      <meshPhongMaterial
        attach="material"
        color={showMesh ? 0x888888 : 0x333333}
        shininess={100}
        transparent={!showMesh}
        opacity={showMesh ? 1.0 : 0.3}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// Annotation Points Component
function AnnotationPoints() {
  const { clickPoints, showAnnotations } = useAnnotationStore()

  if (!showAnnotations || clickPoints.length === 0) return null

  return (
    <group>
      {clickPoints.map((point, index) => (
        <mesh key={`${point.id}-${index}`} position={[point.x, point.y, point.z || 0]}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color="red" />
        </mesh>
      ))}
    </group>
  )
}

// Segmentation Overlay Component (for displaying SAM results)
function SegmentationOverlay() {
  const { samMasks, showAnnotations } = useAnnotationStore()

  if (!showAnnotations || samMasks.length === 0) return null

  // This would render the segmentation results as overlays
  // For now, we'll return null and implement this when we have SAM integration
  return null
}

// Camera Controller Component
function CameraController() {
  const { cameraPosition, cameraTarget } = useAnnotationStore()
  const { camera } = useThree()

  useEffect(() => {
    camera.position.set(...cameraPosition)
    camera.lookAt(...cameraTarget)
  }, [camera, cameraPosition, cameraTarget])

  return null
}

// Main STL Viewer Component
export function STLViewer({ modelUrl, onPointClick, className = '' }: STLViewerProps) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  const {
    setCameraPosition,
    setCameraTarget,
    addClickPoint,
    activeTool,
    activeLabel
  } = useAnnotationStore()

  // Handle point clicks from 3D model
  const handleModelClick = useCallback((worldPoint: THREE.Vector3, screenPosition: THREE.Vector2) => {
    // Add click point to store
    const clickPoint = {
      id: `point-${Date.now()}`,
      x: worldPoint.x,
      y: worldPoint.y,
      z: worldPoint.z,
      label: activeLabel || 'default',
      confidence: 1.0
    }
    
    addClickPoint(clickPoint)

    // Trigger SAM segmentation via callback
    if (onPointClick) {
      onPointClick(worldPoint, screenPosition)
    }
  }, [addClickPoint, activeLabel, onPointClick])

  // Handle camera changes
  const handleCameraChange = useCallback((event: any) => {
    const camera = event.target
    setCameraPosition([camera.position.x, camera.position.y, camera.position.z])
    setCameraTarget([camera.target.x, camera.target.y, camera.target.z])
  }, [setCameraPosition, setCameraTarget])

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className}`}>
      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
          <div className="text-white text-lg">Loading 3D model...</div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 z-10">
          <div className="text-red-400 text-lg">Error: {error}</div>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        shadows
        camera={{ position: [0, 0, 10], fov: 50 }}
        onCreated={() => setIsLoading(false)}
        onError={(error) => setError(error.message)}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* Environment */}
        <Environment preset="studio" />

        {/* Camera Controller */}
        <CameraController />

        {/* 3D Model */}
        <STLModel url={modelUrl} onPointClick={handleModelClick} />

        {/* Annotation Points */}
        <AnnotationPoints />

        {/* Segmentation Overlay */}
        <SegmentationOverlay />

        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          onChange={handleCameraChange}
          target={[0, 0, 0]}
        />
      </Canvas>

      {/* Tool Cursor Indicator */}
      {activeTool === 'segment' && (
        <div className="absolute top-4 left-4 bg-blue-600 text-white px-3 py-1 rounded-full text-sm z-20">
          Click to segment
        </div>
      )}
    </div>
  )
}

export default STLViewer 