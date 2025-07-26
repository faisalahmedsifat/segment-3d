import Link from 'next/link'
import { FileText, Zap, Users, BarChart3, Upload, Eye } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      {/* Header */}
      <header className="px-4 lg:px-6 h-14 flex items-center border-b">
        <div className="flex items-center justify-center">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mr-2">
            <FileText className="h-5 w-5 text-white" />
          </div>
          <span className="font-bold text-xl">Segment3D</span>
        </div>
        <nav className="ml-auto flex gap-4 sm:gap-6">
          <Link className="text-sm font-medium hover:underline underline-offset-4" href="/login">
            Login
          </Link>
          <Link className="text-sm font-medium hover:underline underline-offset-4" href="/register">
            Sign Up
          </Link>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                SAM-Powered 3D/2D Annotation Platform
              </h1>
              <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
                Professional annotation tool leveraging Meta's Segment Anything Model for intelligent 
                3D model and 2D image segmentation. Perfect for creating ML training datasets.
              </p>
            </div>
            <div className="space-x-4">
              <Link
                className="inline-flex h-9 items-center justify-center rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow transition-colors hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-950 disabled:pointer-events-none disabled:opacity-50"
                href="/register"
              >
                Get Started
              </Link>
              <Link
                className="inline-flex h-9 items-center justify-center rounded-md border border-gray-200 bg-white px-4 py-2 text-sm font-medium shadow-sm transition-colors hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-950 disabled:pointer-events-none disabled:opacity-50"
                href="/demo"
              >
                View Demo
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-50">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
                Powerful Annotation Features
              </h2>
              <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Everything you need for professional 3D and 2D annotation workflows
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-3 lg:gap-12">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-blue-100 p-3">
                <Zap className="h-6 w-6 text-blue-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">AI-Powered Segmentation</h3>
                <p className="text-gray-500">
                  Leverage Meta's SAM model for intelligent click-to-segment annotation 
                  on both 3D models and 2D images.
                </p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-green-100 p-3">
                <Users className="h-6 w-6 text-green-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">Team Collaboration</h3>
                <p className="text-gray-500">
                  Real-time collaboration with role-based access, task assignment, 
                  and quality control workflows.
                </p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-purple-100 p-3">
                <BarChart3 className="h-6 w-6 text-purple-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">Project Management</h3>
                <p className="text-gray-500">
                  Comprehensive project tracking, analytics, and export to 
                  COCO, YOLO, Pascal VOC formats.
                </p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-orange-100 p-3">
                <Upload className="h-6 w-6 text-orange-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">Multi-Format Support</h3>
                <p className="text-gray-500">
                  Support for STL, PLY, OBJ 3D models and PNG, JPG, TIFF images 
                  with cloud storage integration.
                </p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-red-100 p-3">
                <Eye className="h-6 w-6 text-red-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">3D Visualization</h3>
                <p className="text-gray-500">
                  Advanced Three.js-based 3D viewer with interactive annotation 
                  tools and multi-angle support.
                </p>
              </div>
            </div>
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="rounded-lg bg-teal-100 p-3">
                <FileText className="h-6 w-6 text-teal-600" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold">Quality Control</h3>
                <p className="text-gray-500">
                  Inter-annotator agreement metrics, validation workflows, 
                  and version control for datasets.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-12 md:py-24 lg:py-32">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
                Ready to Start Annotating?
              </h2>
              <p className="mx-auto max-w-[600px] text-gray-500 md:text-xl">
                Join researchers and teams worldwide using Segment3D for their ML dataset creation.
              </p>
            </div>
            <div className="space-x-4">
              <Link
                className="inline-flex h-9 items-center justify-center rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow transition-colors hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-950 disabled:pointer-events-none disabled:opacity-50"
                href="/register"
              >
                Create Free Account
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t">
        <p className="text-xs text-gray-500">
          © 2024 Segment3D Platform. From desktop to cloud - transforming 3D annotation.
        </p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link className="text-xs hover:underline underline-offset-4" href="/docs">
            Documentation
          </Link>
          <Link className="text-xs hover:underline underline-offset-4" href="/api">
            API
          </Link>
          <Link className="text-xs hover:underline underline-offset-4" href="/support">
            Support
          </Link>
        </nav>
      </footer>
    </div>
  )
} 