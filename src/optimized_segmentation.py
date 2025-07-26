import numpy as np
import vtk
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.neighbors import KDTree
import time
from functools import partial

class OptimizedSegmentation:
    """
    Optimized version of 3D point mapping with multiple performance strategies
    """
    
    def __init__(self, renderer, vtk_widget):
        self.ren = renderer
        self.vtk_widget = vtk_widget
        self.original_point_cloud = None
        self.kdtree = None
        
    def build_spatial_index(self, point_cloud):
        """Build KD-tree for fast spatial queries"""
        print("Building spatial index...")
        start_time = time.time()
        self.original_point_cloud = point_cloud
        self.kdtree = KDTree(point_cloud)
        print(f"Spatial index built in {time.time() - start_time:.2f}s")
    
    def batch_pixel_to_world(self, pixel_batch):
        """Convert batch of pixels to world coordinates using VTK picker"""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.1)
        
        world_coords = []
        for pixel_coord in pixel_batch:
            x, y = pixel_coord[0], pixel_coord[1]
            picker.Pick(x, y, 0, self.ren)
            world_coords.append(picker.GetPickPosition())
        
        return np.array(world_coords)
    
    def find_nearby_points_vectorized(self, world_coords_batch, radius=0.5):
        """Find nearby points using KD-tree (much faster than brute force)"""
        if self.kdtree is None:
            raise ValueError("Spatial index not built. Call build_spatial_index first.")
        
        all_indices = []
        for world_coord in world_coords_batch:
            # Query KD-tree for points within radius
            indices = self.kdtree.query_radius([world_coord], r=radius)[0]
            all_indices.extend(indices)
        
        return np.unique(all_indices)
    
    def process_pixel_chunk_optimized(self, pixel_chunk_args):
        """Process a chunk of pixels - designed for multithreading"""
        pixel_chunk, height, radius = pixel_chunk_args
        
        # Adjust y-coordinates
        adjusted_chunk = pixel_chunk.copy()
        adjusted_chunk[:, 1] = height - adjusted_chunk[:, 1]
        
        # Convert pixels to world coordinates (VTK operations must be on main thread)
        world_coords = self.batch_pixel_to_world(adjusted_chunk)
        
        # Find nearby points using spatial index
        point_indices = self.find_nearby_points_vectorized(world_coords, radius)
        
        return point_indices
    
    def get_3d_points_optimized_v1_multithreading(self, pixel_coords, original_point_cloud, 
                                                   original_faces, radius=0.5, max_workers=4):
        """
        STRATEGY 1: Multithreading + KD-tree optimization
        Best for: Medium datasets, balanced CPU usage
        """
        print(f"Processing {len(pixel_coords)} pixels with multithreading (v1)...")
        start_time = time.time()
        
        # Build spatial index
        self.build_spatial_index(original_point_cloud)
        
        height = self.vtk_widget.GetRenderWindow().GetSize()[1]
        
        # Sample every 20th pixel (matching original behavior)
        sampled_pixels = pixel_coords[::20]
        print(f"Sampled to {len(sampled_pixels)} pixels")
        
        # Split into chunks for multithreading
        chunk_size = max(len(sampled_pixels) // max_workers, 100)
        pixel_chunks = [sampled_pixels[i:i + chunk_size] 
                       for i in range(0, len(sampled_pixels), chunk_size)]
        
        # Process chunks in parallel
        chunk_args = [(chunk, height, radius) for chunk in pixel_chunks]
        
        all_point_indices = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.process_pixel_chunk_optimized, chunk_args)
            
            for indices in results:
                all_point_indices.update(indices)
        
        # Extract final points
        points_in_region = original_point_cloud[list(all_point_indices)]
        
        print(f"Multithreading optimization completed in {time.time() - start_time:.2f}s")
        return points_in_region
    
    def get_3d_points_optimized_v2_vectorized(self, pixel_coords, original_point_cloud, 
                                             original_faces, radius=0.5):
        """
        STRATEGY 2: Pure vectorization + smart sampling
        Best for: Large datasets, maximum speed
        """
        print(f"Processing {len(pixel_coords)} pixels with vectorization (v2)...")
        start_time = time.time()
        
        # Build spatial index
        self.build_spatial_index(original_point_cloud)
        
        height = self.vtk_widget.GetRenderWindow().GetSize()[1]
        
        # Smart sampling: Take every nth pixel based on mask density
        sampling_rate = max(1, len(pixel_coords) // 1000)  # Max 1000 samples
        sampled_pixels = pixel_coords[::sampling_rate]
        sampled_pixels[:, 1] = height - sampled_pixels[:, 1]
        
        print(f"Smart sampled to {len(sampled_pixels)} pixels (rate: {sampling_rate})")
        
        # Batch process all pixels at once
        world_coords = self.batch_pixel_to_world(sampled_pixels)
        
        # Vectorized spatial query
        all_indices = set()
        for world_coord in world_coords:
            indices = self.kdtree.query_radius([world_coord], r=radius)[0]
            all_indices.update(indices)
        
        points_in_region = original_point_cloud[list(all_indices)]
        
        print(f"Vectorized optimization completed in {time.time() - start_time:.2f}s")
        return points_in_region
    
    def get_3d_points_optimized_v3_adaptive(self, pixel_coords, original_point_cloud, 
                                           original_faces, radius=0.5):
        """
        STRATEGY 3: Adaptive sampling + caching
        Best for: Interactive use, progressive refinement
        """
        print(f"Processing {len(pixel_coords)} pixels with adaptive sampling (v3)...")
        start_time = time.time()
        
        # Build spatial index
        self.build_spatial_index(original_point_cloud)
        
        height = self.vtk_widget.GetRenderWindow().GetSize()[1]
        
        # Multi-level adaptive sampling
        points_in_region = set()
        
        # Level 1: Coarse sampling (every 50th pixel)
        coarse_pixels = pixel_coords[::50]
        coarse_pixels[:, 1] = height - coarse_pixels[:, 1]
        world_coords_coarse = self.batch_pixel_to_world(coarse_pixels)
        
        for world_coord in world_coords_coarse:
            indices = self.kdtree.query_radius([world_coord], r=radius)[0]
            points_in_region.update(indices)
        
        # Level 2: Medium sampling in sparse areas (every 20th pixel)
        medium_pixels = pixel_coords[::20]
        medium_pixels[:, 1] = height - medium_pixels[:, 1]
        world_coords_medium = self.batch_pixel_to_world(medium_pixels)
        
        for world_coord in world_coords_medium:
            indices = self.kdtree.query_radius([world_coord], r=radius * 0.7)[0]
            points_in_region.update(indices)
        
        points_array = original_point_cloud[list(points_in_region)]
        
        print(f"Adaptive optimization completed in {time.time() - start_time:.2f}s")
        return points_array


# Performance comparison function
def benchmark_optimizations(segmentation_tool, pixel_coords, original_point_cloud, original_faces):
    """Compare all optimization strategies"""
    
    optimizer = OptimizedSegmentation(segmentation_tool.ren, segmentation_tool.vtk_widget)
    
    methods = [
        ("Original Method", segmentation_tool.get_3d_points_in_region),
        ("V1: Multithreading", optimizer.get_3d_points_optimized_v1_multithreading),
        ("V2: Vectorized", optimizer.get_3d_points_optimized_v2_vectorized),
        ("V3: Adaptive", optimizer.get_3d_points_optimized_v3_adaptive)
    ]
    
    results = {}
    
    for name, method in methods:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        try:
            if name == "Original Method":
                points = method(pixel_coords, original_point_cloud, original_faces)
            else:
                points = method(pixel_coords, original_point_cloud, original_faces)
            
            elapsed = time.time() - start_time
            results[name] = {
                'time': elapsed,
                'points': len(points),
                'success': True
            }
            print(f"✅ {name}: {elapsed:.2f}s, {len(points)} points")
            
        except Exception as e:
            results[name] = {'time': float('inf'), 'points': 0, 'success': False, 'error': str(e)}
            print(f"❌ {name}: Failed - {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    
    for name, result in results.items():
        if result['success']:
            print(f"{name:20}: {result['time']:6.2f}s ({result['points']} points)")
        else:
            print(f"{name:20}: FAILED")
    
    return results 