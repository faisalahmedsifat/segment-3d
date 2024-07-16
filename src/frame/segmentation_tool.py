import time


import cv2
import open3d as o3d
from PIL import Image
import io

import numpy as np
import torch

import vtkmodules.all as vtk

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QMainWindow,

                             QPushButton, QSizePolicy, QSpacerItem,

                             QVBoxLayout, QWidget)

from segment_anything import SamPredictor, sam_model_registry

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


import matplotlib.pyplot as plt

from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QLabel
from stl import mesh


from vtk.util import numpy_support

import trimesh

from vtkmodules.all import vtkPropPicker

from PyQt5.QtCore import QPoint
import os
from frame.stl_view import STLView


class SegmentationTool(QMainWindow):

    def __init__(self, stl_file):

        super().__init__()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = self.getTemporaryDirectory()
        self.output_dir = os.path.join(self.base_dir, "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.depth = 0.0

        button_style = """

            QPushButton {

                background-color: #007BFF;

                border: none;

                color: white;

                padding: 10px 20px;

                text-align: center;

                text-decoration: none;

                display: inline-block;

                font-size: 16px;

                border-radius: 5px;

            }

            QPushButton:hover {

                background-color: #0056b3;

            }

            """

        print("Creating SAM...")

        # the checkpoint of the model
        self.sam_checkpoint = "../model_checkpoint/sam_vit_b_01ec64.pth"

        self.sam = sam_model_registry["vit_b"](checkpoint=self.sam_checkpoint)

        self.sam.to(device="cuda")

        self.predictor = SamPredictor(self.sam)

        self.setGeometry(100, 100, 800, 600)

        self.setWindowTitle("3D Segmentation Tool" + " - " + stl_file)

        self.central_widget = QWidget(self)

        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        self.sidebar = QWidget()

        self.sidebar_layout = QVBoxLayout(self.sidebar)

        self.interact_button = QPushButton("Segment", self)

        self.interact_button.clicked.connect(self.segment_anything)

        self.interact_button.setStyleSheet(button_style)

        self.sidebar_layout.addWidget(self.interact_button)

        self.hide_button = QPushButton("Hide Model", self)

        self.hide_button.clicked.connect(self.hide_model)

        self.hide_button.setStyleSheet(button_style)

        self.sidebar_layout.addWidget(self.hide_button)

        # self.clear_button = QPushButton("Clear", self)

        # self.clear_button.clicked.connect(self.clear)

        # self.clear_button.setStyleSheet(button_style)

        # self.sidebar_layout.addWidget(self.clear_button)

        # self.screenshot_button = QPushButton("Screenshot", self)

        # self.screenshot_button.clicked.connect(self.clear)

        # self.screenshot_button.setStyleSheet(button_style)

        # self.screenshot_button.clicked.connect(self.screenshot)

        # self.sidebar_layout.addWidget(self.screenshot_button)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum,
                             QSizePolicy.Expanding)

        self.sidebar_layout.addItem(spacer)

        self.sidebar.setFixedWidth(200)

        self.vtk_widget = QVTKRenderWindowInteractor(self)

        self.main_layout.addWidget(self.vtk_widget)

        self.hbox_layout = QHBoxLayout()

        self.hbox_layout.addWidget(self.sidebar)

        self.hbox_layout.addWidget(self.vtk_widget)

        self.main_layout.addLayout(self.hbox_layout)

        self.ren = vtk.vtkRenderer()

        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)

        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.seg_actor = vtk.vtkActor()

        self.reader = vtk.vtkSTLReader()
        
        self.seg_reader = vtk.vtkSTLReader()

        self.reader.SetFileName(stl_file)

        self.mapper = vtk.vtkPolyDataMapper()
        
        self.seg_mapper = vtk.vtkPolyDataMapper()

        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        self.actor = vtk.vtkActor()

        self.actor.SetMapper(self.mapper)

        self.polydata = self.reader.GetOutput()

        self.ren.AddActor(self.actor)

        self.ren.SetBackground(0.4, 0.4, 0.4)

        self.iren.Initialize()

        self.showMaximized()

        self.iren.RemoveObservers("RightButtonPressEvent")

        self.iren.AddObserver("RightButtonPressEvent", self.get_pointer_coords)

        self.selected_point_actors = []

        self.mask_actor = None

        self.camera_position = []

        self.camera_focal_point = []
        self.camera_view_up = []
        
        self.point_actor = None

        # TODO
        self.mask_image = None
        # TODO
        self.mask = None

        self.mask_label = None

        self.seg_point = []

        ## TODO
        self.screenshots = []
        
        self.clicked_points = []
        self.labels = []
        self.screen_coordinates = []

        self.concatenated_seg_point = None
        self.loaded_seg_stl = False
        
    def getTemporaryDirectory(self):
        temp_dir = os.path.join(self.base_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return temp_dir

    def hide_model(self):
        if self.actor.GetVisibility() == 1:
            self.actor.VisibilityOff()
            if self.loaded_seg_stl:
                self.seg_actor.VisibilityOn()

        else:
            self.actor.VisibilityOn()
            if self.loaded_seg_stl:
                self.seg_actor.VisibilityOff()

            
            print("rendering actor visibility else: ", self.actor.GetVisibility())
            
        # if(self.loaded_seg_stl != False and self.seg_actor.GetVisibility() == 0 ):
        #         self.seg_actor.VisibilityOn()
                
        #         print("rendering seg actor visibility if: ", self.seg_actor.GetVisibility())
                
        # else:
        #         self.seg_actor.VisibilityOff()

        #         print("rendering seg actor visibility else: ", self.seg_actor.GetVisibility())
        self.vtk_widget.GetRenderWindow().Render()
    def segment_anything(self):

        for screenshot, click_point, label in zip(self.screenshots, self.clicked_points, self.labels):
            self.generate_mask(click_point, label, screenshot)

        i = 0
        for camera_position, camera_focal_point, camera_view_up, screen_coordinate in zip(self.camera_position, self.camera_focal_point, self.camera_view_up, self.screen_coordinates):
            self.test_function(
                camera_position, camera_focal_point, camera_view_up)
            original_point_cloud, faces = self.generate_point_clouds()

            np.save(self.temp_dir+"/original_point_cloud.npy", original_point_cloud)
            np.save(self.temp_dir+"/faces.npy", faces)
            points_in_region = self.get_3d_points_in_region(
                screen_coordinate, original_point_cloud, faces, name=self.temp_dir+"/points_in_region-"+str(i)+".npy")
            i += 1

        self.concatenated_seg_point = np.concatenate(self.seg_point, axis=0)
        print("concatenated seg point shape ",
              self.concatenated_seg_point.shape)
        # save the concatenated point cloud as a numpy array
        np.save(self.temp_dir+"/concatenated_seg_point.npy", self.concatenated_seg_point)

        # # convert the concatenated point cloud to a stl file
        self.save_stl_file()

        print("SAM Created!")
        self.load_seg_stl()

        print("="*80)
        # self.combine_point_clouds()
        self.show_stl_dialog(os.path.join(self.output_dir , "seg_model.stl"))
        
    def show_stl_dialog(self, stl_file):
        self.stl_dialog = STLView(stl_file, self)
        self.stl_dialog.exec_()

    def load_seg_stl(self, stl_file="seg_model.stl"):
        stl_file = os.path.join(self.output_dir, stl_file)
        # load rhw segmented stl file
        self.seg_reader.SetFileName(stl_file)
        self.seg_mapper.SetInputConnection(self.seg_reader.GetOutputPort())
        self.seg_actor.SetMapper(self.seg_mapper)
        self.seg_actor.GetProperty().SetColor(0.9, 0, 0)
        self.seg_actor.GetProperty().SetPointSize(5)
        self.ren.AddActor(self.seg_actor)
        self.vtk_widget.GetRenderWindow().Render()
        
        
        self.loaded_seg_stl = True
        # self.hide_model()
        
    def get_3d_points_in_region(self, pixel_coords, original_point_cloud, original_faces, name="points_in_region.npy"):
        self.clear()
        # Create lists to store the 3D points and faces in the region
        # points_in_region = []
        points_in_region = set()
        # faces_in_region = []
        faces_in_region = set()

        point_indices = set()

        height = self.vtk_widget.GetRenderWindow().GetSize()[1]

        pixel_coords[:, 1] = height - pixel_coords[:, 1]

        # Create a vtkCellPicker
        picker = vtk.vtkCellPicker()

        # Set the picker's tolerance (adjust as needed)
        picker.SetTolerance(0.1)
        length = len(pixel_coords)
        l = 0
        for pixel_coord in pixel_coords:
            if l % 20 == 0:
                # Convert pixel coordinates to VTK-style coordinates
                x, y = pixel_coord[0], pixel_coord[1]

                # Use the picker to get 3D world coordinates
                picker.Pick(x, y, 0, self.ren)

                # Get the 3D world coordinates
                world_coords = picker.GetPickPosition()

                # Calculate the Euclidean distances between world_coords and all points in the cloud
                distances = np.linalg.norm(
                    original_point_cloud - world_coords, axis=1)

                # Find the indices of the closest points
                closest_point_indices = np.where(distances < 0.5)[0]

                print("running ", l, "th iteration out of ",
                      length, " iterations")
                # Store the 3D points and faces in the region
                for point_index in closest_point_indices:
                    points_in_region.add(
                        tuple(original_point_cloud[point_index]))
                    # You'll need to find the faces connected to this point in your original mesh
                    # Assuming you have an array of faces in the same order as the points
                    faces_in_region.add(tuple(original_faces[point_index]))
                    # extract the faces connected to the point

            l += 1

        points_in_region = [list(point) for point in points_in_region]
        # faces_in_region = [list(face) for face in faces_in_region]
        point_indices = list(point_indices)

        self.seg_point.append(points_in_region)

        # convert the points_in_region and faces_in_region to numpy arrays
        points_in_region = np.array(points_in_region)
        # self.seg_point = np.array(self.seg_point)
        # faces_in_region = np.array(faces_in_region)

        print("points in region shape ", np.array(points_in_region).shape)

        # save points_in_region as numpy array
        np.save(name, points_in_region)
        return points_in_region

    def clear(self):

        if len(self.selected_point_actors) != 0:

            for actor in self.selected_point_actors:

                self.ren.RemoveActor(actor)

            self.vtk_widget.GetRenderWindow().Render()

            self.selected_point_actor = []

        if self.mask_label:

            self.mask_label.close()

            self.mask_label = None

        if self.mask_actor:

            self.ren.RemoveActor(self.mask_actor)

            self.vtk_widget.GetRenderWindow().Render()

            self.mask_actor = None

    def screenshot(self, save_path="./screenshot.jpg"):

        w2if = vtk.vtkWindowToImageFilter()

        w2if.SetInput(self.vtk_widget.GetRenderWindow())

        w2if.Update()

        self.camera_position.append(self.ren.GetActiveCamera().GetPosition())

        self.camera_focal_point.append(
            self.ren.GetActiveCamera().GetFocalPoint())
        self.camera_view_up.append(self.ren.GetActiveCamera().GetViewUp())

        writer = vtk.vtkPNGWriter()

        writer.SetFileName(save_path)

        writer.SetInputConnection(w2if.GetOutputPort())

        writer.Write()
        return save_path

    def test_function(self, camera_position, camera_focal_point, camera_view_up):
        print("test function")
        self.ren.GetActiveCamera().SetPosition(camera_position)
        self.ren.GetActiveCamera().SetFocalPoint(camera_focal_point)
        self.ren.GetActiveCamera().SetViewUp(camera_view_up)
        self.vtk_widget.GetRenderWindow().Render()

    def get_pointer_coords(self, obj, event):

        click_pos = self.iren.GetEventPosition()

        x_axis = click_pos[0]

        y_axis = click_pos[1]

        print(self.vtk_widget.GetRenderWindow().GetSize())

        height = self.vtk_widget.GetRenderWindow().GetSize()[1]

        y_axis = height - y_axis

        print("height ", height)

        print("click pos ", click_pos)

        label = np.array([1])
        self.screenshots.append(self.screenshot(
            save_path=self.temp_dir+"/screenshot"+str(len(self.screenshots))+".jpg"))

        self.overlay_selected_point(click_pos)

        click_pos = np.array([[x_axis, y_axis]])

        self.clicked_points.append(click_pos)
        self.labels.append(label)

    # def overlay_mask_on_3d_object(self, seg_point, name="seg_point.npy"):

    #     seg_np = np.array(seg_point)
    #     np.save(name, seg_np)

    #     print("seg point shape overlay: ", seg_np.shape)

    #     point_clouds = seg_point

    #     print("point clouds shape ", point_clouds.shape)

    #     vtk_array = numpy_support.numpy_to_vtk(point_clouds)

    #     vtk_points = vtk.vtkPoints()

    #     vtk_points.SetData(vtk_array)

    #     polydata = vtk.vtkPolyData()

    #     polydata.SetPoints(vtk_points)

    #     vertex = vtk.vtkVertexGlyphFilter()

    #     vertex.SetInputData(polydata)

    #     vertex.Update()

    #     print("vertex output port ", vertex.GetOutputPort())

    #     mapper = vtk.vtkPolyDataMapper()

    #     mapper.SetInputConnection(vertex.GetOutputPort())

    #     actor = vtk.vtkActor()

    #     actor.SetMapper(mapper)

    #     actor.GetProperty().SetColor(1, 0, 0)

    #     # make the point size bigger
    #     actor.GetProperty().SetPointSize(5)

    #     self.ren.AddActor(actor)

    #     self.vtk_widget.GetRenderWindow().Render()

        # select the point cloud of the 3d mask on the selected point
        # point_clouds = point_clouds[0]
        # print("point clouds shape overlay3d: ", point_clouds.shape)

        # vtk_array = numpy_support.numpy_to_vtk(point_clouds)

    def overlay_selected_point(self, click_pos):

        picker = vtk.vtkCellPicker()

        picker.SetTolerance(0.01)  # Adjust the tolerance as needed

        picker.PickFromListOn()

        picker.AddPickList(self.actor)

        picker.Pick(click_pos[0], click_pos[1], 0, self.ren)

        print("picker cell id: ", picker.GetCellId())

        if picker.GetCellId() >= 0:

            world_coords = picker.GetPickPosition()

            print("picker world coords: ", world_coords)

            sphere_source = vtk.vtkSphereSource()

            sphere_source.SetCenter(world_coords)

            sphere_source.SetRadius(0.5)  # Adjust the radius as needed

            sphere_mapper = vtk.vtkPolyDataMapper()

            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

            self.point_actor = vtk.vtkActor()

            self.point_actor.SetMapper(sphere_mapper)

            self.point_actor.GetProperty().SetColor(
                1, 0, 0)  # Red color for highlighting

            self.ren.AddActor(self.point_actor)
            self.selected_point_actors.append(self.point_actor)

            self.vtk_widget.GetRenderWindow().Render()

    def generate_mask(self, click_pos, label, image_path):

        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)

        self.mask, scores, logits = self.predictor.predict(click_pos, label)

        self.mask = np.array(self.mask, dtype=np.uint8)

        mask_with_scores = zip(self.mask, scores)

        self.mask, self.score = max(mask_with_scores, key=lambda x: x[1])

        print("mask shape ", self.mask.shape)

        print("score ", self.score)

        self.mask = np.array(self.mask, dtype=np.uint8)

        mask_image_path = self.temp_dir+"/mask.png"  # Choose a suitable path for your mask image

        cv2.imwrite(mask_image_path, self.mask)

        self.overlay_mask(self.mask)

    def overlay_mask(self, mask):

        print("mask shape", mask.shape)

        self.mask_image = self.mask_to_image(
            mask, self.vtk_widget.GetRenderWindow().GetSize())

        print("mask image shape ", self.mask_image.shape)

        qpixmap = self.mask_image_to_qpixmap(self.mask_image)

        self.mask_label = QLabel(self)

        self.mask_label.setPixmap(qpixmap)

        self.mask_label.setGeometry(215, 10, qpixmap.width(), qpixmap.height())

        self.mask_label.setMask(qpixmap.mask())

        mask_label_mask_location = self.mask_label.geometry()

        print("mask label mask location ", mask_label_mask_location)

        screen_coordinates = self.get_screen_coordinates_of_mask_label(
            mask_label_mask_location, self.mask_image, self.mask_label)

        self.screen_coordinates.append(screen_coordinates)

    def get_screen_coordinates_of_mask_label(self, mask_label_mask_location, mask_image, mask_label):

        mask_label_pixel_coordinates = np.argwhere(mask_image)

        mask_label_pixel_coordinates = mask_label_pixel_coordinates[:, [1, 0]]

        """

            return the screen coordinates of the mask label [x, y] of the screen

        """

        return mask_label_pixel_coordinates

    def mask_to_image(self, input_mask, vtk_canvas_size):

        width, height = vtk_canvas_size[0], vtk_canvas_size[1]

        threshold = 0.0

        mask = (input_mask > threshold).astype(
            np.uint8)  # Convert the Boolean mask to uint8

        mask_image = np.zeros((height, width, 4), dtype=np.uint8)

        r, g, b, a = 0, 114, 189, 255  # Mask's blue color

        mask_image[..., 0] = r * mask

        mask_image[..., 1] = g * mask

        mask_image[..., 2] = b * mask

        mask_image[..., 3] = a * mask

        self.mask_image = mask_image

        cv2.imwrite(self.temp_dir+ "/mask_image.png", mask_image)

        return mask_image

    def mask_image_to_qpixmap(self, mask_image):

        qimage = QImage(
            mask_image.data, mask_image.shape[1], mask_image.shape[0], QImage.Format_RGBA8888)

        qpixmap = QPixmap.fromImage(qimage)

        return qpixmap

    def generate_point_clouds(self):
        # 1. Create empty lists to store the points and faces in the point cloud.
        points = []
        faces = []

        # 2. Get the output of the STL reader (which represents the 3D model).
        model = self.reader.GetOutput()

        # 3. Iterate over the points and cells (faces) in the model.
        for pointId in range(model.GetNumberOfPoints()):
            # Get the 3D coordinates of the point.
            x, y, z = model.GetPoint(pointId)
            point = [x, y, z]
            points.append(point)

        for cellId in range(model.GetNumberOfCells()):
            # Get the cell (face) by its ID.
            cell = model.GetCell(cellId)
            # Get the vertices of the cell (face).
            cell_points = [model.GetPoint(cell.GetPointId(i))
                           for i in range(cell.GetNumberOfPoints())]
            # Convert the vertices to a list of point IDs.
            face = [model.FindPoint(point) for point in cell_points]
            faces.append(face)

        # 4. Convert the lists of points and faces to NumPy arrays or other suitable data structures.
        points = np.array(points)
        faces = np.array(faces)

        return points, faces

    def ray_cast(self, screen_coordinates, picker):

        picker.Pick(screen_coordinates[0], screen_coordinates[1], 0, self.ren)

        world_coords = picker.GetPickPosition()

        # extract the point clouds from the world coordinates

        points = [world_coords[0], world_coords[1], world_coords[2]]

        return points

    def save_stl_file(self, name="seg_model.stl"):
        
        # Load the 3D model data
        model_faces = np.load(self.temp_dir+'/faces.npy')
        model_points = np.load(self.temp_dir+'/original_point_cloud.npy')

        segmented_points = np.load(self.temp_dir+'/concatenated_seg_point.npy')

        # Calculate a mask for the vertices that are part of the segmented point clouds
        mask = np.isin(model_points, segmented_points).all(axis=1)

        # Create a list of faces that reference the vertices in 'segmented_points'
        segmented_faces = []

        for face in model_faces:
            if all(mask[face]):  # Check if all three vertices of the face are in 'segmented_points'
                segmented_faces.append(face)

        # Convert the list of faces to a numpy array
        segmented_faces = np.array(segmented_faces)

        # Assuming you have 'model_faces' and 'segmented_points' as numpy arrays
        # Create a mesh object using 'numpy-stl'
        output_stl_file = os.path.join(self.output_dir, name)
        # output_stl_file = self.output_dir+"/seg_model.stl"
        stl_mesh = mesh.Mesh(
            np.zeros(len(segmented_faces), dtype=mesh.Mesh.dtype))

        # Assign vertices to the mesh using segmented_points
        for i, face in enumerate(segmented_faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = model_points[face[j]]
                print("face", face[j])
                print("segmented_points", model_points[face[j]])

        # Save the mesh as an STL file
        stl_mesh.save(output_stl_file)
