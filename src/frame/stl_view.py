from PyQt5.QtWidgets import QDialog, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk

class STLView(QDialog):
    def __init__(self, stl_file, parent=None):
        super(STLView, self).__init__(parent)
        self.setWindowTitle("3D STL Viewer - Dialog")
        self.setGeometry(100, 100, 800, 600)
        self.stl_file = stl_file

        layout = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)

        self.ren = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()

        self.loadSTL()

    def loadSTL(self):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(self.stl_file)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.ren.AddActor(actor)
        self.ren.SetBackground(0.4, 0.4, 0.4)
        self.ren.ResetCamera()  # Make sure the camera is properly positioned
        self.vtk_widget.GetRenderWindow().Render()  # Trigger the initial render

        # Start the interactor after the render window is set up
        self.iren.Start()

    def closeEvent(self, event):
        self.iren.ExitCallback()
        event.accept()
