from .mask3d import Mask3D
from .img2d import Img2D
from typing import List, Union, Tuple
import numpy as np
import vtk


class Drawer3D(Mask3D):
	@staticmethod
	def draw3DvolumesFromMsk(arrs: List[np.ndarray], colors: Union[List[Tuple], Tuple, None] = None, 
		vtk_transform: vtk.vtkTransform = vtk.vtkTransform(),
		bg_color: List[np.uint8] = [51, 77, 102, 255],
		bbox: bool = False, save_to: Union[str, None] = None, show_window = True, window_magnification:int = 1,
		actors: List[vtk.vtkActor] = []):
		"""
		- arrs: list of numpy arrays to be shown (bool array)
		- colors: list of colors (4D) for each array, can be None
		- bbox: draw bounding box
		- vtk_transform: transform apply to the polydata (refer to: https://vtk.org/doc/nightly/html/classvtkTransform.html)
		- show_window: wether to show the render window
		- window_magnification: magnification for saving image file
		- actors: additional vtk actor (e.g. text actor)
		"""

		if colors is None:
			colors = [[255, 125, 64, 255] for _ in range(len(arrs))]

		assert len(arrs) == len(colors), "The length of the arrs should be the same as colors"

		# Render property
		ren = vtk.vtkRenderer()
		ren_win = vtk.vtkRenderWindow()
		ren_win.AddRenderer(ren)
		iren = vtk.vtkRenderWindowInteractor()
		iren.SetRenderWindow(ren_win)
		iren.Initialize()

		vtk_colors = vtk.vtkNamedColors()
		vtk_colors.SetColor("BkgColor", bg_color)

		for i in range(len(arrs)):
			arr = arrs[i]
			color = colors[i]
			data_str = arr.tobytes()
			importer = vtk.vtkImageImport()
			importer.CopyImportVoidPointer(data_str, len(data_str))
			importer.SetDataScalarTypeToUnsignedChar()
			importer.SetNumberOfScalarComponents(1)
			importer.SetDataExtent(0, arr.shape[1]-1, 0, arr.shape[2]-1, 0, arr.shape[0]-1)
			importer.SetWholeExtent(0, arr.shape[1]-1, 0, arr.shape[2]-1, 0, arr.shape[0]-1)


			surface_extractor = vtk.vtkMarchingCubes()
			surface_extractor.SetInputConnection(importer.GetOutputPort())
			surface_extractor.SetValue(0, 1)

			stripper = vtk.vtkStripper()
			stripper.SetInputConnection(surface_extractor.GetOutputPort())

			transform_filter = vtk.vtkTransformPolyDataFilter()
			transform_filter.SetInputConnection(stripper.GetOutputPort())
			transform_filter.SetTransform(vtk_transform)
			transform_filter.Update()

			surface_mapper = vtk.vtkPolyDataMapper()
			surface_mapper.SetInputConnection(transform_filter.GetOutputPort())
			surface_mapper.ScalarVisibilityOff()
			vtk_colors.SetColor("SurfaceColor", color)

			surface = vtk.vtkActor()
			surface.SetMapper(surface_mapper)
			surface.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d("SurfaceColor"))

			ren.AddActor(surface)

			for actor_ in actors:
				ren.AddActor(actor_)			

			if bbox:
				# An outline provides context around the data.
				ouline_data = vtk.vtkOutlineFilter()
				ouline_data.SetInputConnection(importer.GetOutputPort())

				map_outline = vtk.vtkPolyDataMapper()
				map_outline.SetInputConnection(ouline_data.GetOutputPort())

				outline = vtk.vtkActor()
				outline.SetMapper(map_outline)
				outline.GetProperty().SetColor(vtk_colors.GetColor3d("Black"))

				ren.AddActor(outline)

		ren.ResetCamera()
		ren.SetBackground(vtk_colors.GetColor3d("BkgColor"))


		if save_to:
			im = vtk.vtkWindowToImageFilter()
			writer = vtk.vtkPNGWriter()
			im.SetInput(ren_win)
			im.SetScale(window_magnification)
			im.Update()
			writer.SetInputConnection(im.GetOutputPort())
			writer.SetFileName(save_to)
			writer.Write()
		ren_win.Render()
		# print(ren_win.GetSize())
		if show_window:
			iren.Start()