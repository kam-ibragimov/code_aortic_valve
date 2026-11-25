import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from data_preprocessing.mask_cutting import MARGIN_MM, cut_mask_between_planes
from data_preprocessing.text_worker import add_info_logging
from scipy.ndimage import label


def convert_stl_to_mask_nii(stl_file, reference_image_file):
    """Convert an STL file to a binary mask that matches the reference image grid."""

    def vtk_image_to_sitk(vtk_image):
        """Convert vtkImageData to SimpleITK Image."""
        # Step 1: Extract VTK image dimensions, spacing, and origin
        dimensions = vtk_image.GetDimensions()
        spacing = vtk_image.GetSpacing()
        origin = vtk_image.GetOrigin()
        np.bool = np.bool_
        # Step 2: Get VTK image data as a NumPy array
        vtk_array = vtk_image.GetPointData().GetScalars()  # Get the VTK data array
        np_array = vtk_to_numpy(vtk_array).astype(np.uint8)  # Convert VTK array to NumPy array
        np_array = np_array.reshape(dimensions[::-1])  # Reshape to match SimpleITK's z, y, x order

        # Step 3: Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(np_array)
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)

        # Step 4: Set Direction (assuming identity if not available)
        # VTK doesn't provide direction, so we often assume identity, or you can define it as needed.
        sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        return sitk_image

    # Step 1: Load the STL file using VTK
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    poly_data = reader.GetOutput()

    # Step 2: Get the reference image properties
    reference_image = sitk.ReadImage(reference_image_file)
    ref_spacing = reference_image.GetSpacing()
    ref_origin = reference_image.GetOrigin()
    ref_direction = reference_image.GetDirection()
    ref_size = reference_image.GetSize()

    # Step 3: Set up VTK to match the reference image grid
    # Define VTK image with the same dimensions and spacing
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(ref_size)
    vtk_image.SetSpacing(ref_spacing)
    vtk_image.SetOrigin(ref_origin)

    # Step 4: Use VTK's PolyDataToImageStencil to convert STL to a mask
    # Create a stencil from the STL poly data
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly_data)
    stencil.SetOutputOrigin(vtk_image.GetOrigin())
    stencil.SetOutputSpacing(vtk_image.GetSpacing())
    stencil.SetOutputWholeExtent(vtk_image.GetExtent())
    stencil.Update()

    # Step 5: Convert stencil to binary mask in VTK
    # Initialize an empty binary mask in VTK and apply stencil to it
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(stencil.GetOutputPort())
    stencil_to_image.SetInsideValue(1)  # Set voxel value inside the STL surface
    stencil_to_image.SetOutsideValue(0)  # Set voxel value outside the STL surface
    stencil_to_image.Update()

    return vtk_image_to_sitk(stencil_to_image.GetOutput())


def _largest_component(mask):
    labeled, _ = label(mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # фон
    return (labeled == sizes.argmax()).astype(np.uint8)


def convert_and_cut_stl(stl_path, image_nii_path, mask_nii_path, nadirs, commissures, margin=MARGIN_MM):
    """Маска из STL, обрезанная плоскостями надиров и комиссур (крупнейшая компонента). Если
    обрезать нельзя, маска не сохраняется: необрезанная GT-маска испортила бы обучение.
    Возвращает True, если маска сохранена."""
    mask_image = convert_stl_to_mask_nii(stl_path, image_nii_path)
    cut = cut_mask_between_planes(mask_image, nadirs, commissures, margin)
    if cut is None or not cut.any():
        reason = 'degenerate planes' if cut is None else 'empty after cutting'
        add_info_logging(f'Aorta GT mask not saved: {mask_nii_path} from {stl_path} ({reason}).',
                         'work_logger')
        return False

    new_image = sitk.GetImageFromArray(_largest_component(cut))
    new_image.CopyInformation(mask_image)
    sitk.WriteImage(new_image, mask_nii_path)
    return True
