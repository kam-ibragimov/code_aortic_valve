import os
from slicer_project_generator.scripts.gui import CaseSelector
from slicer_project_generator.scripts.generator import ProjectGenerator
from slicer_project_generator.scripts.utils import get_available_cases, json_reader

def main(data_path):
    dict_all_case_path = os.path.join(data_path, "dict_all_case.json")
    output_folder = os.path.join(data_path, "result", "cases_visualization")
    original_img_folder = os.path.join(data_path, "image_nii")
    original_aorta_mask_folder = os.path.join(data_path, "mask_aorta_segment")
    pred_aorta_mask_folder = os.path.join(data_path,  "nnUNet_folder", "nnUNet_test", "Dataset421_AortaSegment")
    gh_lines_pred_mask_folder = os.path.join(data_path, "nnUNet_folder", "nnUNet_test", "Dataset424_GhLines")
    ci_lines_pred_mask_folder = os.path.join(data_path, "nnUNet_folder", "nnUNet_test", "Dataset425_CiLines")
    br_2d_pred_mask_folder = os.path.join(data_path, "nnUNet_folder", "nnUNet_test", "Dataset426_2DBasalRing")
    dict_all_case = json_reader(dict_all_case_path)
    # get a list of cases
    cases = list(dict_all_case.keys())

    # show the selection window
    selector = CaseSelector(cases, gh_lines_pred_mask_folder)
    selected_case, gh_lines_pred_mask_file = selector.run()

    if not selected_case:
        print("Case not selected, exit...")
        return

    def _pred_file(folder):
        path = os.path.join(folder, selected_case + ".nii.gz")
        return path if os.path.exists(path) else None

    # generate the project
    generator = ProjectGenerator(case_name=selected_case,
                                 output_folder=output_folder,
                                 original_img_folder=original_img_folder,
                                 original_aorta_mask_folder=original_aorta_mask_folder,
                                 case_data=dict_all_case[selected_case],
                                 gh_lines_pred_mask_file=gh_lines_pred_mask_file,
                                 ci_lines_pred_mask_file=_pred_file(ci_lines_pred_mask_folder),
                                 br_2d_pred_mask_file=_pred_file(br_2d_pred_mask_folder),
                                 pred_aorta_mask_file=_pred_file(pred_aorta_mask_folder))
    project_file = generator.generate()

    print(f"Project created: {project_file}")

if __name__ == "__main__":
    data_path = "C:/Users/Kamil/Aortic_valve/data/"
    main(data_path)