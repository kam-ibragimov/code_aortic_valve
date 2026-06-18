import json
import os
import re
import shutil
from pathlib import Path
from data_preprocessing.text_worker import add_info_logging
from models.landmarking_heart import landmarking_computeMeasurements_simplified
from morpho_gcn import MorphoGCN_Trainer, nnUnet_CandidatePointGenerator, CandidateRobustness
from plots_data.plots import creat_box_plot, plot_robustness_noise

# ---------------------------------------------------------------------------
# Project-specific configuration
# ---------------------------------------------------------------------------

_LANDMARK_NAMES = ["R", "L", "N", "RLC", "RNC", "LNC"]

_COHORT_COLORS = {
    "German (HOM…)": "#1f77b4",
    "Slovenian norm (n…)": "#ff7f0e",
    "Slovenian pathology (p…)": "#2ca02c",
    "Other": "#7f7f7f",
}

_LENGTH_MEASUREMENTS = [
    'BR_perimeter', 'BR_max', 'BR_min', 'BR_diameter',
    'IC_R', 'IC_L', 'IC_N', 'IC_distance',
    'RL_comm_height', 'RN_comm_height', 'LN_comm_height', 'mean_comm_heigh',
    'ST_perimeter', 'ST_max', 'ST_min', 'ST_diameter',
    'commissural_diameter', 'centroid_valve_height',
]

_ANGLE_MEASUREMENTS = [
    'RL_angle', 'RN_angle', 'LN_angle',
    'R_flat_angle', 'L_flat_angle', 'N_flat_angle',
    'R_vertical_angle', 'L_vertical_angle', 'N_vertical_angle',
    'mean_vertical_angle', 'BR_C_plane_angle',
]

_MEASUREMENT_GROUPS = {
    "length_mm": _LENGTH_MEASUREMENTS,
    "angle_deg": _ANGLE_MEASUREMENTS,
}


def _measurement_fn(landmarks):
    return landmarking_computeMeasurements_simplified(landmarks).get_all_metrics()


def _cohort_fn(case):
    if case.startswith("HOM"):
        return "German (HOM…)"
    if case.startswith("n"):
        return "Slovenian norm (n…)"
    if case.startswith("p"):
        return "Slovenian pathology (p…)"
    return "Other"


class _AorticMorphoGCN_Trainer(MorphoGCN_Trainer):
    """Project adapter: adds creat_box_plot calls after core comparison methods."""

    def _save_comparison_results(self, df, result_folder):
        super()._save_comparison_results(df, result_folder)
        creat_box_plot(result_folder, "gnn_vs_center_comparison.csv")

    def run_ablation_from_csv(self, gnn_results_csv, ablation_candidates_json_file, ablation_result_folder):
        super().run_ablation_from_csv(gnn_results_csv, ablation_candidates_json_file, ablation_result_folder)
        creat_box_plot(ablation_result_folder, "gnn_vs_center_comparison.csv",
                       method3_col="abs_err_ablation", method3_label="nnUNet avg")

    def compare_interobserver(self, testing_folder, candidates_1set_json, candidates_2set_json, result_folder):
        super().compare_interobserver(testing_folder, candidates_1set_json, candidates_2set_json, result_folder)
        creat_box_plot(result_folder, "interobserver_comparison.csv",
                       method1_col="abs_err_inter", method2_col="abs_err_intra",
                       method1_label="Inter-observer", method2_label="Intra-observer")


def _make_trainer():
    return _AorticMorphoGCN_Trainer(
        feature_names=landmarking_computeMeasurements_simplified.get_measurement_names(),
        measurement_fn=_measurement_fn,
        landmark_names=_LANDMARK_NAMES,
        cohort_fn=_cohort_fn,
        cohort_colors=_COHORT_COLORS,
        measurement_groups=_MEASUREMENT_GROUPS,
    )


# ---------------------------------------------------------------------------
# Project classes
# ---------------------------------------------------------------------------

class GNNProject:

    MIN_DIST_GNN = 1.0
    MIN_DIST_SIMPLE = 1.0

    def __init__(self, result_6_nnunet_folder, gnn_folder, train_test_lists):
        self.result_6_nnunet_folder = result_6_nnunet_folder
        self.gnn_folder = gnn_folder
        self.train_test_lists = train_test_lists

    def landmark_nnUnet_generateCandidates(self, min_dist=None, output_filename='landmark_candidates.json',
                                           include_com=True):
        if min_dist is None:
            min_dist = self.MIN_DIST_GNN
        output_path = os.path.join(self.gnn_folder, output_filename)
        if not os.path.isfile(output_path):
            extractor = nnUnet_CandidatePointGenerator(
                json_path=self.result_6_nnunet_folder + '/dataset.json',
                n_candidates=5,
                min_dist=min_dist,
                threshold=0.15,
                include_com=include_com
            )
            results = extractor.extract_candidate_points(self.result_6_nnunet_folder)
            extractor.save_results(results, output_path)

    def landmark_GNN_train(self):
        trainer = _make_trainer()
        trainer.train_morpho_gcn2(self.gnn_folder, self.gnn_folder + '/data/training')

    def landmark_GNN_test(self, mae_length_threshold=None, max_retries=5):
        trainer = _make_trainer()
        trainer.compare_gnn_vs_center(self.gnn_folder, self.gnn_folder + '/data/testing',
                                      self.gnn_folder + '/landmark_candidates.json',
                                      self.gnn_folder + '/results/',
                                      mae_length_threshold=mae_length_threshold,
                                      max_retries=max_retries)

    def configure_folder(self, json_info_folder):
        def _clear_folder(folder):
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
                return
            for item in folder.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        def _get_file_list(df, series_type, column, suffix, base_path):
            base_path = Path(base_path)
            return [
                base_path / f"{name}{suffix}"
                for name in df[df["type_series"] == series_type][column].dropna()
            ]

        def _copy_img(input_imgs_path, output_folder):
            _clear_folder(output_folder)
            df = self.train_test_lists
            for img_path in input_imgs_path:
                if img_path.name[0] == "H":
                    case_name = df.loc[df["case_name"] == img_path.name[:-5], "used_case_name"].iloc[0]
                    img_path = img_path.with_name(img_path.name.replace("_MJ.json", ".json"))
                    shutil.copy(img_path, output_folder / f"{case_name}.json")
                else:
                    shutil.copy(img_path, output_folder / img_path.name)

        list_train_cases = _get_file_list(self.train_test_lists, "train", "case_name", ".json", json_info_folder)
        list_test_cases = _get_file_list(self.train_test_lists, "test", "case_name", ".json", json_info_folder)
        _copy_img(list_train_cases, Path(self.gnn_folder) / "data" / "training")
        _copy_img(list_test_cases, Path(self.gnn_folder) / "data" / "testing")


def process_gnn(result_6_nnunet_folder, gnn_folder, train_test_lists, json_info_folder, create_ds=False,
                training_mod=False, testing_mod=False,
                mae_length_threshold=None, max_retries=5, include_com=True):

    gnn_worker = GNNProject(result_6_nnunet_folder=result_6_nnunet_folder,
                            gnn_folder=gnn_folder,
                            train_test_lists=train_test_lists)
    if create_ds:
        gnn_worker.configure_folder(json_info_folder=json_info_folder)
        gnn_worker.landmark_nnUnet_generateCandidates(include_com=include_com)
    if training_mod:
        gnn_worker.landmark_GNN_train()
    if testing_mod:
        gnn_worker.landmark_GNN_test(mae_length_threshold=mae_length_threshold, max_retries=max_retries)
    print('Hi')


def process_gnn_robustness(gnn_folder, result_folder,
                           sigma_grid=None, p_grid=None, n_seeds=None):
    sweep_csv = os.path.join(result_folder, "robustness_noise_sweep.csv")

    if not os.path.exists(sweep_csv):
        candidates_json = os.path.join(gnn_folder, "landmark_candidates.json")
        testing_folder = os.path.join(gnn_folder, "data", "testing")

        robustness = CandidateRobustness()
        robustness.run_sweep(
            trainer=_make_trainer(),
            model_folder=gnn_folder,
            reference_landmark_folders=testing_folder,
            candidates_json_path=candidates_json,
            result_folder=result_folder,
            landmark_keys=_LANDMARK_NAMES,
            sigma_grid=sigma_grid,
            p_grid=p_grid,
            n_seeds=n_seeds,
        )

    plot_robustness_noise(robustness_folder=result_folder)


def process_gnn_post_analysis(gnn_folder, gnn_interobserver_folder, json_info_folder,
                               interobserver_txt_folder,
                               result_6_nnunet_folder=None, train_test_lists=None, include_com=True,
                               ablation_mod=True, interobserver_mod=True, robustness_mod=True):
    if result_6_nnunet_folder is not None and train_test_lists is not None:
        gnn_worker = GNNProject(result_6_nnunet_folder=result_6_nnunet_folder,
                                gnn_folder=gnn_folder, train_test_lists=train_test_lists)
        gnn_worker.landmark_nnUnet_generateCandidates(
            min_dist=GNNProject.MIN_DIST_SIMPLE,
            output_filename='landmark_candidates_simple.json',
            include_com=include_com
        )

    if ablation_mod:
        candidates_simple = os.path.join(gnn_folder, "landmark_candidates_simple.json")
        gnn_results_csv = os.path.join(gnn_folder, "results", "gnn_vs_center_comparison.csv")
        missing = [p for p in [candidates_simple, gnn_results_csv] if not os.path.isfile(p)]
        if missing:
            print(f"[post_analysis:ablation] Skipping — missing files: {missing}")
        else:
            trainer = _make_trainer()
            trainer.run_ablation_from_csv(
                gnn_results_csv=gnn_results_csv,
                ablation_candidates_json_file=candidates_simple,
                ablation_result_folder=os.path.join(gnn_folder, "results_ablation") + "/"
            )

    if interobserver_mod:
        set1 = os.path.join(interobserver_txt_folder, "1 set")
        set2 = os.path.join(interobserver_txt_folder, "2 set")
        has_set1 = os.path.isdir(set1) and any(Path(set1).glob("*.txt"))
        has_set2 = os.path.isdir(set2) and any(Path(set2).glob("*.txt"))
        if not (has_set1 and has_set2):
            print(f"[post_analysis:interobserver] Skipping — txt files not found at {interobserver_txt_folder}")
        else:
            process_gnn_interobserver(gnn_folder=gnn_interobserver_folder,
                                       json_info_folder=json_info_folder,
                                       interobserver_txt_folder=interobserver_txt_folder,
                                       create_ds=True, testing_mod=True)

    if robustness_mod:
        candidates_json = os.path.join(gnn_folder, "landmark_candidates.json")
        if not os.path.isfile(candidates_json):
            print(f"[post_analysis:robustness] Skipping — landmark_candidates.json not found at {gnn_folder}")
        else:
            process_gnn_robustness(gnn_folder=gnn_folder,
                                   result_folder=os.path.join(gnn_folder, "robustness_noise"))


class GNNInterobserverProject:

    LANDMARK_KEYS = {'R', 'L', 'N', 'RLC', 'RNC', 'LNC'}

    def __init__(self, gnn_folder):
        self.gnn_folder = gnn_folder

    def _parse_interobserver_txt(self, filepath):
        result = {}
        with open(filepath, 'r') as f:
            lines = [line.rstrip() for line in f]

        in_data = False
        in_point = False

        for line in lines:
            stripped = line.strip()
            if stripped == 'Data':
                in_data = True
                in_point = False
                continue
            if stripped == 'Legend':
                in_data = False
                in_point = False
                continue
            if stripped.startswith('='):
                continue
            if in_data and stripped == 'Point:':
                in_point = True
                continue
            if in_data and in_point:
                if not stripped:
                    continue
                if stripped.endswith(':') and len(stripped.split()) == 1:
                    in_point = False
                    continue
                parts = stripped.split()
                if parts and parts[0] in self.LANDMARK_KEYS and len(parts) >= 4:
                    result[parts[0]] = [float(parts[1]), float(parts[2]), float(parts[3])]

        return result

    def _get_case_number(self, filename):
        stem = Path(filename).stem
        m = re.search(r'\d+$', stem)
        return m.group() if m else None

    def configure_folder(self, json_info_folder, interobserver_txt_folder):
        testing_folder = Path(self.gnn_folder) / "data" / "testing"
        testing_folder.mkdir(parents=True, exist_ok=True)

        set1_folder = Path(interobserver_txt_folder) / "1 set"
        set2_folder = Path(interobserver_txt_folder) / "2 set"

        numbers_set1 = {self._get_case_number(f) for f in set1_folder.iterdir()
                        if f.suffix == '.txt' and self._get_case_number(f)}
        numbers_set2 = {self._get_case_number(f) for f in set2_folder.iterdir()
                        if f.suffix == '.txt' and self._get_case_number(f)}
        case_numbers = numbers_set1 & numbers_set2

        for num in case_numbers:
            src = Path(json_info_folder) / f"n{num}.json"
            dst = testing_folder / f"n{num}.json"
            if src.is_file():
                shutil.copy(src, dst)
            else:
                add_info_logging(f"json_info: n{num}.json not found, skipping.", "work_logger")

    def generate_interobserver_candidates(self, interobserver_txt_folder):
        set1_folder = Path(interobserver_txt_folder) / "1 set"
        set2_folder = Path(interobserver_txt_folder) / "2 set"

        for set_folder, out_name in [(set1_folder, "landmark_candidates_1set.json"),
                                     (set2_folder, "landmark_candidates_2set.json")]:
            candidates = {}
            for txt_file in sorted(set_folder.iterdir()):
                if txt_file.suffix != '.txt':
                    continue
                num = self._get_case_number(txt_file)
                if num is None:
                    continue
                case_name = f"n{num}"
                points = self._parse_interobserver_txt(txt_file)

                if not points:
                    add_info_logging(f"{txt_file.name}: no landmark points parsed.", "work_logger")
                    continue

                case_entry = {}
                for lm in self.LANDMARK_KEYS:
                    if lm not in points:
                        add_info_logging(f"{txt_file.name}: missing landmark {lm}.", "work_logger")
                        continue
                    case_entry[lm] = {
                        "candidate_points": [points[lm]],
                        "candidate_weights": [1.0]
                    }
                candidates[case_name] = case_entry

            out_path = Path(self.gnn_folder) / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(candidates, f, indent=2)

    def run_interobserver_test(self):
        trainer = _make_trainer()
        testing_folder = self.gnn_folder + '/data/testing'
        candidates_1set = self.gnn_folder + '/landmark_candidates_1set.json'
        candidates_2set = self.gnn_folder + '/landmark_candidates_2set.json'
        result_folder = self.gnn_folder + '/results/'
        trainer.compare_interobserver(testing_folder, candidates_1set, candidates_2set, result_folder)


def process_gnn_interobserver(gnn_folder, json_info_folder, interobserver_txt_folder,
                               create_ds=False, testing_mod=False):

    worker = GNNInterobserverProject(gnn_folder=gnn_folder)
    if create_ds:
        worker.configure_folder(json_info_folder=json_info_folder,
                                interobserver_txt_folder=interobserver_txt_folder)
        worker.generate_interobserver_candidates(interobserver_txt_folder=interobserver_txt_folder)
    if testing_mod:
        worker.run_interobserver_test()