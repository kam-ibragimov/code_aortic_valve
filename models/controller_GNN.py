import json
import os
import re
import shutil
from pathlib import Path
from data_preprocessing.text_worker import add_info_logging
from models.landmarking_heart import landmarking_computeMeasurements_simplified
from models.implementationGNN import MorphoGCN_Trainer, nnUnet_CandidatePointGenerator


class GNNProject:

    def __init__(self, result_6_nnunet_folder, gnn_folder, train_test_lists):
        self.result_6_nnunet_folder = result_6_nnunet_folder
        self.gnn_folder = gnn_folder
        self.train_test_lists = train_test_lists

    def landmark_nnUnet_generateCandidates(self):
        if os.path.isfile(self.gnn_folder + '/landmark_candidates.json'):
            pass
        else:
            extractor = nnUnet_CandidatePointGenerator(
                json_path = self.result_6_nnunet_folder + '/dataset.json',
                n_candidates = 5,
                min_dist = 1,
                threshold = 0.15,
                include_com = True
            )
            results = extractor.extract_candidate_points(self.result_6_nnunet_folder)
            # save to JSON
            extractor.save_results(results, self.gnn_folder + '/landmark_candidates.json')

    def landmark_GNN_train(self):
        measurment_tester = MorphoGCN_Trainer(landmarking_computeMeasurements_simplified.get_measurement_names())
        measurment_tester.train_morpho_gcn2(self.gnn_folder, self.gnn_folder + '/data/training')

    def landmark_GNN_test(self):
        measurment_tester = MorphoGCN_Trainer(landmarking_computeMeasurements_simplified.get_measurement_names())
        # tester1.test_morpho_gcn_nnUnet(heart_GNN, heart_GNN + '/data/testing', heart_nnUnet + '/Landmarking/temp/landmark_candidates.json')
        measurment_tester.compare_gnn_vs_center(self.gnn_folder, self.gnn_folder + '/data/testing',
                                                self.gnn_folder + '/landmark_candidates.json',
                                                self.gnn_folder + '/results/')

    def configure_folder(self, json_info_folder):
        def _clear_folder(folder):
            """Очищает папку, удаляя все файлы и подпапки"""
            if not folder.exists():
                add_info_logging(f"Folder '{str(folder)}' does not exist.", "work_logger")
                return

            for item in folder.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()  # Удаляем файл или символическую ссылку
                elif item.is_dir():
                    shutil.rmtree(item)  # Удаляем папку рекурсивно

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

        list_train_cases = _get_file_list(self.train_test_lists,
                                         "train",
                                         "case_name",
                                         ".json",
                                         json_info_folder)
        list_test_cases = _get_file_list(self.train_test_lists,
                                        "test"
                                        , "case_name", ".json",
                                        json_info_folder)
        _copy_img(list_train_cases, Path(self.gnn_folder) / "data" / "training")
        _copy_img(list_test_cases, Path(self.gnn_folder) / "data" / "testing")


def process_gnn(result_6_nnunet_folder, gnn_folder, train_test_lists, json_info_folder, create_ds=False,
                training_mod=False, testing_mod=False,):

    gnn_worker = GNNProject(result_6_nnunet_folder=result_6_nnunet_folder,
                            gnn_folder=gnn_folder,
                            train_test_lists=train_test_lists)
    if create_ds:
        gnn_worker.configure_folder(json_info_folder=json_info_folder)
        gnn_worker.landmark_nnUnet_generateCandidates()
    if training_mod:
        gnn_worker.landmark_GNN_train()
    if testing_mod:
        gnn_worker.landmark_GNN_test()
    print('Hi')


class GNNInterobserverProject:

    LANDMARK_KEYS = {'R', 'L', 'N', 'RLC', 'RNC', 'LNC'}

    def __init__(self, gnn_folder):
        self.gnn_folder = gnn_folder

    def _parse_interobserver_txt(self, filepath):
        """Parse Point section from interobserver txt file (supports in/n/rn prefix formats)."""
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
                # New subsection starts (single word ending with ":")
                if stripped.endswith(':') and len(stripped.split()) == 1:
                    in_point = False
                    continue
                parts = stripped.split()
                # Skip header line (Name  X1  Y1  Z1) and parse only target landmarks
                if parts and parts[0] in self.LANDMARK_KEYS and len(parts) >= 4:
                    result[parts[0]] = [float(parts[1]), float(parts[2]), float(parts[3])]

        return result

    def _get_case_number(self, filename):
        """Extract numeric suffix from filename stem (e.g. 'in1' → '1', 'rn105' → '105')."""
        stem = Path(filename).stem
        m = re.search(r'\d+$', stem)
        return m.group() if m else None

    def configure_folder(self, json_info_folder, interobserver_txt_folder):
        """Copy n-case JSONs to data/testing based on available interobserver txt files."""
        testing_folder = Path(self.gnn_folder) / "data" / "testing"
        testing_folder.mkdir(parents=True, exist_ok=True)

        # Collect case numbers present in both sets
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
        """
        Build landmark_candidates_1set.json (Inter-observer) and
        landmark_candidates_2set.json (Intra-observer) from txt files.
        Each landmark gets one candidate point with weight 1.0.
        """
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
        """Run inter-/intra-observer comparison against reference JSONs."""
        measurment_tester = MorphoGCN_Trainer(landmarking_computeMeasurements_simplified.get_measurement_names())
        testing_folder = self.gnn_folder + '/data/testing'
        candidates_1set = self.gnn_folder + '/landmark_candidates_1set.json'
        candidates_2set = self.gnn_folder + '/landmark_candidates_2set.json'
        result_folder = self.gnn_folder + '/results/'
        measurment_tester.compare_interobserver(testing_folder, candidates_1set, candidates_2set, result_folder)


def process_gnn_interobserver(gnn_folder, json_info_folder, interobserver_txt_folder,
                               create_ds=False, testing_mod=False):

    worker = GNNInterobserverProject(gnn_folder=gnn_folder)
    if create_ds:
        worker.configure_folder(json_info_folder=json_info_folder,
                                interobserver_txt_folder=interobserver_txt_folder)
        worker.generate_interobserver_candidates(interobserver_txt_folder=interobserver_txt_folder)
    if testing_mod:
        worker.run_interobserver_test()
