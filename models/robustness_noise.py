import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.landmarking_heart import landmarking_computeMeasurements_simplified
from models.implementationGNN import MorphoGCN_Trainer

_LANDMARK_KEYS = ("R", "L", "N", "RLC", "RNC", "LNC")


class CandidateRobustness:

    SIGMA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    P_GRID = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    N_SEEDS = 10

    @staticmethod
    def perturb_candidates(original_data, sigma, p, seed):
        """
        Deep copy original_data and add isotropic Gaussian noise to candidate_points.
        Each point is perturbed independently with probability p.
        candidate_weights are never modified.
        """
        rng = np.random.default_rng(seed)
        data = copy.deepcopy(original_data)

        if sigma == 0.0 or p == 0.0:
            return data

        for case_dict in data.values():
            for lm_payload in case_dict.values():
                pts = np.asarray(lm_payload["candidate_points"], dtype=float)
                for i in range(len(pts)):
                    if rng.random() < p:
                        pts[i] += rng.normal(0.0, sigma, size=3)
                lm_payload["candidate_points"] = pts.tolist()

        return data

    @staticmethod
    def _compute_detection_error(gt_folder, perturbed_data):
        """
        Euclidean distance from pts[0] (CoM candidate) to GT per landmark.
        Returns mean over landmarks of the median error across cases (mm).
        """
        files = list(Path(gt_folder).glob("*.json"))
        if not files:
            return float("nan")

        errors = {lm: [] for lm in _LANDMARK_KEYS}

        for fp in files:
            case = fp.stem
            if case not in perturbed_data:
                continue
            with open(fp) as f:
                gt = json.load(f)

            case_cands = perturbed_data[case]
            for lm in _LANDMARK_KEYS:
                if lm not in gt or lm not in case_cands:
                    continue
                pts = np.asarray(case_cands[lm]["candidate_points"], dtype=float)
                if len(pts) == 0:
                    continue
                gt_pt = np.asarray(gt[lm], dtype=float)
                errors[lm].append(float(np.linalg.norm(pts[0] - gt_pt)))

        per_lm_medians = [np.median(v) for v in errors.values() if v]
        return float(np.mean(per_lm_medians)) if per_lm_medians else float("nan")

    def run_sweep(self, model_folder, reference_landmark_folders, candidates_json_path,
                  result_folder, sigma_grid=None, p_grid=None, n_seeds=None):
        """
        Sweep over (sigma, p) grid with n_seeds repetitions each.
        For every cell: perturb candidates → single GNN inference → record errors.
        Saves per-metric CSVs and a combined summary CSV.
        """
        sigma_grid = sigma_grid if sigma_grid is not None else self.SIGMA_GRID
        p_grid = p_grid if p_grid is not None else self.P_GRID
        n_seeds = n_seeds if n_seeds is not None else self.N_SEEDS

        trainer = MorphoGCN_Trainer(
            landmarking_computeMeasurements_simplified.get_measurement_names()
        )

        with open(candidates_json_path) as f:
            original_data = json.load(f)

        result_folder = Path(result_folder)
        result_folder.mkdir(parents=True, exist_ok=True)
        tmp_json = result_folder / "_perturbed_candidates_tmp.json"

        rows = []

        for sigma in sigma_grid:
            for p in p_grid:
                det_errors, gnn_maes, com_maes = [], [], []

                for seed in range(n_seeds):
                    perturbed = self.perturb_candidates(original_data, sigma, p, seed)

                    with open(tmp_json, "w") as f:
                        json.dump(perturbed, f)

                    df, _, _, _ = trainer._evaluate_once(
                        model_folder=model_folder,
                        reference_landmark_folders=reference_landmark_folders,
                        candidates_json_file=str(tmp_json)
                    )

                    gnn_maes.append(float(df["abs_err_gnn"].mean()))
                    com_maes.append(float(df["abs_err_center"].mean()))
                    det_errors.append(
                        self._compute_detection_error(reference_landmark_folders, perturbed)
                    )

                    print(f"[robustness] sigma={sigma}  p={p}  seed={seed}  "
                          f"gnn={gnn_maes[-1]:.4f}  com={com_maes[-1]:.4f}  "
                          f"det={det_errors[-1]:.4f}")

                rows.append({
                    "sigma": sigma,
                    "p": p,
                    "det_mean": float(np.mean(det_errors)),
                    "det_std":  float(np.std(det_errors, ddof=1)) if len(det_errors) > 1 else 0.0,
                    "gnn_mean": float(np.mean(gnn_maes)),
                    "gnn_std":  float(np.std(gnn_maes, ddof=1)) if len(gnn_maes) > 1 else 0.0,
                    "com_mean": float(np.mean(com_maes)),
                    "com_std":  float(np.std(com_maes, ddof=1)) if len(com_maes) > 1 else 0.0,
                })

        if tmp_json.exists():
            tmp_json.unlink()

        df_results = pd.DataFrame(rows)
        df_results.to_csv(result_folder / "robustness_noise_sweep.csv", index=False)

        for metric, mean_col, std_col in [
            ("detection_error_mm", "det_mean", "det_std"),
            ("gnn_mae",            "gnn_mean", "gnn_std"),
            ("com_mae",            "com_mean", "com_std"),
        ]:
            self._save_metric_csv(df_results, metric, mean_col, std_col, result_folder)
            self._print_summary_table(df_results, metric, mean_col, sigma_grid, p_grid)

    @staticmethod
    def _save_metric_csv(df, metric_name, mean_col, std_col, result_folder):
        out = df[["sigma", "p", mean_col, std_col]].rename(
            columns={mean_col: "mean", std_col: "std"}
        )
        out.to_csv(result_folder / f"robustness_{metric_name}.csv", index=False)

    @staticmethod
    def _print_summary_table(df, metric_name, mean_col, sigma_grid, p_grid):
        print(f"\n=== {metric_name} (mean) ===")
        pivot = df.pivot(index="sigma", columns="p", values=mean_col)
        print(pivot.to_string())
        print()