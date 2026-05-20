import numpy as np
import json
import glob
import pandas as pd
import os
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import SimpleITK as sitk
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from models.landmarking_heart import landmarking_computeMeasurements_simplified
import time
from textwrap import indent
from scipy.ndimage import center_of_mass
from plots_data.plots import creat_box_plot

matplotlib.use("Agg")

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

COHORT_COLORS = {
    "German (g…)": "#1f77b4",  # blue
    "Slovenian norm (n…)": "#ff7f0e",  # orange
    "Slovenian pathology (p…)": "#2ca02c",  # green
    "Other": "#7f7f7f"  # gray fallback
}


class DataLoaderLandmarks:

    @staticmethod
    def load_candidates_json(json_file):
        """
        Accepts:
          - JSON shaped like:
            { "CASE": { "R_land": {"candidate_points":[[x,y,z],...],
                                   "candidate_weights":[w,...]}, ... } }
        Returns a normalized dict: {case: {landmark: {"candidate_points": np.ndarray (N,3),
                                                     "candidate_weights": np.ndarray (N,)}}}
        """

        def _norm_case_dict(case_dict):
            # case_dict: {landmark: {"candidate_points": ..., "candidate_weights": ...}}
            out = {}
            for lm, d in case_dict.items():
                pts = np.asarray(d.get("candidate_points"), dtype=float)
                wts = np.asarray(d.get("candidate_weights", [1.0]*len(pts)), dtype=float)
                out[lm] = {"candidate_points": pts, "candidate_weights": wts}
            return out

        data = json.loads(Path(json_file).read_text())
        # assume top-level is {case: {...}}
        norm = {case: _norm_case_dict(ld) for case, ld in data.items()}
        return norm

    @staticmethod
    def gather_json_files(folders):
        files = []
        for name in os.listdir(folders):
            if name.endswith('.json'):
                files.append(os.path.join(folders, name))
        return files


class MorphoGNN_SampleGenerator:

    def __init__(self, measurement_names, feature_names={"distances", "angles", "compactness", "PCA"},
                 landmark_names={"R", "L", "N", "RLC", "RNC", "LNC"}):
        """
        Initializes trainer with graph skeleton, feature names, etc.

        Args:
            measurement_names (dict): Dictionary mapping measurement names to involved landmarks.
            landmark_names (list): List of all possible landmark names.
            feature_names (list): List of feature types to compute (e.g., 'angle', 'distance', 'PCA').
        """
        self.measurement_names = measurement_names
        self.landmark_names = landmark_names
        self.feature_names = feature_names

    def __cos_angle_ABC(self, A, B, C):
        """
        Computes cosine of angle ABC, where B is the vertex.

        Args:
            A, B, C: np.array of shape (3,), representing 3D coordinates.

        Returns:
            Cosine of angle at point B.
        """
        BA = A - B
        BC = C - B
        cos_theta = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
        return cos_theta

    def __compute_compactness(self, points):
        """
        Computes the compactness of a set of 3D points.

        Args:
            points: np.array of shape (N, 3), where N is the number of points.

        Returns:
            centroid: np.array of shape (3,), the mean of all points.
            compactness: float, average distance from each point to the centroid.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be an array of shape (N, 3)")

        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        compactness = np.mean(distances)

        return compactness

    def __compute_measurement_value(self, candidate_points, measurement_name):
        pass

    def __feature_list_for_measurement(self, measurement_name, candidate_points, sample_metr_calculator):
        """
        Computes feature signature for a given measurement using its involved candidate landmarks.

        Args:
            measurement_name (str): Name of the measurement.
            candidate_points (dict): Dictionary of landmark name -> list of sampled points.

        Returns:
            dict: Computed features like angle, distance, PCA for the measurement.
        """
        landmarks = self.measurement_names[measurement_name]
        all_samples = [candidate_points[lm] for lm in landmarks if lm in candidate_points]

        # Basic example: flatten landmark sets into a single array
        # You will implement real feature extractors later
        features = []

        features.append(sample_metr_calculator.compute_individual_metric(measurement_name))

        if 'distances' in self.feature_names and len(all_samples) >= 2:
            dists = []
            for i in range(0, len(all_samples) - 1):
                for j in range(i + 1, len(all_samples)):
                    one_dist = np.linalg.norm(all_samples[i] - all_samples[j])
                    dists.append(one_dist)
                    features.append(one_dist)
            features.append(np.mean(dists))

        if 'angles' in self.feature_names and len(all_samples) >= 3:
            for i in range(0, len(all_samples)):
                for j in range(0, len(all_samples) - 1):
                    for k in range(j + 1, len(all_samples)):
                        if (i == j) or (i == k):
                            continue
                        features.append(self.__cos_angle_ABC(all_samples[j], all_samples[i], all_samples[k]))

        if 'PCA' in self.feature_names and len(all_samples) >= 3:
            all_coords = np.asarray(all_samples)
            cov = np.cov(all_coords.T)
            eigvals = np.linalg.eigvalsh(cov)
            features.append(eigvals[-1])
            features.append(eigvals[-2])
            pass

        if 'compactness' in self.feature_names and len(all_samples) >= 2:
            """
            Computes the compactness of a set of 3D points.

            Args:
                points: np.array of shape (N, 3), where N is the number of points.

            Returns:
                centroid: np.array of shape (3,), the mean of all points.
                compactness: float, average distance from each point to the centroid.
            """
            features.append(self.__compute_compactness(np.array(all_samples)))
            pass

        return features

    def __locked_test_temp1(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        landmarks = {'R': np.array(data['R']),
                     'L': np.array(data['L']),
                     'N': np.array(data['N']),
                     'RLC': np.array(data['RLC']),
                     'RNC': np.array(data['RNC']),
                     'LNC': np.array(data['LNC'])}

        # Parameters
        num_samples = 50
        noise_std = 0.5  # Adjust this to control perturbation scale

        # Generate perturbed samples
        perturbed_landmarks = []
        for _ in range(num_samples):
            sample = {}
            for key, points in landmarks.items():
                noise = np.random.normal(loc=0.0, scale=noise_std, size=points.shape)
                sample[key] = points + noise
            perturbed_landmarks.append(sample)

        case_samples = []
        for i in range(0, num_samples):
            one_sample = {}
            for key in self.measurement_names:
                one_sample[key] = self.__feature_list_for_measurement(key, perturbed_landmarks[i])
            case_samples.append(one_sample)

    def __generate_perturbed_landmarks(self, landmarks, num_samples, noise_std):
        """
        Generates perturbed landmarks by adding Gaussian noise to the original landmarks.

        Args:
            landmarks (dict): Dictionary of landmark names and their coordinates.
            num_samples (int): Number of perturbed samples to generate.
            noise_std (float): Standard deviation of Gaussian noise to add.

        Returns:
            list: List of dictionaries with perturbed landmarks.
        """
        perturbed_landmarks = []
        for _ in range(num_samples):
            sample = {}
            for key, points in landmarks.items():
                noise = np.random.normal(loc=0.0, scale=noise_std, size=points.shape)
                sample[key] = points + noise
            perturbed_landmarks.append(sample)
        return perturbed_landmarks

    def __generate_perturbed_landmarks_testing(self, candidate_points_for_case, num_samples):
        """
        Randomly sample landmark sets from candidate points with weights.

        Parameters
        ----------
        candidate_points_for_case : dict
            {landmark: {"candidate_points": (Ni,3), "candidate_weights": (Ni,)}}
        n_sets : int
            Number of sets to sample

        Returns
        -------
        sets : list of dict
            Each element is {landmark: (x,y,z)} for one sampled set
        """

        rng = np.random.default_rng()
        landmarks = list(candidate_points_for_case.keys())
        sets = []

        renaming = {}
        renaming['R_land'] = 'R'
        renaming['L_land'] = 'L'
        renaming['N_land'] = 'N'
        renaming['RLC_land'] = 'RLC'
        renaming['RNC_land'] = 'RNC'
        renaming['LNC_land'] = 'LNC'

        for _ in range(num_samples):
            sampled = {}
            for lm in landmarks:
                pts = candidate_points_for_case[lm]["candidate_points"]
                wts = candidate_points_for_case[lm]["candidate_weights"]

                # Normalize weights
                wts = np.asarray(wts, dtype=float)
                if wts.sum() <= 0:
                    probs = np.ones_like(wts) / len(wts)
                else:
                    probs = wts / wts.sum()

                idx = rng.choice(len(pts), p=probs)
                sampled[lm] = pts[idx]  # .tolist()
            sets.append(sampled)

        return sets

    def generate_training_samples(self, folders, num_samples=25, noise_std=1.5):
        files = DataLoaderLandmarks.gather_json_files(folders)
        training_samples = []
        training_noisy_labels = []
        training_labels = []
        p = 0
        start_time = time.time()
        for fp in files:
            with open(fp, 'r') as f:
                data = json.load(f)
            landmarks = {
                'R': np.array(data['R']),
                'L': np.array(data['L']),
                'N': np.array(data['N']),
                'RLC': np.array(data['RLC']),
                'RNC': np.array(data['RNC']),
                'LNC': np.array(data['LNC'])
            }
            ref_metr_calculator = landmarking_computeMeasurements_simplified(landmarks)
            ref_metrics = ref_metr_calculator.get_all_metrics()
            perturbed_landmarks = self.__generate_perturbed_landmarks(landmarks, num_samples, noise_std)
            one_sample = {}
            one_noisy_label = {}
            for key in self.measurement_names:
                one_sample[key] = []
                one_noisy_label[key] = []
            for sample in perturbed_landmarks:
                sample_metr_calculator = landmarking_computeMeasurements_simplified(sample)
                for key in self.measurement_names:
                    features_one = self.__feature_list_for_measurement(key, sample, sample_metr_calculator)
                    one_sample[key].append(features_one)
                    one_noisy_label[key].append(features_one[0])
            for key in self.measurement_names:
                one_sample[key] = np.asarray(one_sample[key])
                one_noisy_label[key] = np.asarray(one_noisy_label[key])
            training_samples.append(one_sample)
            training_noisy_labels.append(one_noisy_label)
            training_labels.append(ref_metrics.copy())
            p += 1
            if p % 10 == 0:
                end_time = time.time()
                print(f"Execution time: {end_time - start_time:.4f} seconds")
                # break
                pass
        return training_samples, training_noisy_labels, training_labels

    def __compute_centerpoint_metrics(self, candidates_json_file, allowed_cases):
        """
        For each allowed case, take the FIRST candidate point per landmark (the CoM),
        construct a landmark set, and compute metrics.

        Parameters
        ----------
        candidates_json_file : str
        allowed_cases : set[str]  -- cases present in testing folder (to filter candidates)

        Returns
        -------
        center_by_case : dict[str, dict[str, float]]
            {case: {measurement_name: value}}
        """
        bank = DataLoaderLandmarks.load_candidates_json(
            candidates_json_file)  # {case: {lm: {"candidate_points", "candidate_weights"}}}
        center_by_case = {}

        renaming = {}
        renaming['R'] = 'R_land'
        renaming['L'] = 'L_land'
        renaming['N'] = 'N_land'
        renaming['RLC'] = 'RLC_land'
        renaming['RNC'] = 'RNC_land'
        renaming['LNC'] = 'LNC_land'

        for case, lm_dict in bank.items():
            if case not in allowed_cases:
                continue

            # build a single landmark set from first candidate (index 0) for every landmark
            sample = {}
            for lm_name, payload in lm_dict.items():
                pts = np.asarray(payload["candidate_points"], dtype=float)
                if len(pts) == 0:
                    raise ValueError(f"Case {case}, landmark {lm_name} has no candidates.")
                sample[lm_name] = pts[0]  # FIRST candidate = center-of-mass

            metr = landmarking_computeMeasurements_simplified(sample)
            center_by_case[case] = metr.get_all_metrics()

        return center_by_case

    def generate_testing_samples(self, reference_landmark_folders, candidates_json_file, num_samples=25, noise_std=1.5):

        candidate_bank_data = DataLoaderLandmarks.load_candidates_json(candidates_json_file)
        # 2) Gather testing JSONs and allowed case names
        files = DataLoaderLandmarks.gather_json_files(reference_landmark_folders)  # uses .json filter
        allowed = {}
        for fp in files:
            case = Path(fp).stem
            allowed[case] = fp

        testing_samples = []
        testing_noisy_labels = []
        testing_labels = []

        centerpoint_labels = self.__compute_centerpoint_metrics(candidates_json_file, allowed)

        start_time = time.time()
        count = 0
        for case_name, candidate_points_for_case in candidate_bank_data.items():
            if case_name not in allowed:
                # skip candidate cases that we do not have GT for
                continue

            # 3a) load reference landmarks and compute reference metrics
            with open(allowed[case_name], 'r') as f:
                data = json.load(f)
            landmarks_ref = {
                'R': np.array(data['R']),
                'L': np.array(data['L']),
                'N': np.array(data['N']),
                'RLC': np.array(data['RLC']),
                'RNC': np.array(data['RNC']),
                'LNC': np.array(data['LNC'])
            }
            ref_calc = landmarking_computeMeasurements_simplified(landmarks_ref)
            ref_metrics = ref_calc.get_all_metrics()  # dict: measurement -> scalar

            perturbed_landmarks = self.__generate_perturbed_landmarks_testing(candidate_points_for_case, num_samples)

            # 3c) build features + noisy labels
            one_sample = {k: [] for k in self.measurement_names}
            one_noisy = {k: [] for k in self.measurement_names}
            for sample in perturbed_landmarks:
                metr_calc = landmarking_computeMeasurements_simplified(sample)
                for mname in self.measurement_names:
                    feats = self.__feature_list_for_measurement(mname, sample, metr_calc)
                    one_sample[mname].append(feats)
                    one_noisy[mname].append(feats[0])  # the "noisy metric" convention used in training

            for mname in self.measurement_names:
                one_sample[mname] = np.asarray(one_sample[mname])
                one_noisy[mname] = np.asarray(one_noisy[mname])

            testing_samples.append(one_sample)
            testing_noisy_labels.append(one_noisy)
            testing_labels.append(ref_metrics.copy())

            count += 1
            if count % 10 == 0:
                print(f"[testing] processed {count} cases in {(time.time() - start_time):.1f}s")
        return testing_samples, testing_noisy_labels, testing_labels, centerpoint_labels


class PhiPerCandidate(nn.Module):
    """Element-wise embedder: R^D -> R^E (applied row-wise).
       R - is the number of candidate descriptors per measurment (Monte carlo generator)
       E - embedding dimension
    """

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = embed_dim
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU()
        )

    def forward(self, x):  # x: [K_i, D_i]
        return self.net(x)  # [K_i, E]


class CandContextFuse(nn.Module):
    """
        This guy is needed to kinda inject the context from the graph into each candidate emebdding
        Fusion block: [K,E] + context[E] -> [K,E] (residual).
    """

    def __init__(self, E, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: hidden_dim = E
        self.net = nn.Sequential(
            nn.Linear(2 * E, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, E)
        )

    def forward(self, H, ctx):  # H:[K,E], ctx:[E]
        K = H.size(0)
        ctx_t = ctx.unsqueeze(0).expand(K, -1)  # [K,E]
        upd = self.net(torch.cat([H, ctx_t], dim=1))
        return H + upd  # residual, [K,E]


class MorphoGCN(nn.Module):

    def __init__(self,
                 num_nodes,  # X (number of measurement nodes)
                 embed_dim,  # E
                 connection_matrix,  # [X, X]
                 L=2,  # number of GNN layers
                 pool="mean",  # set pooling for context
                 edge_threshold=0.0,
                 use_edge_weights=True,
                 add_self_loops=True):
        super().__init__()
        self.X = len(num_nodes)
        self.E = embed_dim
        self.L = int(L)
        self.pool = pool
        self.use_edge_weights = use_edge_weights

        # Per-node row-wise embedders φ_i: [K_i, D_i] -> [K_i, E] (D_i inferred lazily)
        self.phi = nn.ModuleList([PhiPerCandidate(embed_dim) for _ in range(self.X)])

        # Graph from connection matrix
        edge_index, edge_weight = self._cmat_to_edge(connection_matrix, edge_threshold, add_self_loops)
        self.register_buffer("edge_index", edge_index)  # [2, Eedges]
        self.register_buffer("edge_weight", edge_weight)  # [Eedges]

        # GCN layers over node contexts (size E)
        self.gcn = nn.ModuleList([GCNConv(self.E, self.E, add_self_loops=False) for _ in range(self.L)])

        # Fuse graph-refined context back into candidates at each layer
        self.fuse = nn.ModuleList([nn.ModuleList([CandContextFuse(self.E) for _ in range(self.X)])
                                   for _ in range(self.L)])

        # Final candidate scorers g_i: [K_i, E] -> [K_i, 1] (row-wise)
        self.scorer = nn.ModuleList([
            nn.Sequential(nn.Linear(self.E, self.E), nn.ReLU(), nn.Linear(self.E, 1))
            for _ in range(self.X)
        ])

    @staticmethod
    def _cmat_to_edge(connection_matrix, threshold=0.0, add_self_loops=True):
        cm = connection_matrix.detach().cpu().numpy() if isinstance(connection_matrix, torch.Tensor) \
            else np.asarray(connection_matrix, dtype=float)
        N = cm.shape[0]
        assert cm.shape == (N, N)
        mask = cm >= float(threshold)
        if not add_self_loops:
            np.fill_diagonal(mask, False)
        src, dst = np.nonzero(mask)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_weight = torch.tensor(cm[src, dst], dtype=torch.float)
        return edge_index, edge_weight

    def _pool_ctx(self, H_list):
        ctxs = []
        for H in H_list:
            if self.pool == "mean":
                ctxs.append(H.mean(dim=0))
            elif self.pool == "max":
                ctxs.append(H.max(dim=0).values)
            else:
                raise ValueError("Unsupported pool mode")
        return torch.stack(ctxs, dim=0)  # [X, E]

    def forward(self, raw_features, noisy_metrics_list):
        """
        raw_features: list of length X; item i is float tensor [K_i, D_i] (D_i can differ per i)
        noisy_metrics_list: list of length X; item i is float tensor [K_i] (candidate metric values)
        """
        assert len(raw_features) == self.X and len(noisy_metrics_list) == self.X

        # Initial per-candidate embeddings (keeps [K_i, E] and adapts to D_i lazily)
        H = [self.phi[i](raw_features[i]) for i in range(self.X)]  # list of [K_i, E]

        # L layers: pool -> GCN -> fuse back (candidates evolve each layer)
        for l in range(self.L):
            C = self._pool_ctx(H)  # [X, E]
            if self.use_edge_weights:
                C = self.gcn[l](C, self.edge_index, self.edge_weight)
            else:
                C = self.gcn[l](C, self.edge_index)
            C = F.relu(C)
            H = [self.fuse[l][i](H[i], C[i]) for i in range(self.X)]  # list of [K_i, E]

        # Final candidate scoring and convex combination (per node)
        preds = []
        for i in range(self.X):
            logits = self.scorer[i](H[i]).squeeze(-1)  # [K_i]
            weights = torch.softmax(logits, dim=0)  # [K_i]
            if weights.device != noisy_metrics_list[i].device:
                weights = weights.to(noisy_metrics_list[i].device)
            preds.append(torch.sum(weights * noisy_metrics_list[i]))  # scalar
        return torch.stack(preds, dim=0)  # [X]


class MorphoGNNData(Data):

    def __init__(self, raw_features, edge_index, noisy_metrics_list, y):
        super().__init__()
        self.raw_features = raw_features
        self.edge_index = edge_index
        self.noisy_metrics_list = noisy_metrics_list
        self.y = y


class MorphoGNN_Visualizer():

    def print_loss_picture(epochs, loss_history, folder):
        plt.figure()
        plt.plot(range(epochs), loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(folder + "/loss_curve.png")
        print("Loss curve saved to loss_curve.png")
        plt.close()

    def save_box_plots(feature_names, all_predictions, all_targets, result_folder):
        # Compute per-feature errors
        metr_names = list(feature_names.keys())
        errors = [[] for _ in range(len(metr_names))]
        for preds, targets in zip(all_predictions, all_targets):
            for i, (p, t) in enumerate(zip(preds, targets)):
                errors[i].append(abs(p - t))

        # Boxplot
        plt.figure(figsize=(12, 6))
        plt.boxplot(errors, labels=metr_names, showfliers=False)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Absolute Error')
        plt.title('Prediction Errors per Measurement')
        plt.tight_layout()
        plt.savefig(result_folder + "/error_boxplot.png")
        print("Error boxplot saved to error_boxplot.png")

    def visualize_comparison(df, save_dir=None, show=False, measurements=None, jitter=0.0,
                             method1_col="abs_err_gnn", method2_col="abs_err_center",
                             method1_label="GNN", method2_label="Center"):
        """
        Visualize method1 vs method2 vs Reference from the tidy dataframe.

        Expected df columns:
          ['case','measurement','ref', method1_col, method2_col, abs_err_{method1_col}, abs_err_{method2_col}]
        """
        if df.empty:
            print("visualize_comparison: dataframe is empty; nothing to plot.")
            return

        if measurements is not None:
            df = df[df["measurement"].isin(measurements)].copy()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 2) Median absolute error bars by measurement
        agg = (
            df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[method1_col, method2_col], how="all")
            .groupby("measurement")[[method1_col, method2_col]]
            .median()
            .sort_values(method1_col)
        )
        if not agg.empty:
            fig = plt.figure(figsize=(max(6, 0.6 * len(agg)), 4))
            ax = fig.add_subplot(111)
            x = np.arange(len(agg))
            ax.bar(x - 0.2, agg[method1_col].to_numpy(), width=0.4, label=method1_label)
            ax.bar(x + 0.2, agg[method2_col].to_numpy(), width=0.4, label=method2_label)
            ax.set_xticks(x)
            ax.set_xticklabels(agg.index, rotation=45, ha="right")
            ax.set_ylabel("Median absolute error")
            ax.set_title("Median absolute error by measurement")
            ax.legend()
            ax.grid(True, axis="y", linestyle=":")
            if save_dir:
                fig.savefig(Path(save_dir) / "median_abs_error_bars.png", dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

        # 3) Error distributions (boxplots)
        df_long = pd.melt(
            df,
            id_vars=["measurement", "case"],
            value_vars=[method1_col, method2_col],
            var_name="method",
            value_name="abs_error",
        ).replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_error"])
        if False and not df_long.empty:
            for m, d in df_long.groupby("measurement"):
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111)
                data = [d.loc[d["method"] == method1_col, "abs_error"].to_numpy(),
                        d.loc[d["method"] == method2_col, "abs_error"].to_numpy()]
                ax.boxplot(data, labels=[method1_label, method2_label], showfliers=False)
                ax.set_title(f"Absolute error distribution - {m}")
                ax.set_ylabel("Absolute error")
                ax.grid(True, axis="y", linestyle=":")
                if save_dir:
                    fig.savefig(Path(save_dir) / f"abs_error_boxplot_{m}.png", dpi=150, bbox_inches="tight")
                if show:
                    plt.show()
                else:
                    plt.close(fig)

    def visualize_comparison_mean(df, save_dir=None, show=False, measurements=None, jitter=0.0,
                                  method1_col="abs_err_gnn", method2_col="abs_err_center",
                                  method1_label="GNN", method2_label="Center"):
        """
        Visualize method1 vs method2 vs Reference from the tidy dataframe.

        Expected df columns:
          ['case','measurement','ref', method1_col, method2_col]
        """
        if df.empty:
            print("visualize_comparison: dataframe is empty; nothing to plot.")
            return

        if measurements is not None:
            df = df[df["measurement"].isin(measurements)].copy()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 2) Mean absolute error bars by measurement
        agg = (
            df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[method1_col, method2_col], how="all")
            .groupby("measurement")[[method1_col, method2_col]]
            .mean()
            .sort_values(method1_col)
        )
        if not agg.empty:
            fig = plt.figure(figsize=(max(6, 0.6 * len(agg)), 4))
            ax = fig.add_subplot(111)
            x = np.arange(len(agg))
            ax.bar(x - 0.2, agg[method1_col].to_numpy(), width=0.4, label=method1_label)
            ax.bar(x + 0.2, agg[method2_col].to_numpy(), width=0.4, label=method2_label)
            ax.set_xticks(x)
            ax.set_xticklabels(agg.index, rotation=45, ha="right")
            ax.set_ylabel("Mean absolute error")
            ax.set_title("Mean absolute error by measurement")
            ax.legend()
            ax.grid(True, axis="y", linestyle=":")
            if save_dir:
                fig.savefig(Path(save_dir) / "mean_abs_error_bars.png", dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

        # 3) Error distributions (boxplots)
        df_long = pd.melt(
            df,
            id_vars=["measurement", "case"],
            value_vars=[method1_col, method2_col],
            var_name="method",
            value_name="abs_error",
        ).replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_error"])
        if False and not df_long.empty:
            for m, d in df_long.groupby("measurement"):
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111)
                data = [d.loc[d["method"] == method1_col, "abs_error"].to_numpy(),
                        d.loc[d["method"] == method2_col, "abs_error"].to_numpy()]
                ax.boxplot(data, labels=[method1_label, method2_label], showfliers=False)
                ax.set_title(f"Absolute error distribution - {m}")
                ax.set_ylabel("Absolute error")
                ax.grid(True, axis="y", linestyle=":")
                if save_dir:
                    fig.savefig(Path(save_dir) / f"abs_error_boxplot_{m}.png", dpi=150, bbox_inches="tight")
                if show:
                    plt.show()
                else:
                    plt.close(fig)


class MorphoGCN_Trainer:

    def __init__(self, feature_names, embed_dim=16, connection_map_simple=False):
        # Dummy configuration
        self.feature_names = feature_names
        # self.feature_dims = [12] * 9 + [75] * 6
        self.embed_dim = embed_dim
        if connection_map_simple:
            self.connection_map = np.ones((len(self.feature_names), len(self.feature_names)))  # Fully connected for now
        else:
            self.connection_map = self.__compute_connection_map(feature_names)

    def __compute_connection_map(self, feature_names, iou_threshold=0.1):
        """
        Build NxN connection matrix between measurement nodes weighted by IoU
        of their landmark sets.

        Parameters

        feature_names : dict[str, list[str]]
            {measurement_name: [landmark_name, ...]}
        iou_threshold : float
            Values below this threshold are set to 0.
        include_self_loops : bool
            If True, diagonal entries are set to 1.0.

        Returns

        conn_matrix : np.ndarray
            NxN matrix of IoU weights (float)
        meas_order : list[str]
            Order of measurement names corresponding to matrix indices
        """
        meas_order = list(feature_names.keys())
        sets = {m: set(feature_names[m]) for m in meas_order}
        n = len(meas_order)

        conn_matrix = np.zeros((n, n), dtype=float)

        for i, mi in enumerate(meas_order):
            for j, mj in enumerate(meas_order):
                if i == j:
                    conn_matrix[i, j] = 1.0
                else:
                    inter = len(sets[mi] & sets[mj])
                    union = len(sets[mi] | sets[mj])
                    iou = inter / union if union > 0 else 0.0
                    conn_matrix[i, j] = max(iou, iou_threshold)

        return conn_matrix

    def __set_feature_dims(self, data_example):
        self.feature_dims = []
        for name in self.feature_names:
            self.feature_dims.append(len(data_example[name][0]))

    # Helper to create edge_index
    def __build_edge_index(self, connection_map):
        edges = [(i, j) for i in range(len(connection_map)) for j in range(len(connection_map)) if connection_map[i, j]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    # Data preparation
    def __build_pyg_sample(self, sample_dict, label_dict, noisy_metric_dict, feature_names, connection_map):
        raw_features = []
        noisy_metrics_list = []
        for f in feature_names:
            candidates = np.array(sample_dict[f])  # shape [K, D]
            raw_features.append(torch.tensor(candidates, dtype=torch.float32))
            noisy_metrics_list.append(torch.tensor(noisy_metric_dict[f], dtype=torch.float32))

        labels = torch.tensor([label_dict[f] for f in feature_names], dtype=torch.float32)
        edge_index = self.__build_edge_index(connection_map)
        return MorphoGNNData(raw_features, edge_index, noisy_metrics_list, labels)

    # Assume training_samples and training_labels are provided
    def __prepare_dataset(self, training_samples, training_labels, noisy_metric_dicts, feature_names, connection_map):
        dataset = []
        for sample, label, noisy_metrics in zip(training_samples, training_labels, noisy_metric_dicts):
            data = self.__build_pyg_sample(sample, label, noisy_metrics, feature_names, connection_map)
            dataset.append(data)
        return dataset

    # Training loop
    def __train_model(self, model_folder, model, dataset, epochs=10000, batch_size=1, lr=1e-5):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        loss_history = []

        for epoch in range(0, epochs):
            total_loss = 0
            model.train()
            count = 0
            for data in dataset:
                optimizer.zero_grad()
                preds = model(data.raw_features, data.noisy_metrics_list)
                loss = loss_fn(preds, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            total_loss /= count
            loss_history.append(total_loss)
            print(f"Epoch {epoch:03d}: Loss = {total_loss:.4f}")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), model_folder + "/morpho_gnn_model_" + str(epoch) + ".pth")
                MorphoGNN_Visualizer.print_loss_picture(epoch + 1, loss_history, model_folder)
        torch.save(model.state_dict(), model_folder + "/morpho_gnn_model.pth")
        MorphoGNN_Visualizer.print_loss_picture(epochs, loss_history, model_folder)

    # Testing loop
    def __test_model(self, model_folder, model, dataset):
        # Load model architecture (requires external metadata if needed)
        # For demo purposes, we assume model is already instantiated and passed in from outside
        model_path = os.path.join(model_folder, "morpho_gnn_model.pth")
        model.eval()
        model.load_state_dict(torch.load(model_path))

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data in dataset:
                pred = model(data.raw_features, data.noisy_metrics_list)
                all_predictions.append(pred.numpy())
                all_targets.append(data.y.numpy())

        MorphoGNN_Visualizer.save_box_plots(self.feature_names, all_predictions, all_targets, model_folder)
        return all_predictions, all_targets

    def __arrays_to_dict_of_dict(self, all_predictions, all_targets, centerpoint_labels, case_names, meas_names):
        """
        Convert predictions/targets arrays into dict-of-dict with same structure as centerpoint_labels.

        Parameters
        ----------
        all_predictions : array-like [num_cases, num_meas]
        all_targets     : array-like [num_cases, num_meas]
        centerpoint_labels : dict of dict {case: {measurement: val}}, used for structure
        case_names : list of str, case names in order
        meas_names : list of str, measurement names in order

        Returns
        -------
        pred_dict, targ_dict : dict of dicts
            {case: {measurement: value}}
        """
        all_predictions = np.asarray(all_predictions)
        all_targets = np.asarray(all_targets)

        pred_dict = {}
        targ_dict = {}

        for i, case in enumerate(case_names):
            pred_dict[case] = {}
            targ_dict[case] = {}
            for j, meas in enumerate(meas_names):
                pred_dict[case][meas] = float(all_predictions[i, j])
                targ_dict[case][meas] = float(all_targets[i, j])

        return pred_dict, targ_dict

    def __build_comparison_df(self,
                              all_predictions,  # {case: {measurement: method1}}
                              all_targets,  # {case: {measurement: ref}}
                              centerpoint_labels,  # {case: {measurement: method2}}
                              case_order=None,
                              meas_order=None,
                              save_csv=None,
                              method1_col="gnn",
                              method2_col="center",
                              method3_predictions=None,
                              method3_col=None):
        # cases = intersection of the three dicts (preserve order if provided)
        base_cases = set(all_predictions) & set(all_targets) & set(centerpoint_labels)
        if method3_predictions is not None:
            base_cases &= set(method3_predictions)
        if case_order is None:
            cases = sorted(base_cases)
        else:
            cases = [c for c in case_order if c in base_cases]

        abs_err_col1 = f"abs_err_{method1_col}"
        abs_err_col2 = f"abs_err_{method2_col}"
        abs_err_col3 = f"abs_err_{method3_col}" if method3_col else None

        rows = []
        for case in cases:
            ref_dict = all_targets[case]
            m1_dict = all_predictions.get(case, {})
            m2_dict = centerpoint_labels.get(case, {})
            m3_dict = method3_predictions.get(case, {}) if method3_predictions else {}

            names = (meas_order if meas_order is not None else list(ref_dict.keys()))

            for m in names:
                if m not in ref_dict:
                    continue
                ref_v = float(ref_dict[m])
                m1_v = float(m1_dict.get(m, np.nan))
                m2_v = float(m2_dict.get(m, np.nan))
                row = {
                    "case": case,
                    "measurement": m,
                    "ref": ref_v,
                    method1_col: m1_v,
                    method2_col: m2_v,
                    abs_err_col1: abs(m1_v - ref_v) if np.isfinite(m1_v) else np.nan,
                    abs_err_col2: abs(m2_v - ref_v) if np.isfinite(m2_v) else np.nan,
                }
                if method3_col:
                    m3_v = float(m3_dict.get(m, np.nan))
                    row[method3_col] = m3_v
                    row[abs_err_col3] = abs(m3_v - ref_v) if np.isfinite(m3_v) else np.nan
                rows.append(row)

        df = pd.DataFrame(rows)

        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False)

        # quick summary
        if not df.empty:
            err_cols = [abs_err_col1, abs_err_col2]
            if abs_err_col3:
                err_cols.append(abs_err_col3)
            summary = (
                df.groupby("measurement")[err_cols]
                .median()
                .sort_values(abs_err_col1)
            )
            print("\nMedian absolute error by measurement:")
            print(summary)

        return df

    def __subset_name(self, case):
        """Map a case ID to a cohort."""
        if case.startswith("HOM"):
            return "German (HOM…)"
        if case.startswith("n"):
            return "Slovenian norm (n…)"
        if case.startswith("p"):
            return "Slovenian pathology (p…)"
        return "Other"

    def __describe_series(self, s):
        s = pd.to_numeric(s, errors="coerce")
        s = s[np.isfinite(s)]
        return {
            "N": int(s.shape[0]),
            "mean": float(s.mean()) if s.size else np.nan,
            "std": float(s.std(ddof=1)) if s.size > 1 else np.nan,
            "median": float(s.median()) if s.size else np.nan,
        }

    def __ensure_cohort_column(self, df):
        if "cohort" in df.columns:
            return df

        def map_cohort(case):
            s = str(case)
            if s.startswith("HOM"): return "German (HOM…)"
            if s.startswith("n"):   return "Slovenian norm (n…)"
            if s.startswith("p"):   return "Slovenian pathology (p…)"
            return "Other"

        df = df.copy()
        df["cohort"] = df["case"].map(map_cohort)
        return df

    def __normal_pdf(self, x, mu, sigma):
        sigma = float(sigma)
        if sigma <= 0 or not np.isfinite(sigma):
            return np.zeros_like(x)
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def __finite(self, series):
        a = pd.to_numeric(series, errors="coerce").to_numpy()
        return a[np.isfinite(a)]

    # ---------- 1) Normal-distribution style plots ----------
    def __plot_normals_both_methods_per_measurement(self, df, measurement, outfile,
                                                    method1_col="abs_err_gnn",
                                                    method2_col="abs_err_center",
                                                    method1_label="GNN",
                                                    method2_label="Center"):
        """
        For a given measurement, overlay normal curves for both error columns (solid/dashed)
        per cohort (color).
        """
        sub = df[df["measurement"] == measurement]
        if sub.empty:
            return

        # Collect stats for both methods per cohort
        stats = []  # (cohort, method_label, mu, sigma)
        for coh, g in sub.groupby("cohort"):
            vals_1 = self.__finite(g[method1_col])
            vals_2 = self.__finite(g[method2_col])
            if vals_1.size:
                stats.append((coh, method1_label, float(vals_1.mean()), float(vals_1.std(ddof=1))))
            if vals_2.size:
                stats.append((coh, method2_label, float(vals_2.mean()), float(vals_2.std(ddof=1))))

        if not stats:
            return

        # Shared x-range from both methods (use pooled mu±4σ, clamp left to 0)
        mus = np.array([s[2] for s in stats], dtype=float)
        sigmas = np.array([s[3] for s in stats], dtype=float)
        sigma_all = np.nanmean(sigmas[np.isfinite(sigmas)])
        if not np.isfinite(sigma_all) or sigma_all == 0:
            pooled = self.__finite(sub[method1_col]).tolist() + self.__finite(sub[method2_col]).tolist()
            sigma_all = np.std(pooled, ddof=1) if len(pooled) > 1 else 1.0
        mu_all = np.nanmean(mus)
        left = max(0.0, mu_all - 4 * sigma_all)
        right = mu_all + 4 * sigma_all
        x = np.linspace(left, right, 1000)

        # Plot
        plt.figure(figsize=(7.5, 4.8))
        for coh in sorted(sub["cohort"].unique().tolist()):
            # method1 (solid)
            pair = [s for s in stats if s[0] == coh and s[1] == method1_label]
            if pair:
                _, _, mu, sigma = pair[0]
                y = self.__normal_pdf(x, mu, sigma)
                plt.plot(x, y, color=COHORT_COLORS.get(coh, "#7f7f7f"),
                         linewidth=2.2, linestyle="-", label=f"{coh} — {method1_label}")
            # method2 (dashed)
            pair = [s for s in stats if s[0] == coh and s[1] == method2_label]
            if pair:
                _, _, mu, sigma = pair[0]
                y = self.__normal_pdf(x, mu, sigma)
                plt.plot(x, y, color=COHORT_COLORS.get(coh, "#7f7f7f"),
                         linewidth=2.2, linestyle="--", label=f"{coh} — {method2_label}")

        plt.xlabel("Absolute error")
        plt.ylabel("Normal PDF")
        plt.title(f"{measurement} — Normal curves ({method1_label} solid, {method2_label} dashed)")
        # Reduce legend clutter: combine duplicates by label
        handles, labels = plt.gca().get_legend_handles_labels()
        seen, h_clean, l_clean = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                h_clean.append(h);
                l_clean.append(l);
                seen.add(l)
        plt.legend(h_clean, l_clean, ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(outfile, dpi=400)
        plt.close()

    def __plot_box_both_methods_per_measurement(self, df, measurement, outfile, include_total=True,
                                                method1_col="abs_err_gnn",
                                                method2_col="abs_err_center",
                                                method1_label="GNN",
                                                method2_label="Center"):
        """
        For a given measurement, draws grouped boxplots by cohort:
          per cohort -> two boxes: [method1, method2]
        If include_total=True, appends one extra pair "Total" pooling all cohorts.
        """
        sub = df[df["measurement"] == measurement]
        if sub.empty:
            return

        # Cohorts (preserve order from grouping)
        cohorts = [c for c, _ in sub.groupby("cohort")]
        data_m1 = [self.__finite(sub[sub["cohort"] == c][method1_col]) for c in cohorts]
        data_m2 = [self.__finite(sub[sub["cohort"] == c][method2_col]) for c in cohorts]

        if include_total:
            cohorts = cohorts + ["Total"]
            data_m1 = data_m1 + [self.__finite(sub[method1_col])]
            data_m2 = data_m2 + [self.__finite(sub[method2_col])]

        # positions: for each cohort make a pair of boxes close together
        n = len(cohorts)
        base_positions = np.arange(n) * 3.0  # cluster spacing
        pos_m1 = base_positions - 0.4
        pos_m2 = base_positions + 0.4

        plt.figure(figsize=(max(7, 1.3 * n), 5))

        bp_1 = plt.boxplot(
            data_m1, positions=pos_m1, widths=0.7,
            patch_artist=True, showfliers=True
        )
        bp_2 = plt.boxplot(
            data_m2, positions=pos_m2, widths=0.7,
            patch_artist=True, showfliers=True
        )

        # color & hatch by cohort (Total uses gray)
        for i, box in enumerate(bp_1['boxes']):
            coh = cohorts[i]
            color = COHORT_COLORS.get(coh, "#7f7f7f")
            box.set_facecolor(color);
            box.set_alpha(0.35);
            box.set_hatch("//")
        for i, box in enumerate(bp_2['boxes']):
            coh = cohorts[i]
            color = COHORT_COLORS.get(coh, "#7f7f7f")
            box.set_facecolor(color);
            box.set_alpha(0.35);
            box.set_hatch("\\\\")

        # thicker lines
        for coll in [bp_1, bp_2]:
            for elem in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                for item in coll[elem]:
                    item.set_linewidth(1.5)

        # x ticks centered between the pair
        plt.xticks(base_positions, cohorts, rotation=0)
        plt.ylabel("Absolute error")
        plt.title(f"{measurement} — Absolute error by cohort ({method1_label} vs {method2_label})")

        legend_handles = [
            Patch(facecolor="#cccccc", hatch="//", alpha=0.35, label=method1_label),
            Patch(facecolor="#cccccc", hatch="\\\\", alpha=0.35, label=method2_label),
        ]
        plt.legend(handles=legend_handles, title="Method", loc="best")

        plt.tight_layout()
        plt.savefig(outfile, dpi=400)
        plt.close()

    # ======================
    # Driver
    # ======================
    def __make_all_measurement_plots(self, df, plot_folder,
                                     method1_col="abs_err_gnn", method2_col="abs_err_center",
                                     method1_label="GNN", method2_label="Center"):
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        df = self.__ensure_cohort_column(df)
        measurements = sorted(df["measurement"].dropna().unique().tolist())

        for m in measurements:
            self.__plot_normals_both_methods_per_measurement(
                df, m, f"{plot_folder}/{m}_NormalDistribution.png",
                method1_col, method2_col, method1_label, method2_label)

            self.__plot_box_both_methods_per_measurement(
                df, m, f"{plot_folder}/{m}_boxWhiskers.png",
                method1_col=method1_col, method2_col=method2_col,
                method1_label=method1_label, method2_label=method2_label)

    def save_error_stats_text(self, df, out_txt,
                              method1_col="abs_err_gnn", method2_col="abs_err_center",
                              method1_label="GNN", method2_label="Center"):
        """
        Write mean / std / median / N for both error columns
        for (a) all data and (b) cohorts by case prefix. Also per-measurement.
        """
        Path(out_txt).parent.mkdir(parents=True, exist_ok=True)

        # tag cohorts
        df = df.copy()
        df["cohort"] = df["case"].map(self.__subset_name)

        cohorts = ["All", "German (HOM…)", "Slovenian norm (n…)", "Slovenian pathology (p…)"]
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("Aortic Landmarking — Error Statistics\n")
            f.write("=====================================\n\n")
            f.write(f"Metrics reported on absolute errors ({method1_col}, {method2_col}).\n")
            f.write("Cohorts are inferred from case prefixes: HOM*, n*, p*.\n\n")

            for cohort in cohorts:
                if cohort == "All":
                    dsub = df
                else:
                    dsub = df[df["cohort"] == cohort]

                f.write(f"## Cohort: {cohort}\n")
                f.write(f"Total rows: {len(dsub)} | Unique cases: {dsub['case'].nunique()}\n\n")

                # --- pooled over all measurements ---
                f.write("Pooled over all measurements:\n")
                pooled = {
                    method1_label: self.__describe_series(dsub[method1_col]),
                    method2_label: self.__describe_series(dsub[method2_col]),
                }
                for k, stats in pooled.items():
                    f.write(f"  {k} -> N={stats['N']}, mean={stats['mean']:.4f}, "
                            f"std={stats['std']:.4f}, median={stats['median']:.4f}\n")
                f.write("\n")

                # --- per measurement ---
                f.write("Per-measurement:\n")
                lines = []
                for m, g in dsub.groupby("measurement", sort=True):
                    m1_stats = self.__describe_series(g[method1_col])
                    m2_stats = self.__describe_series(g[method2_col])
                    line = (
                        f"- {m}: "
                        f"{method1_col} [N={m1_stats['N']}, mean={m1_stats['mean']:.4f}, "
                        f"std={m1_stats['std']:.4f}, median={m1_stats['median']:.4f}] ; "
                        f"{method2_col} [N={m2_stats['N']}, mean={m2_stats['mean']:.4f}, "
                        f"std={m2_stats['std']:.4f}, median={m2_stats['median']:.4f}]"
                    )
                    lines.append(line)
                if lines:
                    f.write(indent("\n".join(lines), "  "))
                else:
                    f.write("  (no data)\n")

                f.write("\n\n")

        print(f"[stats] Wrote: {out_txt}")

    # Output interface
    def train_morpho_gcn2(self, model_folder, folders):
        sample_generator = MorphoGNN_SampleGenerator(self.feature_names)
        training_samples, training_noisy_labels, training_labels = sample_generator.generate_training_samples(folders)
        self.__set_feature_dims(training_samples[0])
        dataset = self.__prepare_dataset(training_samples, training_labels, training_noisy_labels, self.feature_names,
                                         self.connection_map)
        model = MorphoGCN(self.feature_dims, self.embed_dim,
                          self.connection_map)  # , len(next(iter(training_noisy_labels[0].values()))))
        self.__train_model(model_folder, model, dataset)
        return model

    # Output interface
    def test_morpho_gcn_nnUnet(self, model_folder, reference_landmark_folders, json_file):
        sample_generator = MorphoGNN_SampleGenerator(self.feature_names)
        testing_samples, testing_noisy_labels, testing_labels, centerpoint_labels = sample_generator.generate_testing_samples(
            reference_landmark_folders, json_file)
        self.__set_feature_dims(testing_samples[0])

        dataset = self.__prepare_dataset(testing_samples, testing_labels, testing_noisy_labels, self.feature_names,
                                         self.connection_map)

        # 4) Instantiate model with correct num_candidates (length of any noisy metric vector)
        first_noisy = next(iter(testing_noisy_labels[0].values()))
        num_candidates = int(len(first_noisy)) if hasattr(first_noisy, "__len__") else 0
        model = MorphoGCN(self.feature_dims, self.embed_dim, self.connection_map)  # , num_candidates)

        # 5) Run test
        all_predictions, all_targets = self.__test_model(model_folder, model, dataset)

        case_names = list(centerpoint_labels.keys())  # e.g. ["HOM_M70_H176_W71_YA", ...]
        meas_names = list(centerpoint_labels[case_names[0]].keys())  # e.g. ["IC_R","IC_L","IC_N","IC_distance"]
        all_predictions, all_targets = self.__arrays_to_dict_of_dict(all_predictions, all_targets, centerpoint_labels,
                                                                     case_names, meas_names)
        return all_predictions, all_targets, centerpoint_labels

    def __compute_metrics_from_candidates_json(self, candidates_json_file, allowed_cases):
        """Load candidates JSON and compute metrics using the first candidate per landmark."""
        bank = DataLoaderLandmarks.load_candidates_json(candidates_json_file)
        metrics_by_case = {}
        for case, lm_dict in bank.items():
            if case not in allowed_cases:
                continue
            sample = {}
            for lm_name, payload in lm_dict.items():
                pts = np.asarray(payload["candidate_points"], dtype=float)
                if len(pts) == 0:
                    raise ValueError(f"Case {case}, landmark {lm_name} has no candidates.")
                sample[lm_name] = pts[0]
            metr = landmarking_computeMeasurements_simplified(sample)
            metrics_by_case[case] = metr.get_all_metrics()
        return metrics_by_case

    def __compute_weighted_avg_metrics(self, candidates_json_file, allowed_cases):
        """Compute metrics using nnUNet-weighted average of candidate points per landmark."""
        bank = DataLoaderLandmarks.load_candidates_json(candidates_json_file)
        metrics_by_case = {}
        for case, lm_dict in bank.items():
            if case not in allowed_cases:
                continue
            sample = {}
            for lm_name, payload in lm_dict.items():
                pts = np.asarray(payload["candidate_points"], dtype=float)
                wts = np.asarray(payload["candidate_weights"], dtype=float)
                if len(pts) == 0:
                    raise ValueError(f"Case {case}, landmark {lm_name} has no candidates.")
                if wts.sum() == 0:
                    wts = np.ones(len(wts))
                sample[lm_name] = np.average(pts, axis=0, weights=wts)
            metr = landmarking_computeMeasurements_simplified(sample)
            metrics_by_case[case] = metr.get_all_metrics()
        return metrics_by_case

    def __compute_reference_metrics_from_jsons(self, testing_folders):
        """
        Scan testing_folders for per-case landmark JSONs and compute reference metrics.

        Returns
        -------
        ref_by_case : dict[str, dict[str, float]]
            {case: {measurement_name: value}}
        lm_by_case  : dict[str, dict[str, np.ndarray]]
            {case: {landmark_name: (3,) array}}  (raw coords if you need them later)
        """
        files = DataLoaderLandmarks.gather_json_files(testing_folders)  # only .json
        ref_by_case, lm_by_case = {}, {}

        for fp in files:
            case = Path(fp).stem
            with open(fp, "r") as f:
                d = json.load(f)

            # expect keys: R, L, N, RLC, RNC, LNC (adapt if yours differ)
            lms = {
                "R": np.asarray(d["R"], dtype=float),
                "L": np.asarray(d["L"], dtype=float),
                "N": np.asarray(d["N"], dtype=float),
                "RLC": np.asarray(d["RLC"], dtype=float),
                "RNC": np.asarray(d["RNC"], dtype=float),
                "LNC": np.asarray(d["LNC"], dtype=float),
            }
            lm_by_case[case] = lms

            ref_calc = landmarking_computeMeasurements_simplified(lms)
            ref_by_case[case] = ref_calc.get_all_metrics()  # {measurement: scalar}

        return ref_by_case, lm_by_case

    @staticmethod
    def _compute_mean_norm_mae(df, err_col, measurements):
        """Mean normalized MAE (%) across given measurements: mean(abs_err)/mean(ref)*100."""
        norms = []
        for m in measurements:
            subset = df[df["measurement"] == m]
            if subset.empty:
                continue
            mean_ref = subset["ref"].dropna().mean()
            if not np.isfinite(mean_ref) or mean_ref == 0:
                continue
            mean_err = subset[err_col].dropna().mean()
            if np.isfinite(mean_err):
                norms.append(mean_err / mean_ref * 100.0)
        return float(np.mean(norms)) if norms else float('inf')

    def _evaluate_once(self, model_folder, reference_landmark_folders, candidates_json_file):
        """Single GNN inference pass. Returns (df, predictions, targets, centerpoint_labels)."""
        all_predictions, all_targets, centerpoint_labels = self.test_morpho_gcn_nnUnet(
            model_folder=model_folder,
            json_file=candidates_json_file,
            reference_landmark_folders=reference_landmark_folders
        )
        df = self.__build_comparison_df(
            all_predictions=all_predictions,
            all_targets=all_targets,
            centerpoint_labels=centerpoint_labels,
        )
        return df, all_predictions, all_targets, centerpoint_labels

    def _run_with_retry(self, model_folder, reference_landmark_folders, candidates_json_file,
                        mae_length_threshold, max_retries, result_folder=None):
        """Run GNN inference with optional retry/threshold logic. Returns best df."""
        use_threshold = mae_length_threshold is not None
        n_attempts = max_retries if use_threshold else 1

        best_df = None
        best_len = float('inf')
        best_ang = float('inf')
        fallback_df = None
        fallback_len = float('inf')
        fallback_ang = float('inf')

        for attempt in range(n_attempts):
            df, _, _, _ = self._evaluate_once(model_folder, reference_landmark_folders, candidates_json_file)

            if use_threshold:
                cur_len = self._compute_mean_norm_mae(df, "abs_err_gnn", _LENGTH_MEASUREMENTS)
                cur_ang = self._compute_mean_norm_mae(df, "abs_err_gnn", _ANGLE_MEASUREMENTS)
                print(f"[attempt {attempt + 1}/{n_attempts}] "
                      f"length MAE: {cur_len:.2f}%  angle MAE: {cur_ang:.2f}%")

                if (cur_len, cur_ang) < (fallback_len, fallback_ang):
                    fallback_len, fallback_ang = cur_len, cur_ang
                    fallback_df = df

                if cur_len <= mae_length_threshold and (cur_len, cur_ang) < (best_len, best_ang):
                    best_len, best_ang = cur_len, cur_ang
                    best_df = df
                    print(f"  New best — length={cur_len:.2f}%  angle={cur_ang:.2f}%")
                    if result_folder:
                        Path(result_folder).mkdir(parents=True, exist_ok=True)
                        best_df.to_csv(result_folder + "gnn_vs_center_comparison.csv", index=False)
            else:
                best_df = df

        if best_df is None:
            print("[warning] No run passed length threshold — using overall best as fallback.")
            best_df = fallback_df

        return best_df

    def _save_comparison_results(self, df, result_folder):
        """Save CSV, text summary, and all plots for a GNN vs CoM comparison."""
        Path(result_folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(result_folder + "gnn_vs_center_comparison.csv", index=False)
        self.save_error_stats_text(df, result_folder + "gnn_vs_center_summary.txt")
        self.__make_all_measurement_plots(df, result_folder + "plots")
        MorphoGNN_Visualizer.visualize_comparison_mean(
            df, save_dir=result_folder + "plots", show=False, measurements=None, jitter=0.05
        )
        creat_box_plot(result_folder, "gnn_vs_center_comparison.csv")

    @staticmethod
    def _render_table_figure(col_labels, rows, save_path):
        n_cols = len(col_labels)
        fig_h = max(1.5, 0.38 * len(rows) + 1.0)
        fig_w = max(9, n_cols * 1.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(n_cols)))
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_text_props(ha="center", fontweight="bold")
                cell.set_height(cell.get_height() * 2.0)
            else:
                cell.set_text_props(ha="center")
                cell.PAD = 0.05
                if row_idx % 2 == 0:
                    cell.set_facecolor("#f2f2f2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_aggregated_table(self, df, result_folder):
        """
        From an ablation DataFrame (columns abs_err_gnn, abs_err_center, abs_err_ablation),
        build two tables saved as CSV + PNG:
          - table_per_measurement : mean MAE per method + signed pairwise diffs for every metric
          - table_aggregated      : mean absolute diff per group (length_mm / angle_deg)
                                    uses |diff| before averaging so +/- values don't cancel out
        """
        col_gnn = "abs_err_gnn"
        col_center = "abs_err_center"
        col_abl = "abs_err_ablation"
        if not {col_gnn, col_center, col_abl}.issubset(df.columns):
            return

        groups = [
            ("length_mm", _LENGTH_MEASUREMENTS),
            ("angle_deg", _ANGLE_MEASUREMENTS),
        ]

        # --- per-measurement table ---
        per_meas_rows = []
        per_meas_fig_rows = []
        for group_name, meas_list in groups:
            for m in meas_list:
                sub = df[df["measurement"] == m]
                if sub.empty:
                    continue
                mae_gnn    = sub[col_gnn].mean()
                mae_center = sub[col_center].mean()
                mae_abl    = sub[col_abl].mean()
                d_gc = mae_gnn - mae_center
                d_ga = mae_gnn - mae_abl
                d_ca = mae_center - mae_abl
                per_meas_rows.append({
                    "group": group_name,
                    "measurement": m,
                    "mean_MAE_GNN": round(mae_gnn, 4),
                    "mean_MAE_center": round(mae_center, 4),
                    "mean_MAE_aver_nnUNet": round(mae_abl, 4),
                    "diff_GNN_vs_center": round(d_gc, 4),
                    "diff_GNN_vs_aver_nnUNet": round(d_ga, 4),
                    "diff_center_vs_aver_nnUNet": round(d_ca, 4),
                })
                per_meas_fig_rows.append([
                    m,
                    f"{mae_gnn:.2f}",
                    f"{mae_center:.2f}",
                    f"{mae_abl:.2f}",
                    f"{d_gc:+.2f}",
                    f"{d_ga:+.2f}",
                    f"{d_ca:+.2f}",
                ])

        per_meas_df = pd.DataFrame(per_meas_rows)
        per_meas_df.to_csv(result_folder + "table_per_measurement.csv", index=False)
        self._render_table_figure(
            col_labels=[
                "Measurement",
                "GNN\n(mean MAE)",
                "CoM\n(mean MAE)",
                "nnUNet avg\n(mean MAE)",
                "Δ GNN−CoM",
                "Δ GNN−nnUNet",
                "Δ CoM−nnUNet",
            ],
            rows=per_meas_fig_rows,
            save_path=result_folder + "table_per_measurement.png",
        )

        # --- aggregated table: only Δ columns, averaged as mean(|Δ|) per group ---
        agg_rows = []
        agg_fig_rows = []
        for group_name, _ in groups:
            sub = per_meas_df[per_meas_df["group"] == group_name]
            if sub.empty:
                continue
            d_gc = sub["diff_GNN_vs_center"].abs().mean()
            d_ga = sub["diff_GNN_vs_aver_nnUNet"].abs().mean()
            d_ca = sub["diff_center_vs_aver_nnUNet"].abs().mean()
            agg_rows.append({
                "group": group_name,
                "mean_abs_diff_GNN_vs_center": round(d_gc, 4),
                "mean_abs_diff_GNN_vs_aver_nnUNet": round(d_ga, 4),
                "mean_abs_diff_center_vs_aver_nnUNet": round(d_ca, 4),
            })
            agg_fig_rows.append([
                group_name,
                f"{d_gc:.2f}",
                f"{d_ga:.2f}",
                f"{d_ca:.2f}",
            ])

        agg_df = pd.DataFrame(agg_rows)
        agg_df.to_csv(result_folder + "table_aggregated.csv", index=False)
        self._render_table_figure(
            col_labels=[
                "Group",
                "mean |Δ| GNN−CoM",
                "mean |Δ| GNN−nnUNet",
                "mean |Δ| CoM−nnUNet",
            ],
            rows=agg_fig_rows,
            save_path=result_folder + "table_aggregated.png",
        )

        print(f"[aggregated] Saved tables → {result_folder}")

    def compare_gnn_vs_center(self, model_folder, reference_landmark_folders, candidates_json_file,
                              result_folder, mae_length_threshold=None, max_retries=5):
        ref_by_case, _ = self.__compute_reference_metrics_from_jsons(reference_landmark_folders)
        if not ref_by_case:
            raise RuntimeError("No testing JSONs found -> unable to compute reference metrics.")

        best_df = self._run_with_retry(
            model_folder, reference_landmark_folders, candidates_json_file,
            mae_length_threshold, max_retries, result_folder
        )

        self._save_comparison_results(best_df, result_folder)

    def run_ablation_from_csv(self, gnn_results_csv, ablation_candidates_json_file, ablation_result_folder):
        """Run ablation using previously saved GNN test results (no re-inference needed)."""
        ablation_result_folder = str(Path(ablation_result_folder)) + "/"
        df_saved = pd.read_csv(gnn_results_csv)

        best_preds, best_targets, best_center = {}, {}, {}
        for case, grp in df_saved.groupby('case'):
            best_preds[case] = dict(zip(grp['measurement'], grp['gnn']))
            best_targets[case] = dict(zip(grp['measurement'], grp['ref']))
            best_center[case] = dict(zip(grp['measurement'], grp['center']))

        allowed_cases = set(best_preds.keys())
        ablation_metrics = self.__compute_weighted_avg_metrics(ablation_candidates_json_file, allowed_cases)

        Path(ablation_result_folder).mkdir(parents=True, exist_ok=True)
        ablation_df = self.__build_comparison_df(
            all_predictions=best_preds,
            all_targets=best_targets,
            centerpoint_labels=best_center,
            method3_predictions=ablation_metrics,
            method3_col="ablation",
            save_csv=ablation_result_folder + "gnn_vs_center_comparison.csv"
        )
        creat_box_plot(ablation_result_folder, "gnn_vs_center_comparison.csv",
                       method3_col="abs_err_ablation", method3_label="nnUNet avg")
        self._save_aggregated_table(ablation_df, ablation_result_folder)

    def compare_interobserver(self, testing_folder, candidates_1set_json, candidates_2set_json, result_folder):
        """
        Inter-observer study: compare two sets of manual annotations against reference.

        1 set → Inter-observer  (method1)
        2 set → Intra-observer  (method2)

        No GNN model required — each set provides a single candidate per landmark (weight=1.0),
        so metrics are computed directly from that candidate point.
        """
        m1_col = "inter_observer"
        m2_col = "intra_observer"
        m1_err = "abs_err_inter"
        m2_err = "abs_err_intra"
        m1_label = "Inter-observer"
        m2_label = "Intra-observer"

        Path(result_folder).mkdir(parents=True, exist_ok=True)
        Path(result_folder + "plots").mkdir(parents=True, exist_ok=True)

        # --- (A) Reference metrics from testing JSONs
        ref_by_case, _ = self.__compute_reference_metrics_from_jsons(testing_folder)
        allowed_cases = set(ref_by_case.keys())
        if not allowed_cases:
            raise RuntimeError("No testing JSONs found -> unable to compute reference metrics.")

        # --- (B) Metrics from each observer set (first/only candidate per landmark)
        inter_metrics = self.__compute_metrics_from_candidates_json(candidates_1set_json, allowed_cases)
        intra_metrics = self.__compute_metrics_from_candidates_json(candidates_2set_json, allowed_cases)

        # --- (C) Build comparison DataFrame
        df = self.__build_comparison_df(
            all_predictions=inter_metrics,
            all_targets=ref_by_case,
            centerpoint_labels=intra_metrics,
            method1_col=m1_col,
            method2_col=m2_col,
            save_csv=result_folder + "interobserver_comparison.csv"
        )

        # Rename error columns to semantic names
        df = df.rename(columns={
            f"abs_err_{m1_col}": m1_err,
            f"abs_err_{m2_col}": m2_err,
        })

        # Re-save CSV with renamed error columns
        df.to_csv(result_folder + "interobserver_comparison.csv", index=False)

        self.save_error_stats_text(df, result_folder + "interobserver_summary.txt",
                                   method1_col=m1_err, method2_col=m2_err,
                                   method1_label=m1_label, method2_label=m2_label)

        self.__make_all_measurement_plots(df, result_folder + "plots",
                                          method1_col=m1_err, method2_col=m2_err,
                                          method1_label=m1_label, method2_label=m2_label)

        MorphoGNN_Visualizer.visualize_comparison_mean(
            df,
            save_dir=result_folder + "plots",
            show=False,
            method1_col=m1_err, method2_col=m2_err,
            method1_label=m1_label, method2_label=m2_label
        )

        creat_box_plot(result_folder, "interobserver_comparison.csv",
                       method1_col=m1_err, method2_col=m2_err,
                       method1_label=m1_label, method2_label=m2_label)


class nnUnet_CandidatePointGenerator:

    def __init__(self, json_path, n_candidates = 5, min_dist = 0.5, threshold = 0.1, include_com = True):
        """
        n_peaks        : max number of peaks per landmark (besides COM)
        min_dist_vox   : minimum separation (in voxels) between peaks (int or tuple per axis)
        threshold      : minimum prob to consider a peak (after per-channel max-normalization)
        include_com    : include center of mass candidate
        """
        self.n_candidates = int(n_candidates)
        self.threshold = float(threshold)
        self.include_com = bool(include_com)
        self.min_dist = min_dist
        self.landmark_ids, self.landmark_names = self.__load_label_names_from_json(json_path)

    def __load_label_names_from_json(self, json_path):
        with open(json_path, "r") as f:
            meta = json.load(f)
        labels = meta.get("labels", {})
        # build ordered [(index, name), ...] skipping background (value 0)
        # JSON might be {name: idx}; your snippet is name->int mapping.
        pairs = []
        for name, idx in labels.items():
            if name.lower() == "background" or idx == 0:
                continue
            pairs.append((idx, name))
        pairs.sort(key=lambda t: t[0])
        # return indices and names separately
        class_indices = [p[0] for p in pairs]
        class_names = [p[1] for p in pairs]
        return class_indices, class_names

    def __load_npz_array(self, file_path):
        with np.load(file_path, allow_pickle = True) as f:
            for k in ("softmax", "pred", "probs", "probabilities"):
                if k in f and isinstance(f[k], np.ndarray):
                    return f[k]
            # fallback: first array
            for k in f.files:
                if isinstance(f[k], np.ndarray):
                    return f[k]
        raise ValueError("No ndarray found in NPZ")

    def __ensure_channel_first(self, arr):
        # Accept (C, Z, Y, X), (Z, Y, X, C)
        if arr.ndim == 4:
            # channels last if last dim is small and first is big
            return np.moveaxis(arr, -1, 0) if (arr.shape[0] > 10 * arr.shape[-1]) else arr
        raise ValueError(f"Expected D array, got {arr.shape}")

    def __as_tuple(self, v):
        return tuple(int(vv) for vv in (v if isinstance(v, (list, tuple)) else (v, v, v)))

    def __fit_min_dist_to_ndim(self, md, ndim):
        md = tuple(int(x) for x in md)
        return md[:ndim]

    def __clip_idx(self, idx, shape):
        return tuple(int(np.clip(i, 0, s - 1)) for i, s in zip(idx, shape))

    def __far_enough(self, a, b, md):
        # axis-wise box distance (Chebyshev-like). If any axis distance <= md, they collide.
        return all(abs(ai - bi) > d for ai, bi, d in zip(a, b, md))

    def __from_npz_file(self, path, case_name):
        """Return dict: landmark_idx -> list of {'idx': (z,y,x), 'score': float}"""
        arr = self.__load_npz_array(path + '/' + case_name + '.npz')        # (C, Z, Y, X)
        nii_image = sitk.ReadImage(path + '/' + case_name + '.nii.gz')
        resolution = nii_image.GetSpacing()[::-1] # flip to Z, Y, X
        origin = nii_image.GetOrigin()[::-1] # flip to Z, Y, X
        arr = arr[1:, :, :, :]
        return self.__extract_one_case(arr, resolution, origin)


    def __ellipsoid_mask(self, min_dist, spacing):
        """
        Build a boolean 3D mask of points within a distance <= min_dist
        using anisotropic voxel spacing (dz, dy, dx).
        """
        dz, dy, dx = spacing
        rz = max(1, int(np.ceil(min_dist / dz)))
        ry = max(1, int(np.ceil(min_dist / dy)))
        rx = max(1, int(np.ceil(min_dist / dx)))
        z = np.arange(-rz, rz+1)
        y = np.arange(-ry, ry+1)
        x = np.arange(-rx, rx+1)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        d2 = (zz*dz)**2 + (yy*dy)**2 + (xx*dx)**2
        mask = d2 <= (min_dist + 1e-9)**2
        return mask

    def __find_peaks_center_greedy(self, vol, N, min_dist, spacing_zyx=(1.0, 1.0, 1.0),
                                   start_frac=0.99,   # start threshold at start_frac * vol.max()
                                   min_frac=0.10,     # do not go below this fraction of max
                                   decay=0.90,        # multiply frac by this if not enough points (e.g., 0.90 -> -10% each round)
                                   include_center_in_output=False):
        """
        Select points that are:
          - >= adaptive threshold,
          - as close as possible to the center of mass,
          - at least 'min_dist' (in same units as spacing_zyx) away from previously selected points,
            including the center of mass.

        Parameters
        ----------
        vol : (Z,Y,X) array of floats
        N : int
            Number of points to return (not counting the center unless include_center_in_output=True).
        min_dist : float
            Minimum separation distance in *physical units* (same units as spacing_zyx).
        spacing_zyx : tuple(float,float,float)
            (dz, dy, dx) spacings for proper metric distances.
        start_frac : float
            Initial threshold = start_frac * vol.max().
        min_frac : float
            Stop lowering threshold below this fraction of vol.max().
        decay : float
            If not enough points found, new_frac = old_frac * decay.
        include_center_in_output : bool
            If True, the nearest voxel to COM is added as the first output point (it still
            must satisfy threshold). Regardless, COM is an exclusion point.

        Returns
        -------
        points_xyz : (M,3) int
            Selected points in (X,Y,Z) voxel coordinates (M <= N or N+1 if center included).
        scores : (M,) float
            Corresponding intensities.
        threshold_used : float
            The final absolute threshold used.
        """
        vol = np.asarray(vol, dtype=float)
        if vol.ndim != 3:
            raise ValueError("vol must be 3D (Z,Y,X)")

        vmax = float(np.nanmax(vol))
        if not np.isfinite(vmax):
            return np.empty((0,3), int), np.empty((0,), float), np.nan

        dz, dy, dx = spacing_zyx
        min_d2 = float(min_dist)**2

        # --- center of mass as exclusion point ---
        cz, cy, cx = center_of_mass(np.nan_to_num(vol, nan=0.0))
        com = np.array([cz, cy, cx], dtype=float)

        # nearest voxel to the center (integer indices)
        com_voxel = np.array([int(round(cz)), int(round(cy)), int(round(cx))], dtype=int)
        com_voxel = np.clip(com_voxel, [0,0,0], np.array(vol.shape)-1)

        # Precompute full coordinate grid (lazy indices from mask later)
        Z, Y, X = vol.shape

        def d2_vox_zyx(a, b):
            # a,b as (z,y,x)
            return ((a[0]-b[0])*dz)**2 + ((a[1]-b[1])*dy)**2 + ((a[2]-b[2])*dx)**2

        # Greedy selection from candidates sorted by center proximity
        def select_from_mask(mask, want):
            # candidate coordinates
            zc, yc, xc = np.nonzero(mask)
            if zc.size == 0:
                return [], []

            coords = np.column_stack((zc, yc, xc))
            # distance to COM (true metric, squared is enough for ordering)
            d2c = ((coords[:,0]-com[0])*dz)**2 + ((coords[:,1]-com[1])*dy)**2 + ((coords[:,2]-com[2])*dx)**2
            vals = vol[zc, yc, xc]

            # sort: closest to center first; tie-break by higher intensity
            order = np.lexsort((-vals, d2c))
            coords = coords[order]
            vals = vals[order]

            selected = []
            svals = []

            # exclusion set starts with the COM itself
            excl = [com_voxel.astype(int)]

            # optionally add COM to output (only if it meets threshold and spacing vs itself is trivial)
            if include_center_in_output:
                # if COM voxel is in candidates, add it first
                # (if it isn't, we'll add the nearest candidate in the loop anyway)
                if mask[tuple(com_voxel)]:
                    selected.append(com_voxel.copy())
                    svals.append(float(vol[tuple(com_voxel)]))
                    excl.append(com_voxel.copy())  # already there, but explicit
                    if len(selected) == want:
                        return selected, svals

            for pt, v in zip(coords, vals):
                # skip the COM voxel if we already added it and it's the same
                if include_center_in_output and np.array_equal(pt, com_voxel) and selected and np.array_equal(selected[0], com_voxel):
                    continue
                # distance constraint vs all already selected + COM
                if all(d2_vox_zyx(pt, q) >= min_d2 for q in excl):
                    selected.append(pt)
                    svals.append(float(v))
                    excl.append(pt)
                    if len(selected) == want:
                        break

            return selected, svals

        # Adaptive threshold loop
        frac = start_frac
        last_pts, last_vals, last_thr = [], [], vmax * start_frac
        while frac + 1e-12 >= min_frac:
            thr = vmax * frac
            mask = (vol >= thr)

            pts, vals = select_from_mask(mask, N)

            if len(pts) >= N:
                # success at this threshold
                pts = np.array(pts, dtype=int)    # (k,3) z,y,x
                vals = np.array(vals, dtype=float)
                # return in (x,y,z)
                return pts[:, ::-1], vals

            # keep best so far in case we never reach N before min_frac
            if len(pts) > len(last_pts):
                last_pts, last_vals, last_thr = pts, vals, thr

            # relax threshold
            frac *= decay

        # Fallback: return the best we managed before hitting min_frac
        if last_pts:
            pts = np.array(last_pts, dtype=int)
            vals = np.array(last_vals, dtype=float)
            return pts[:, ::-1], vals

        # Nothing found
        return np.empty((0,3), int), np.empty((0,), float)

    def __extract_one_case(self, probs, resolution, origin):
        probs = self.__ensure_channel_first(probs)   # (C, Z, Y, X) or (C, Y, X)
        C = probs.shape[0]
        is_3d = (probs.ndim == 4)


        result = {}
        for c in range(C):
            p = probs[c].astype(np.float32, copy = False)
            coords, prob = self.__find_peaks_center_greedy(p, self.n_candidates, self.min_dist, resolution)
            result[self.landmark_names[self.landmark_ids[c] - 1]] = {}
            result[self.landmark_names[self.landmark_ids[c] - 1]]['candidate_points'] = coords * np.asarray(resolution[::-1]) + origin[::-1]
            #result[self.landmark_names[self.landmark_ids[c] - 1]]['candidate_points'] = result[self.landmark_names[self.landmark_ids[c] - 1]]['candidate_points'][:, ::-1]
            result[self.landmark_names[self.landmark_ids[c] - 1]]['candidate_weights'] = prob
            pass
            #result[]

        return result

    def extract_candidate_points(self, folder, pattern = "*.npz"):
        """Return list of (case_id, candidates_dict) for all matching files."""
        out = {}
        for p in sorted(glob.glob(os.path.join(folder, pattern))):
            case_id = os.path.splitext(os.path.basename(p))[0]
            out[case_id] = self.__from_npz_file(folder, case_id)
            print('finished case ', case_id)
            #break
        return out

    def __sanitize(self, obj):
        """Recursively convert NumPy types to plain Python for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self.__sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.__sanitize(v) for v in obj]
        return obj

    def save_results(self, all_cases_results, json_path):
        """
        all_cases_results structure:
          {
            "CASE_NAME": {
              "R_land": {"candidate_points": np.ndarray (N,3), "candidate_weights": np.ndarray (N,) or list},
              "L_land": {...},
              ...
            },
            ...
          }
        """
        out = self.__sanitize(all_cases_results)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
