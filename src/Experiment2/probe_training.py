"""Train Experiment 2 probes with template-based cross-validation."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).parent.parent))
import utils


warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EXPERIMENT1_DIR = DATA_DIR / "experiment1"
EXPERIMENT2_DIR = DATA_DIR / "experiment2"
PLOTS_DIR = BASE_DIR / "plots" / "experiment2"
SIGNIFICANT_MODELS_PATH = EXPERIMENT1_DIR / "significant_models.json"
SIGNIFICANCE_REPORT_PATH = EXPERIMENT1_DIR / "significance_tests.txt"
ACTIVATIONS_DIR = EXPERIMENT2_DIR
EVAL_PATH = EXPERIMENT2_DIR / "probe_evaluation.txt"

TARGET_MODEL_KEYS: Tuple[str, ...] = (
    "mistral",
    "gemma-2-9b",
    "llama-3.1-8b-instruct",
)

TARGETS: Dict[str, str] = {"targets_c": "C", "targets_p": "P"}
LAMBDA_GRID = np.concatenate(([0.0], np.logspace(-3, 0.75, num=8)))


@dataclass
class FeatureSubset:
    head_pairs: List[Tuple[int, int]]
    feature_indices: np.ndarray
    group_slices: List[slice]

    @classmethod
    def from_layers(
        cls, head_pairs: Iterable[Tuple[int, int]], num_heads: int, head_dim: int
    ) -> "FeatureSubset":
        indices: List[int] = []
        group_slices: List[slice] = []
        head_list = list(head_pairs)
        for idx, (layer, head) in enumerate(head_list):
            start = ((layer * num_heads) + head) * head_dim
            indices.extend(range(start, start + head_dim))
            group_slices.append(slice(idx * head_dim, (idx + 1) * head_dim))
        return cls(head_list, np.asarray(indices, dtype=np.int64), group_slices)

    def selected_heads(self, coefficients: np.ndarray, threshold: float = 1e-8) -> List[dict]:
        selected: List[dict] = []
        for (layer, head), group_slice in zip(self.head_pairs, self.group_slices):
            weights = coefficients[group_slice]
            norm = float(np.linalg.norm(weights))
            if norm > threshold:
                selected.append({"layer": layer, "head": head, "norm": norm})
        return selected


class GroupLassoLogisticRegression:
    def __init__(
        self,
        group_slices: Sequence[slice],
        lambda_reg: float,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        self.group_slices = list(group_slices)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    @staticmethod
    def _logistic_objective(
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float,
        lambda_reg: float,
        group_slices: Sequence[slice],
    ) -> float:
        logits = X @ weights + bias
        probs = expit(logits)
        eps = 1e-12
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        if lambda_reg <= 0:
            return loss
        penalty = 0.0
        for group_slice in group_slices:
            group_weights = weights[group_slice]
            penalty += np.linalg.norm(group_weights)
        return loss + lambda_reg * penalty

    @staticmethod
    def _initial_step_size(X: np.ndarray) -> float:
        n_samples = max(X.shape[0], 1)
        col_norms = np.sum(X ** 2, axis=0) / n_samples
        lipschitz = 0.25 * float(np.max(col_norms))
        if lipschitz <= 0:
            lipschitz = 1.0
        return 1.0 / lipschitz

    def _prox(self, weights: np.ndarray, step_lambda: float) -> np.ndarray:
        if step_lambda <= 0:
            return weights
        updated = weights.copy()
        for group_slice in self.group_slices:
            group_weights = updated[group_slice]
            norm = np.linalg.norm(group_weights)
            if norm == 0:
                continue
            shrinkage = max(0.0, 1.0 - step_lambda / norm)
            updated[group_slice] = group_weights * shrinkage
        return updated

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLassoLogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched shapes between X and y")

        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        X_scaled = self._transform(X)

        weights = np.zeros(X_scaled.shape[1], dtype=np.float64)
        bias = 0.0
        step_size = self._initial_step_size(X_scaled)
        prev_objective = self._logistic_objective(
            X_scaled, y, weights, bias, self.lambda_reg, self.group_slices
        )

        for iteration in range(self.max_iter):
            logits = X_scaled @ weights + bias
            probs = expit(logits)
            errors = probs - y
            grad_w = X_scaled.T @ errors / X_scaled.shape[0]
            grad_b = float(np.sum(errors) / X_scaled.shape[0])

            tentative_weights = weights - step_size * grad_w
            tentative_bias = bias - step_size * grad_b
            tentative_weights = self._prox(tentative_weights, step_size * self.lambda_reg)

            objective = self._logistic_objective(
                X_scaled,
                y,
                tentative_weights,
                tentative_bias,
                self.lambda_reg,
                self.group_slices,
            )

            if objective > prev_objective + 1e-9:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
                continue

            weights, bias = tentative_weights, tentative_bias
            if abs(prev_objective - objective) < self.tol:
                prev_objective = objective
                break
            prev_objective = objective

        self.coef_ = weights
        self.intercept_ = bias
        self.n_iter_ = iteration + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        probs = expit(logits)
        return np.vstack([1.0 - probs, probs]).T

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._transform(X) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class ActivationDataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            self.activations = handle["activations"][:].astype(np.float32)
            self.targets = {
                name: handle[name][:].astype(np.float32)
                for name in TARGETS
                if name in handle
            }
            dataset_bytes = handle["datasets"][:]
            self.datasets = np.array(
                [ds.decode("utf-8") if isinstance(ds, bytes) else str(ds) for ds in dataset_bytes]
            )
            self.template_ids = handle["template_ids"][:].astype(np.int32)
        self.metadata = utils.load_activation_metadata(self.path)
        self.feature_names = utils.load_feature_names(str(self.path))
        self.num_layers = int(self.metadata["num_layers"])
        self.num_heads = int(self.metadata["num_heads"])
        self.head_dim = int(self.metadata["head_dim"])

        self.train_indices = np.where(self.datasets == utils.DATASET_NAMES["probe_train"])[0]
        self.aux_indices = np.where(self.datasets == utils.DATASET_NAMES["probe_validate"])[0]
        self.test_indices = np.where(self.datasets == utils.DATASET_NAMES["probe_test"])[0]
        self.template_folds = self._build_template_folds()

    def _build_template_folds(self) -> List[np.ndarray]:
        folds: List[np.ndarray] = []
        train_templates = self.template_ids[self.train_indices]
        for template_id in sorted(np.unique(train_templates)):
            fold_indices = self.train_indices[train_templates == template_id]
            if fold_indices.size:
                folds.append(fold_indices)
        return folds

    def training_indices(self, include_aux: bool = True) -> np.ndarray:
        if include_aux and self.aux_indices.size:
            return np.unique(np.concatenate([self.train_indices, self.aux_indices]))
        return self.train_indices

    def head_slice(self, layer: int, head: int) -> slice:
        start = ((layer * self.num_heads) + head) * self.head_dim
        end = start + self.head_dim
        return slice(start, end)

    def build_subset(self, layer_range: range) -> FeatureSubset:
        head_pairs: List[Tuple[int, int]] = []
        for layer in layer_range:
            if layer >= self.num_layers:
                continue
            for head in range(self.num_heads):
                head_pairs.append((layer, head))
        return FeatureSubset.from_layers(head_pairs, self.num_heads, self.head_dim)


def compute_head_accuracies(dataset: ActivationDataset, targets: np.ndarray) -> np.ndarray:
    accuracies = np.zeros((dataset.num_layers, dataset.num_heads), dtype=np.float32)
    folds = dataset.template_folds
    if len(folds) < 2:
        return accuracies
    for layer in range(dataset.num_layers):
        for head in range(dataset.num_heads):
            head_slice = dataset.head_slice(layer, head)
            features = dataset.activations[:, head_slice]
            fold_scores: List[float] = []
            for idx, val_indices in enumerate(folds):
                train_parts = [fold for j, fold in enumerate(folds) if j != idx]
                if not train_parts:
                    fold_scores.append(np.nan)
                    continue
                train_indices = np.concatenate(train_parts)
                X_train = features[train_indices]
                y_train = targets[train_indices]
                X_val = features[val_indices]
                y_val = targets[val_indices]
                if len(np.unique(y_train)) < 2:
                    fold_scores.append(np.nan)
                    continue
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                fold_scores.append(accuracy_score(y_val, y_pred))
            accuracies[layer, head] = float(np.nanmean(fold_scores))
    return accuracies


def plot_head_heatmap(
    model_key: str,
    target_name: str,
    accuracies: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        accuracies,
        annot=False,
        cmap="viridis",
        xticklabels=range(accuracies.shape[1]),
        yticklabels=range(accuracies.shape[0]),
    )
    plt.title(f"{model_key} – Attention head accuracy ({target_name})")
    plt.xlabel("Attention head")
    plt.ylabel("Layer")
    plot_path = output_dir / f"{model_key}_{target_name}_head_heatmap.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def cross_validate_group_lasso(
    dataset: ActivationDataset,
    subset: FeatureSubset,
    targets: np.ndarray,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    folds = dataset.template_folds
    if len(folds) < 2:
        return None, float("nan"), {}

    features = dataset.activations[:, subset.feature_indices]
    lambda_scores: Dict[float, List[float]] = {}
    best_lambda: Optional[float] = None
    best_score = -np.inf

    for lambda_value in LAMBDA_GRID:
        fold_scores: List[float] = []
        for idx, val_indices in enumerate(folds):
            train_indices = np.concatenate(
                [fold for j, fold in enumerate(folds) if j != idx]
            )
            X_train = features[train_indices]
            y_train = targets[train_indices]
            X_val = features[val_indices]
            y_val = targets[val_indices]
            if len(np.unique(y_train)) < 2 or X_train.size == 0 or X_val.size == 0:
                fold_scores.append(np.nan)
                continue
            model = GroupLassoLogisticRegression(subset.group_slices, lambda_value)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_scores.append(accuracy_score(y_val, y_pred))
        lambda_scores[float(lambda_value)] = fold_scores
        mean_score = float(np.nanmean(fold_scores))
        if np.isnan(mean_score):
            continue
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = float(lambda_value)

    return best_lambda, best_score, lambda_scores


def evaluate_model(
    dataset: ActivationDataset,
    subset: FeatureSubset,
    targets: np.ndarray,
    lambda_value: float,
) -> Tuple[Optional[GroupLassoLogisticRegression], Dict[str, float]]:
    train_indices = dataset.training_indices(include_aux=True)
    if train_indices.size == 0:
        return None, {}

    features = dataset.activations[:, subset.feature_indices]
    X_train = features[train_indices]
    y_train = targets[train_indices]
    if len(np.unique(y_train)) < 2:
        return None, {}

    model = GroupLassoLogisticRegression(subset.group_slices, lambda_value)
    model.fit(X_train, y_train)

    test_indices = dataset.test_indices
    if test_indices.size == 0:
        return model, {}

    X_test = features[test_indices]
    y_test = targets[test_indices]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }
    return model, metrics


def append_report(
    path: Path,
    model_key: str,
    model_id: str,
    target_name: str,
    results: List[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"Model: {model_key} ({model_id})\n")
        handle.write(f"Target: {target_name}\n")
        for result in results:
            handle.write(f"Configuration: {result['configuration']}\n")
            handle.write(
                f"  Selected lambda: {result['lambda'] if result['lambda'] is not None else 'N/A'}\n"
            )
            if result.get("cv_score") is not None:
                handle.write(f"  Cross-validation accuracy: {result['cv_score']:.4f}\n")
            metrics = result.get("metrics") or {}
            if metrics:
                handle.write("  Test metrics:\n")
                for name, value in metrics.items():
                    handle.write(f"    {name}: {value:.4f}\n")
            else:
                handle.write("  Test metrics: N/A\n")
            selected_heads = result.get("selected_heads", [])
            handle.write(f"  Selected heads ({len(selected_heads)}):\n")
            if selected_heads:
                for head_info in selected_heads:
                    handle.write(
                        "    Layer {layer:02d}, Head {head:02d}, norm={norm:.4f}\n".format(
                            **head_info
                        )
                    )
            else:
                handle.write("    (none)\n")
            handle.write("\n")


def main() -> None:
    EXPERIMENT2_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    significant_models = utils.load_significant_models(
        SIGNIFICANT_MODELS_PATH, SIGNIFICANCE_REPORT_PATH
    )
    eligible_models = [model for model in TARGET_MODEL_KEYS if model in significant_models]

    if not eligible_models:
        print("No significant models available for probe training.")
        return

    ignored = sorted(set(significant_models) - set(eligible_models))
    if ignored:
        print("Skipping unsupported models: " + ", ".join(ignored))

    for model_key in eligible_models:
        activation_path = ACTIVATIONS_DIR / f"{model_key}_activations.h5"
        if not activation_path.exists():
            print(f"Activation file missing for model '{model_key}': {activation_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Training probes for model: {model_key}")
        print("=" * 80)

        dataset = ActivationDataset(activation_path)
        model_id = str(dataset.metadata.get("model_id", "unknown"))

        for target_key, target_label in TARGETS.items():
            if target_key not in dataset.targets:
                print(f"Target '{target_key}' not found in {activation_path}")
                continue
            print(f"\n--- Target {target_label} ---")
            target_values = dataset.targets[target_key]

            head_accuracies = compute_head_accuracies(dataset, target_values)
            plot_head_heatmap(model_key, target_label, head_accuracies, PLOTS_DIR)

            half_layers = max(dataset.num_layers // 2, 1)
            layer_configs = {
                "first_half": range(0, half_layers),
                "all_layers": range(0, dataset.num_layers),
            }

            results: List[dict] = []
            for config_name, layer_range in layer_configs.items():
                subset = dataset.build_subset(layer_range)
                if subset.feature_indices.size == 0:
                    results.append(
                        {
                            "configuration": config_name,
                            "lambda": None,
                            "cv_score": None,
                            "metrics": {},
                            "selected_heads": [],
                        }
                    )
                    continue

                best_lambda, cv_score, _ = cross_validate_group_lasso(
                    dataset, subset, target_values
                )
                if best_lambda is None:
                    print(
                        f"Unable to determine lambda for configuration '{config_name}' – skipping."
                    )
                    results.append(
                        {
                            "configuration": config_name,
                            "lambda": None,
                            "cv_score": None,
                            "metrics": {},
                            "selected_heads": [],
                        }
                    )
                    continue

                model, metrics = evaluate_model(dataset, subset, target_values, best_lambda)
                selected_heads: List[dict] = []
                if model is not None and model.coef_ is not None:
                    selected_heads = subset.selected_heads(model.coef_)

                print(
                    f"Configuration '{config_name}': lambda={best_lambda:.4f}, CV acc={cv_score:.4f}"
                )
                if metrics:
                    print("  Test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                else:
                    print("  Test metrics unavailable (no test data)")

                results.append(
                    {
                        "configuration": config_name,
                        "lambda": best_lambda,
                        "cv_score": cv_score,
                        "metrics": metrics,
                        "selected_heads": selected_heads,
                    }
                )

            append_report(EVAL_PATH, model_key, model_id, target_label, results)


if __name__ == "__main__":
    main()
