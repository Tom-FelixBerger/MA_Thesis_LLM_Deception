from utils import utils
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from group_lasso import GroupLasso


def load_attention_outputs(model_key):
    path = utils.DATA_DIR / "Experiment2" / f"{model_key}_attention_outputs.h5"

    with h5py.File(path, "r") as f:
        attention = f["attention"][:]     # (N, L, H, D)
        y_c = f["target_c"][:]            # (N,)
        y_p = f["target_p"][:]            # (N,)
        template_ids = f["template_id"][:] # (N,)

    num_layers, num_heads, head_dim = utils.get_model_dims(model_key)
    N = attention.shape[0]
    X = attention.reshape(N, num_layers * num_heads * head_dim)

    return X, y_c, y_p, template_ids

def structure_features(model_key, att_dim):
    num_layers, num_heads, head_dim = utils.get_model_dims(model_key)

    # construction of feature names
    feature_names = []
    for l in range(num_layers):
        for h in range(num_heads):
            for d in range(head_dim):
                feature_names.append(f"L{l:02d}_H{h:02d}_d{d:03d}")

    # assignment of heads to feature ranges and of features to head_ids
    groups = np.empty(att_dim, dtype=int)
    head_slices = {}
    for l in range(num_layers):
        for h in range(num_heads):
            start = (l * num_heads + h) * head_dim
            head_slices[(l, h)] = slice(start, start + head_dim)
            groups[start : start + head_dim] = l * num_heads + h

    return feature_names, head_slices, groups

######## FROM BELOW HERE IS COPY PACED ##############
def first_half_mask(model_key: str) -> np.ndarray:
    """Boolean mask over heads (length L*H) selecting only the first half of layers."""
    num_layers, num_heads, head_dim = utils.get_model_dims(model_key)
    total_heads = num_layers * num_heads
    mask = np.zeros(total_heads, dtype=bool)
    halfL = num_layers // 2
    # enable heads for layers [0, halfL-1]
    for l in range(halfL):
        for h in range(num_heads):
            mask[l * num_heads + h] = True
    return mask


def select_groups(X, groups, group_mask):
    """Select only features that belong to heads where group_mask[gid] is True.
    Returns (X_reduced, groups_reindexed)
    """
    keep_feature_mask = group_mask[groups]
    X_sel = X[:, keep_feature_mask]

    # reindex the kept groups contiguously 0..K-1
    old_to_new = {}
    new_gid = 0
    groups_sel = groups[keep_feature_mask].copy()
    for i, g in enumerate(groups_sel):
        if g not in old_to_new:
            old_to_new[g] = new_gid
            new_gid += 1
        groups_sel[i] = old_to_new[g]
    return X_sel, groups_sel


def build_template_folds(template_ids, n_folds = 5):
    """Return dict fold_idx -> boolean mask over samples belonging to that fold.
    Strategy:
      - Identify unique template_ids and their counts (descending).
      - Take the top n_folds template ids as the CV folds (one template per fold).
      - Any remaining templates are assigned to the held-out test set (not part of folds).
    """
    uniq, counts = np.unique(template_ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_templates = uniq[order][:n_folds]
    folds = {}
    for i, t in enumerate(top_templates):
        folds[i] = (template_ids == t)
    return folds  # fold indices 0..n_folds-1


def heldout_test_mask(template_ids, folds):
    """Samples whose template_id is NOT in the CV folds are test."""
    cv_mask = np.zeros_like(template_ids, dtype=bool)
    for m in folds.values():
        cv_mask |= m
    return ~cv_mask


def standardize_fit_transform(X_train, X_val, X_test):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


# ------------------------------
# Per-head probe (for visualization)
# ------------------------------
def per_head_cv_heatmap(
    model_key,
    X,
    y,
    folds,
    head_map,
    title,
    outfile,
):
    num_layers, num_heads, head_dim = utils.get_model_dims(model_key)
    acc = np.zeros((num_layers, num_heads), dtype=float)

    # For each head, run 5-fold CV accuracy using only that head's features
    for l in range(num_layers):
        for h in range(num_heads):
            sl = head_map[(l, h)]
            Xh = X[:, sl]
            # CV across folds
            scores = []
            for k in range(len(folds)):
                test_mask = folds[k]
                train_mask = ~test_mask

                X_tr, X_te = Xh[train_mask], Xh[test_mask]
                y_tr, y_te = y[train_mask], y[test_mask]

                X_tr_s, X_te_s, _, _ = standardize_fit_transform(X_tr, X_te, X_te)

                # Simple LR on this head only (no penalty tuning)
                clf = LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                )
                clf.fit(X_tr_s, y_tr)
                pred = clf.predict(X_te_s)
                scores.append(accuracy_score(y_te, pred))

            acc[l, h] = float(np.mean(scores))

    # Plot heatmap
    plt.figure(figsize=(max(6, num_heads * 0.4), max(6, num_layers * 0.4)))
    plt.imshow(acc, aspect="auto")
    plt.colorbar()
    plt.xlabel("Head index")
    plt.ylabel("Layer index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


# ------------------------------
# Group Lasso with CV (first-half vs all layers)
# ------------------------------
def group_lasso_cv(
    X,
    y,
    folds,
    groups,
    lambda_grid,
):
    """Return best model (refit on all CV folds) and best lambda based on mean CV ROC-AUC."""
    best_lambda = None
    best_score = -np.inf

    for lam in lambda_grid:
        fold_scores = []
        for k in range(len(folds)):
            val_mask = folds[k]
            train_mask = ~val_mask

            X_tr, X_va = X[train_mask], X[val_mask]
            y_tr, y_va = y[train_mask], y[val_mask]

            X_tr_s, X_va_s, _, scaler = standardize_fit_transform(X_tr, X_va, X_va)

            gl = GroupLasso(
                groups=groups,
                group_reg=lam,
                l1_reg=0.0,
                frobenius_lipschitz=False,
                scale_reg="group_size",  # penalize proportional to sqrt(group size)
                subsampling_scheme=None,
                supress_warning=True,
                n_iter=2000,
                tol=1e-4,
                # classification loss:
                fit_intercept=True,
                random_state=0,
            )
            gl.fit(X_tr_s, y_tr)
            proba = gl.predict_proba(X_va_s)
            # GroupLasso returns probabilities; take column 1 if shape (N,2)
            if proba.ndim == 2 and proba.shape[1] == 2:
                pos = proba[:, 1]
            else:
                pos = proba.ravel()
            fold_scores.append(roc_auc_score(y_va, pos))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lam

    # Refit on all CV folds with best lambda
    X_all = X
    y_all = y
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_all_s = scaler.fit_transform(X_all)
    gl_best = GroupLasso(
        groups=groups,
        group_reg=best_lambda,
        l1_reg=0.0,
        frobenius_lipschitz=False,
        scale_reg="group_size",
        subsampling_scheme=None,
        supress_warning=True,
        n_iter=3000,
        tol=1e-4,
        fit_intercept=True,
        random_state=0,
    )
    gl_best.fit(X_all_s, y_all)

    # Attach scaler for later use
    gl_best._scaler = scaler
    return gl_best, best_lambda


def eval_on_test(model, X_test, y_test):
    scaler = getattr(model, "_scaler")
    X_ts = scaler.transform(X_test)
    proba = model.predict_proba(X_ts)
    if proba.ndim == 2 and proba.shape[1] == 2:
        pos = proba[:, 1]
    else:
        pos = proba.ravel()
    pred = (pos >= 0.5).astype(int)

    metrics = dict(
        roc_auc=roc_auc_score(y_test, pos),
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
    )
    return metrics


def save_report(metrics, out_txt, header, extra= None):
    lines = [header, ""]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    for k, v in metrics.items():
        lines.append(f"{k}: {v:.4f}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")



def main():
    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue
        
        X, y_c, y_p, template_ids = load_attention_outputs(model_key)
        n_vign, att_dim = X.shape
        
        feature_names, head_slices, groups = structure_features(model_key, att_dim)
        num_layers, num_heads, head_dim = utils.get_model_dims(model_key)

        ##### FROM BELOW HERE IS COPY PACED #####
        first_half_heads_mask = first_half_mask(model_key)  # length L*H

        # 5 template-based folds (one template per fold)
        folds = build_template_folds(template_ids, n_folds=5)
        test_mask = heldout_test_mask(template_ids, folds)



        print("---------------- Per-head CV heatmaps ----------------")
        per_head_cv_heatmap(
            model_key=model_key,
            X=X,
            y=y_c,
            folds=folds,
            head_map=head_slices,
            title=f"{model_key} — Per-head CV accuracy (target C)",
            outfile= utils.PLOTS_DIR / f"{model_key}_per_head_cv_accuracy_C.png",
        )
        per_head_cv_heatmap(
            model_key=model_key,
            X=X,
            y=y_p,
            folds=folds,
            head_map=head_slices,
            title=f"{model_key} — Per-head CV accuracy (target P)",
            outfile= utils.PLOTS_DIR / f"{model_key}_per_head_cv_accuracy_P.png",
        )

        print("---------------- Group Lasso CV (first half vs all) ----------------")
        # Lambda grid (penalty strengths). Adjust if needed.
        lambda_grid = [0.001, 0.003, 0.01, 0.03, 0.1]

        # Build train/val pool = only the CV folds (exclude held-out templates)
        cv_pool_mask = ~test_mask
        X_cv = X[cv_pool_mask]
        y_c_cv = y_c[cv_pool_mask]
        y_p_cv = y_p[cv_pool_mask]

        # Build fold masks in the reduced space
        # We keep fold indices but map masks to the reduced subset
        idx_all = np.arange(n_vign)
        idx_cv = idx_all[cv_pool_mask]
        folds_cv = {}
        for k, m in folds.items():
            folds_cv[k] = np.isin(idx_cv, idx_all[m])

        # ----- Target C -----
        # (A) first-half layers
        X_cv_half, groups_half = select_groups(X_cv, groups, first_half_heads_mask)
        probe_c_half, lam_c_half = group_lasso_cv(X_cv_half, y_c_cv, folds_cv, groups_half)

        # (B) all layers
        probe_c_all, lam_c_all = group_lasso_cv(X_cv, y_c_cv, folds_cv, groups)

        # ----- Target P -----
        # (A) first-half layers
        X_cv_half, groups_half = select_groups(X_cv, groups, first_half_heads_mask)
        probe_p_half, lam_p_half = group_lasso_cv(X_cv_half, y_p_cv, folds_cv, groups_half)

        # (B) all layers
        probe_p_all, lam_p_all = group_lasso_cv(X_cv, y_p_cv, folds_cv, groups)

        print("---------------- Held-out test evaluation ----------------")
        # Test set = templates not used in CV folds
        X_test = X[test_mask]
        y_c_test = y_c[test_mask]
        y_p_test = y_p[test_mask]

        # Prepare feature selection for first-half models on test set
        X_test_half, _ = select_groups(X_test, groups, first_half_heads_mask)

        # Evaluate and save reports
        # Target C
        metrics_c_half = eval_on_test(probe_c_half, X_test_half, y_c_test)
        save_report(
            metrics_c_half,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_C_first_half_report.txt",
            header=f"{model_key} — Probe for target C (first-half layers)",
            extra={"lambda": str(lam_c_half)},
        )

        metrics_c_all = eval_on_test(probe_c_all, X_test, y_c_test)
        save_report(
            metrics_c_all,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_C_all_layers_report.txt",
            header=f"{model_key} — Probe for target C (all layers)",
            extra={"lambda": str(lam_c_all)},
        )

        # Target P
        metrics_p_half = eval_on_test(probe_p_half, X_test_half, y_p_test)
        save_report(
            metrics_p_half,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_P_first_half_report.txt",
            header=f"{model_key} — Probe for target P (first-half layers)",
            extra={"lambda": str(lam_p_half)},
        )

        metrics_p_all = eval_on_test(probe_p_all, X_test, y_p_test)
        save_report(
            metrics_p_all,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_P_all_layers_report.txt",
            header=f"{model_key} — Probe for target P (all layers)",
            extra={"lambda": str(lam_p_all)},
        )

        # ---------------- Save final probes (first-half only, as requested) ----------------
        # We attach metadata to each saved probe for clarity.
        def save_probe(obj, path, target_name, lambda_val):
            meta = {
                "model_key": model_key,
                "target": target_name,
                "layers": num_layers,
                "heads": num_heads,
                "head_dim": head_dim,
                "lambda": lambda_val,
                "groups_note": "groups index head-wise over features; scaler attached at attribute _scaler",
                "feature_layout": "Concatenated by heads within layers; names Lxx_Hxx_dxxx",
            }
            bundle = {"probe": obj, "metadata": meta}
            with open(path, "wb") as f:
                pickle.dump(bundle, f)

        save_probe(
            probe_c_half,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_C_first_half.pkl",
            target_name="C",
            lambda_val=lam_c_half,
        )
        save_probe(
            probe_p_half,
            utils.DATA_DIR / "Experiment2" / f"{model_key}_probe_P_first_half.pkl",
            target_name="P",
            lambda_val=lam_p_half,
        )


if __name__ == "__main__":
    main()