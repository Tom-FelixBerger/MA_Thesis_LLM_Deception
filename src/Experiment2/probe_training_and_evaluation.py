from utils import utils, templates
import h5py
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_attention_outputs(model_key):
    path = utils.DATA_DIR / "Experiment2" / f"{model_key}_attention_outputs.h5"
    with h5py.File(path, "r") as f:
        att = f["attention"][:]
        y_c = f["target_c"][:].astype(int)
        y_p = f["target_p"][:].astype(int)
        tids = f["template_id"][:]

    L, H, D = utils.get_model_dims(model_key)
    N = att.shape[0]
    X = att.reshape(N, L * H * D)
    return X, y_c, y_p, tids


def structure_features(model_key, att_dim):
    L, H, D = utils.get_model_dims(model_key)
    head_slices = {}

    for l in range(L):
        for h in range(H):
            start = (l * H + h) * D
            head_slices[(l, h)] = slice(start, start + D)

    return head_slices


def template_folds(template_ids, train_ids):
    return [np.where(template_ids == t)[0] for t in train_ids]


def run_cv_logreg_head(X, y, folds, slc):
    accs = []
    for i in range(len(folds)):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

        Xtr, Xval = X[tr_idx][:, slc], X[val_idx][:, slc]
        ytr, yval = y[tr_idx], y[val_idx]

        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)

        preds = clf.predict(sc.transform(Xval))
        accs.append(accuracy_score(yval, preds))

    return np.mean(accs)


def evaluate(clf, sc, X, y):
    Xs = sc.transform(X)
    probs = clf.predict_proba(Xs)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return dict(
        roc_auc=roc_auc_score(y, probs),
        accuracy=accuracy_score(y, preds),
        precision=precision_score(y, preds),
        recall=recall_score(y, preds),
        f1=f1_score(y, preds)
    )

def best_heads(heatmap, top_k, mode="all_layers"):
    L, H = heatmap.shape

    if mode == "first_half":
        # create masked view without copying more than necessary
        masked = heatmap.copy()
        masked[L // 2:] = -np.inf
        flat = masked.ravel()
    else:  # "all_layers"
        flat = heatmap.ravel()

    best = np.argpartition(flat, -top_k)[-top_k:]   # faster than full sort
    best = best[np.argsort(flat[best])[::-1]]       # order descending

    return [(i // H, i % H) for i in best]


def train_eval_top_heads(X_train, X_test, y_train, y_test, head_slices, heads):
    # Combine slices into one big feature set
    slcs = [head_slices[(l, h)] for (l, h) in heads]
    full_slc = np.hstack([np.arange(s.start, s.stop) for s in slcs])

    Xtr_sub = X_train[:, full_slc]
    Xte_sub = X_test[:, full_slc]

    sc = StandardScaler().fit(Xtr_sub)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr_sub), y_train)

    metrics = evaluate(clf, sc, Xte_sub, y_test)
    metrics["selected_heads"] = [f"L{int(h[0])}, H{int(h[1])}" for h in heads]
    return clf, sc, metrics, full_slc


def main():
    results = []

    for model_key in utils.MODELS:
        if utils.MODELS[model_key]["excluded"]:
            continue

        print("Running:", model_key)
        X, y_c, y_p, tids = load_attention_outputs(model_key)

        train_ids, test_ids = templates.TRAIN_IDS, templates.TEST_IDS
        test_idx = np.isin(tids, test_ids)
        train_idx = np.isin(tids, train_ids)

        X_train, X_test = X[train_idx], X[test_idx]
        y_c_train, y_c_test = y_c[train_idx], y_c[test_idx]
        y_p_train, y_p_test = y_p[train_idx], y_p[test_idx]

        folds = template_folds(tids[train_idx], train_ids)

        L, H, D = utils.get_model_dims(model_key)
        head_slices = structure_features(model_key, X.shape[1])

        heat_c = np.zeros((L, H))
        heat_p = np.zeros((L, H))

        for l in range(L):
            print(f"Evaluating heads for layer {l}/{L}")
            for h in range(H):
                slc = head_slices[(l, h)]
                heat_c[l, h] = run_cv_logreg_head(X_train, y_c_train, folds, slc)
                heat_p[l, h] = run_cv_logreg_head(X_train, y_p_train, folds, slc)

        heatmap_path = utils.DATA_DIR / "Experiment2" / f"{model_key}_heatmap_data.pkl"
        with open(heatmap_path, "wb") as f:
            pickle.dump(dict(heat_c=heat_c, heat_p=heat_p), f)


        for target_name, y_tr, y_te, heatmap, mode in [
            ("C", y_c_train, y_c_test, heat_c, "all_layers"),
            ("P", y_p_train, y_p_test, heat_p, "all_layers"),
            ("C", y_c_train, y_c_test, heat_c, "first_half"),
            ("P", y_p_train, y_p_test, heat_p, "first_half"),
        ]:
            print(f"Training {mode} logistic regression probe for target {target_name}")
            heads = best_heads(heatmap, top_k=10, mode=mode)

            clf, sc, metrics, slc = train_eval_top_heads(
                X_train, X_test, y_tr, y_te, head_slices, heads
            )

            metrics["model"] = model_key
            metrics["target"] = target_name
            metrics["mode"] = mode
            results.append(metrics)

            with open(utils.DATA_DIR/"Experiment2"/f"{model_key}_{target_name}_{mode}_probe.pkl","wb") as f:
                pickle.dump(dict(clf=clf, scaler=sc, heads=heads, slice=slc), f)

    import csv
    out_path = utils.DATA_DIR/"Experiment2"/"probe_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()