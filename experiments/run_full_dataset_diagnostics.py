"""Full BBBP out-of-fold diagnostics: notation, atom order, chemistry, quantum SVM.

Run: ``python -m experiments.run_full_dataset_diagnostics``.
Writes confusion matrices, per-molecule out-of-fold predictions, metrics and
PNG diagnostics under ``Results/``. Outer folds are scaffold-disjoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import encode_states, fidelity_kernel
from molecular_encoding.quantum.simulator import statevector_features

DATA = Path("data/BBBP_clean_full.csv")
OUT = Path("Results")
SEEDS = (11, 23, 37)


def _score_arm(name, y, splits, features=None, kernel=None):
    pred = np.full(len(y), -1, dtype=int)
    decision = np.full(len(y), np.nan)
    rows = []
    for fold, (train, test) in enumerate(splits, 1):
        if kernel is None:
            model = make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf"))
            model.fit(features[train], y[train])
            pred[test] = model.predict(features[test])
            decision[test] = model.decision_function(features[test])
        else:
            model = SVC(C=10, kernel="precomputed")
            model.fit(kernel[np.ix_(train, train)], y[train])
            pred[test] = model.predict(kernel[np.ix_(test, train)])
            decision[test] = model.decision_function(kernel[np.ix_(test, train)])
        rows.append({"fold": fold, "train_n": len(train), "test_n": len(test),
                     "train_positive": int(y[train].sum()), "test_positive": int(y[test].sum()),
                     "accuracy": accuracy_score(y[test], pred[test]),
                     "f1": f1_score(y[test], pred[test], zero_division=0),
                     "roc_auc": roc_auc_score(y[test], decision[test])})
    if np.any(pred < 0):
        raise RuntimeError(f"Some molecules never received an out-of-fold prediction: {name}")
    return pred, decision, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    y = df.label.to_numpy(dtype=int)
    src = df.source_smiles.astype(str).tolist()
    can = df.canonical_smiles.astype(str).tolist()
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>" for s in can])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))
    # Three deterministic randomized serializations, representing notation sensitivity.
    random_smiles = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        random_smiles[seed] = []
        for s in can:
            mol = Chem.MolFromSmiles(s)
            random_smiles[seed].append(Chem.MolToSmiles(mol, canonical=False, doRandom=True,
                                                          isomericSmiles=True, rootedAtAtom=int(rng.integers(mol.GetNumAtoms()))))
    arms = {
        "Source SMILES order": np.array([statevector_features(s, Strategy.SMILES_ORDER, "pauling_en") for s in src]),
        "Canonical traversal": np.array([statevector_features(s, Strategy.CANONICAL_TRAVERSAL, "pauling_en") for s in can]),
        "Chemical coverage": np.array([statevector_features(s, Strategy.CHEMICAL_COVERAGE, "pauling_en") for s in can]),
    }
    for seed, strings in random_smiles.items():
        arms[f"Randomized SMILES seed {seed}"] = np.array([statevector_features(s, Strategy.SMILES_ORDER, "pauling_en") for s in strings])

    # Fidelity kernel on the invariant chemical-coverage encoding (full complex state).
    states = encode_states(can, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    kernel = fidelity_kernel(states)
    result_rows, prediction_rows, confusion_rows = [], [], []
    for name, X in arms.items():
        pred, dec, rows = _score_arm(name, y, splits, features=X)
        for row in rows:
            result_rows.append({"method": name, **row})
        cm = confusion_matrix(y, pred, labels=[0, 1])
        confusion_rows.extend({"method": name, "true_label": i, "predicted_label": j, "count": int(cm[i,j])}
                              for i in range(2) for j in range(2))
        prediction_rows.extend({"source_row": int(df.iloc[i].source_row), "method": name,
                                "true_label": int(y[i]), "predicted_label": int(pred[i]),
                                "decision_score": float(dec[i]), "smiles_length": len(src[i]),
                                "first_character": src[i][0], "lowercase_aromatic_count": sum(c in "cnos" for c in src[i])}
                               for i in range(len(y)))
    pred, dec, rows = _score_arm("Quantum fidelity kernel + SVM", y, splits, kernel=kernel)
    result_rows.extend({"method": "Quantum fidelity kernel + SVM", **r} for r in rows)
    cm = confusion_matrix(y, pred, labels=[0,1])
    confusion_rows.extend({"method":"Quantum fidelity kernel + SVM", "true_label":i,
                           "predicted_label":j,"count":int(cm[i,j])} for i in range(2) for j in range(2))
    prediction_rows.extend({"source_row":int(df.iloc[i].source_row),"method":"Quantum fidelity kernel + SVM",
                            "true_label":int(y[i]),"predicted_label":int(pred[i]),"decision_score":float(dec[i]),
                            "smiles_length":len(src[i]),"first_character":src[i][0],
                            "lowercase_aromatic_count":sum(c in "cnos" for c in src[i])} for i in range(len(y)))

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    confusion = pd.DataFrame(confusion_rows)
    results.to_csv(OUT/"full_bbbp_fold_metrics.csv", index=False)
    predictions.to_csv(OUT/"full_bbbp_oof_predictions.csv", index=False)
    confusion.to_csv(OUT/"full_bbbp_confusion_counts.csv", index=False)
    summary = results.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy","mean"), accuracy_sd=("accuracy","std"),
        f1_mean=("f1","mean"), f1_sd=("f1","std"), auc_mean=("roc_auc","mean"), auc_sd=("roc_auc","std"),
        total_train=("train_n","sum"), total_test=("test_n","sum")).reset_index()
    summary.to_csv(OUT/"full_bbbp_summary.csv", index=False)

    names = list(dict.fromkeys(confusion.method))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.flat, names):
        sub = confusion[confusion.method == name].pivot(index="true_label", columns="predicted_label", values="count").values
        ConfusionMatrixDisplay(sub, display_labels=["non-BBB", "BBB"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[len(names):]: ax.axis("off")
    fig.suptitle("Full cleaned BBBP · scaffold-held-out out-of-fold confusion matrices")
    fig.tight_layout(); fig.savefig(OUT/"confusion_matrices.png", dpi=180); plt.close(fig)

    # Plot performance against source notation characteristics: does serialization style track label?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color, name in ((0,"#4477AA","non-BBB"),(1,"#CC6677","BBB")):
        mask = y == label
        axes[0].scatter(np.array([len(s) for s in src])[mask], np.array([sum(c in "cnos" for c in s) for s in src])[mask],
                        s=12, alpha=.5, color=color, label=name)
    axes[0].set_xlabel("Source SMILES string length (writing)"); axes[0].set_ylabel("Lowercase aromatic tokens in source string"); axes[0].legend()
    # Display held-out prediction score vs source string length for order-vs-chemistry.
    for name, color in (("Source SMILES order","#CC6677"),("Chemical coverage","#4477AA")):
        sub = predictions[predictions.method == name]
        axes[1].scatter(sub.smiles_length, sub.decision_score, s=11, alpha=.45, color=color, label=name)
    axes[1].set_xlabel("Source SMILES string length (writing)"); axes[1].set_ylabel("Out-of-fold SVM decision score"); axes[1].legend()
    fig.suptitle("Does the written SMILES relate to label or model score?"); fig.tight_layout()
    fig.savefig(OUT/"smiles_writing_diagnostics.png", dpi=180); plt.close(fig)

    # Two-dimensional PCA view of Z/ZZ features, colored by labels, for order vs chemistry.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, ("Source SMILES order","Chemical coverage")):
        X = arms[name]
        Z = np.empty_like(X)
        for tr, te in splits:
            scaler=StandardScaler().fit(X[tr]); Z[te]=scaler.transform(X[te])
        coords=PCA(n_components=2, random_state=42).fit_transform(Z)
        ax.scatter(coords[y==0,0],coords[y==0,1],s=10,alpha=.45,label="non-BBB")
        ax.scatter(coords[y==1,0],coords[y==1,1],s=10,alpha=.45,label="BBB")
        ax.set_title(name); ax.set_xlabel("PCA 1 (78 Z/ZZ features)"); ax.set_ylabel("PCA 2"); ax.legend()
    fig.suptitle("Out-of-fold-scaled quantum feature space"); fig.tight_layout()
    fig.savefig(OUT/"quantum_feature_separation.png",dpi=180); plt.close(fig)

    (OUT/"full_bbbp_settings.json").write_text(json.dumps({
        "dataset":str(DATA),"n_molecules":len(y),"class_counts":pd.Series(y).value_counts().to_dict(),
        "scaffolds":int(len(set(groups))),"outer_cv":"5-fold StratifiedGroupKFold by Bemis-Murcko scaffold; acyclic together",
        "test_predictions":"each molecule predicted once out-of-fold","fixed_C":10,
        "randomized_smiles_seeds":list(SEEDS),"quantum_kernel":"fidelity |<psi_i|psi_j>|^2 on full statevectors; precomputed SVC",
        "train_test_sizes":[{"fold":i,"train":len(tr),"test":len(te)} for i,(tr,te) in enumerate(splits,1)],
        "caveat":"PCA is descriptive; randomized serialization arms use one deterministic randomized traversal per molecule and seed; no nested hyperparameter tuning in this diagnostic"
    },indent=2)+"\n")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print("Saved diagnostic CSV and PNG files in Results/")


if __name__ == "__main__": main()
