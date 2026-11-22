"""Full BBBP scaffold-held-out ablation of chemical quantum inputs and PCA.

Run ``python -m experiments.run_full_dataset_chemistry_pca``. All learned
scalers/PCA are fit inside each outer training fold. Compares Pauling/TPSA/
multi-physchem graph encodings with classical RDKit descriptors and train-fit
PCA; outputs are written under Results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.run_rdkit_descriptor_ablation import descriptor_matrix, ALL_DESCRIPTORS
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.analysis.pca_features import molecular_descriptor_matrix
from molecular_encoding.quantum.simulator import statevector_features

DATA=Path("data/BBBP_clean_full.csv")
OUT=Path("Results")


def score_fold(X, y, train, test, name):
    model=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10))
    model.fit(X[train],y[train]); dec=model.decision_function(X[test]); pred=model.predict(X[test])
    return {"method":name,"accuracy":accuracy_score(y[test],pred),"f1":f1_score(y[test],pred,zero_division=0),"roc_auc":roc_auc_score(y[test],dec)}


def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(DATA); y=df.label.to_numpy(int); smi=df.canonical_smiles.astype(str).tolist()
    groups=np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>" for s in smi])
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
    configs=[("chemical_Pauling","pauling_en"),("chemical_TPSA","tpsa_contrib"),("chemical_multi_physchem","multi_physchem")]
    matrices={}
    for name,prop in configs:
        print(f"Encoding {name}",flush=True)
        matrices[name]=np.stack([statevector_features(s,Strategy.CHEMICAL_COVERAGE,prop=prop) for s in smi])
    descriptors=descriptor_matrix(smi)
    # Length is deliberately isolated as a notation/provenance baseline; atom
    # count is a separate chemical-size descriptor, not SMILES string length.
    source_length=df.source_smiles.astype(str).str.len().to_numpy(float)[:,None]
    matrices["source_string_length_only"]=source_length
    matrices["RDKit_13_descriptors"]=descriptors
    rows=[]; pca_rows=[]; explained=[]
    for fold,(tr,te) in enumerate(splits,1):
        for name,X in matrices.items():
            rows.append({"fold":fold,**score_fold(X,y,tr,te,name)})
        # Pipeline ensures scaler and PCA never see outer test observations.
        for n_pc in (2,4,8,13):
            n_pc=min(n_pc,descriptors.shape[1],len(tr)-1)
            model=make_pipeline(StandardScaler(),PCA(n_components=n_pc,svd_solver="full"),SVC(kernel="rbf",C=10))
            model.fit(descriptors[tr],y[tr]); dec=model.decision_function(descriptors[te]); pred=model.predict(descriptors[te])
            rows.append({"fold":fold,**{"method":f"RDKit_PCA_{n_pc}PC", "accuracy":accuracy_score(y[te],pred),"f1":f1_score(y[te],pred,zero_division=0),"roc_auc":roc_auc_score(y[te],dec)}})
            pca=model.named_steps["pca"]
            explained.append({"fold":fold,"n_components":n_pc,"cumulative_variance":float(pca.explained_variance_ratio_.sum())})
        # Fit score/plots in each fold held-out transform only; collect for PCA visualization.
        pca=PCA(n_components=2,svd_solver="full").fit(StandardScaler().fit(descriptors[tr]).transform(descriptors[tr]))
        scaler=StandardScaler().fit(descriptors[tr]); coords=pca.transform(scaler.transform(descriptors[te]))
        pca_rows.extend({"fold":fold,"row":int(i),"pc1":coords[k,0],"pc2":coords[k,1],"label":int(y[i])} for k,i in enumerate(te))
        print(f"Completed scaffold fold {fold}/5",flush=True)
    per=pd.DataFrame(rows); per.to_csv(OUT/"chemistry_pca_fold_metrics.csv",index=False)
    summary=per.groupby("method",sort=False).agg(accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),f1_mean=("f1","mean"),f1_sd=("f1","std"),roc_auc_mean=("roc_auc","mean"),roc_auc_sd=("roc_auc","std")).reset_index()
    summary.to_csv(OUT/"chemistry_pca_summary.csv",index=False)
    pd.DataFrame(explained).to_csv(OUT/"chemistry_pca_explained_variance.csv",index=False)
    coords=pd.DataFrame(pca_rows); coords.to_csv(OUT/"rdkit_pca_oof_coordinates.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for lab,label in ((0,"non-BBB"),(1,"BBB")):
        d=coords[coords.label==lab]; ax.scatter(d.pc1,d.pc2,s=10,alpha=.5,label=label)
    ax.set(xlabel="PC1 (fold-fitted descriptors)",ylabel="PC2 (fold-fitted descriptors)",title="RDKit descriptor PCA · held-out molecules"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"rdkit_descriptor_pca.png",dpi=180); plt.close(fig)
    settings={"dataset":str(DATA),"n":len(y),"class_counts":pd.Series(y).value_counts().to_dict(),"folds":"5-fold StratifiedGroupKFold by Bemis-Murcko scaffold; acyclic together","seed":42,"quantum_readout":"78 Zi/ZiZj, exact NumPy statevector","fixed_C":10,"pca":"StandardScaler + PCA fitted within each outer training fold; component counts 2,4,8,13","notation_length":"source SMILES character count, isolated artifact baseline","descriptors":list(ALL_DESCRIPTORS),"multi_physchem_definition":"multi-axis Pauling/Gasteiger/TPSA/Crippen/donor-acceptor/aromaticity upload; differs from scalar angle mixtures","caveat":"exploratory repeated model comparisons; no independent external test; source length is a writing-artifact diagnostic"}
    (OUT/"chemistry_pca_settings.json").write_text(json.dumps(settings,indent=2)+"\n")
    print(summary.to_string(index=False,float_format=lambda v:f"{v:.3f}"))


if __name__=="__main__": main()
