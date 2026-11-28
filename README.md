# Full BBBP + 10+2 Quantum Feature Map

Branch **`analysis_features`** – all work is in this branch.

This repository contains a clean pipeline for the BBBP dataset and a 12‑qubit quantum feature map (10 backbone + 2 tail qubits).  The README explains **what was cleaned**, **how the embedding is built**, and the **three main experiments** that compare different qubit‑selection strategies.

![Graph‑to‑circuit diagram](results/embedding_diagram.png)

*Figure 1 – Heavy‑atom graph of aspirin and its conversion into the fixed‑width 10+2 circuit.  Blue nodes = backbone (q0‑q9); orange nodes = tail (q10/q11).* 

---

## 1. Dataset (cleaned)

- Source: `data/raw/BBBP.csv` – 2050 rows from DeepChem.
- Cleaning stages (see `analysis_features/verify.py`):
  - **Parse** with RDKit – 11 invalid SMILES.
  - **Remove salts/solvents** – 105 rows kept, parent molecule chosen by a deterministic rule (largest heavy‑atom fragment).
  - **Resolve label conflicts** – 11 structures with contradictory labels (e.g. propranolol) are removed.
  - **Deduplicate** – 62 duplicate structures retained by first occurrence.
  - **Final count** – 1955 unique molecules (461 non‑permeable, 1494 permeable).
- The clean CSV, audit logs, RDKit version and a timestamp are written to `data/processed/`.
- RDKit **2024.03.5**.

### Key decisions
| Decision | Description |
|---|---|
| **Charges** | Kept as‑is; no neutralisation. |
| **Tautomers** | Not merged; marked if ambiguous. |
| **Stereochemistry** | Retained; the quantum circuit is stereocontrolled‑blind. |
| **Electronegativities** | Hannan table extended with Br 2.96, I 2.66, P 2.19; missing elements raise an error. |
| **Tail size** | `atoms_beyond_10 = max(0, heavy_atom_count‑10)`; 94.7 % of molecules have a tail.

### 1.1 Cleaning requirements, answered point by point

This is the audit a reviewer should demand. Every number below is reproduced by
`python -m analysis_features verify` (34 checks) and logged in
`data/processed/BBBP_clean_log.txt`. The paste-ready methods prose is in
`docs/METHODS.md`.

1. **Full raw dataset, not the 250 subset.** Input is `data/raw/BBBP.csv` with
   **2050 rows** (`SMILES`, `p_np`). `BBBP_previous_250.csv` is *only* a
   regression check: all **250/250** of its molecules survive cleaning with the
   same label and are marked by `in_previous_250`. No result here is computed
   on 250 molecules.
2. **Parse every SMILES with RDKit; count the failures.** `Chem.MolFromSmiles`
   rejects **11** structures (systematic valence errors, mostly guanidine /
   `[NH2]` motifs). They are listed in `BBBP_invalid_smiles.csv` and were *not*
   repaired: repairing a valence error is a chemical edit, not a parsing step.
   Parsed rows: **2039** = 2050 − 11 (log Stage 1).
3. **Strip salts/solvents, keep the parent.** **105** rows are multi-fragment;
   we keep the largest **carbon-containing** fragment, breaking ties by heavy
   atoms, heteroatoms and canonical SMILES; **2** parent choices are chemically
   ambiguous and flagged in `BBBP_fragment_audit.csv`. The first entry,
   propranolol `[Cl].CC(C)NCC(O)COc1cccc2ccccc12`, becomes
   `CC(C)NCC(O)COc1cccc2ccccc12`. Because the paper assigns qubits in atom
   order, the counter-ion would otherwise occupy **q0**. `verify.py` asserts
   both sides: default embedding gives q0 = C, while the explicit control
   (`order="source"`, `strip_salts=False`) still gives q0 = Cl.
4. **Standardise and write canonical SMILES; document charges and tautomers.**
   Parents are written as `MolToSmiles(canonical=True, isomericSmiles=True)`.
   **Charges are kept, never neutralised** (34 charged structures; net charge in
   `formal_charge`) because protonation changes the permeating species.
   **Tautomers are not merged or canonicalised**: they can move a proton and
   change the species; only 8 tautomer groups with contradictory labels are
   flagged via `tautomer_label_conflict`. Stereochemistry is preserved in the
   SMILES but ignored by the circuit — a documented limitation (629 molecules
   carry `@`/`@@`; see `results/README.md`).
5. **Remove duplicates; exclude unsupported label conflicts.** **11 structures**
   appear with both labels (**22 rows**), propranolol among them (HCl salt
   labelled 1, free base labelled 0). With no external evidence for the
   correct label we **exclude both rows** rather than impute one. Then **62**
   identical-structure duplicates are collapsed, keeping the first occurrence.
   Final table: **1955 unique molecules** (461 label 0, 1494 label 1).
   `BBBP_clean_audit.csv` records `excluded_reason ∈ {included,
   label_conflict, duplicate}`; uniqueness is asserted by `verify.py`.
6. **Electronegativity coverage.** Hannan's original 10-element table is kept
   **verbatim** for reproducibility. Present elements: B, Br, C, Cl, F, I, N,
   O, P, S. Missing from Hannan: **Br, I, P** → added explicitly with Pauling
   values **2.96, 2.66, 2.19**. Missing after extension: **none**.
   `electronegativity()` raises `KeyError` for an unknown element instead of
   silently defaulting to 1.0 — the original notebook's silent fallback is
   removed.
7. **Requested columns.** `heavy_atom_count`, `bond_count` and
   `atoms_beyond_10 = max(0, heavy_atom_count − 10)` are in the clean CSV.
   **1851 molecules** (94.7 %) have atoms beyond the tenth, i.e. a tail is the
   normal case, not an exception. `verify.py` also checks
   `bond_count ≥ heavy_atom_count − 1`.
8. **Saved artefacts.** `data/processed/BBBP_clean_full.csv`,
   `BBBP_clean_audit.csv`, `BBBP_invalid_smiles.csv`,
   `BBBP_fragment_audit.csv`, `BBBP_clean_settings.json`,
   `BBBP_clean_log.txt`; the pipeline itself is `analysis_features/cleaning.py`
   (`python -m analysis_features clean`). RDKit **2024.03.5** is recorded in
   both the settings JSON and the log. *Caveat:* the analysis environment used
   for the figures has RDKit 2025.09.6, so the single `verify.py` version check
   reports a mismatch; the cleaning artefacts themselves were produced and
   audited with the recorded version and are unchanged.
9. **Per-stage log (verbatim counts).** 2050 input → Stage 1: −11 invalid
   (2039) → Stage 2: 105 fragment-stripped rows, 2202 fragments, 2 ambiguous
   parents → Stage 3: 2039 canonical isomeric SMILES → Stage 4: element audit
   above → Stage 5a: −22 conflict rows (11 structures) → Stage 5b: −62
   duplicates → Stage 5: **1955** rows (461 / 1494) → Stage 6: 6 stereo
   conflicts flagged → Stage 7: 8 tautomer conflicts flagged → Stage 8:
   250/250 previous-subset molecules recovered. A short explanation of every
   decision is in `docs/METHODS.md` and in the `decisions` block of
   `BBBP_clean_settings.json`.

---

## 2. Methods: from molecular graph to circuit

### 2.1 Embedding construction

The 12‑qubit circuit is built as follows:

1. **Parse SMILES → RDKit molecule → heavy‑atom graph**.
2. **Select 10 backbone atoms** using one of three strategies:
   * `paper_smiles_order` – first 10 atoms in the SMILES order.
   * `random_backbone` – 10 atoms chosen uniformly at random.
   * `community_importance` – scores each heavy atom by a weighted sum of nine RDKit descriptors (see Table 1), then picks the highest‑scoring atom per community detected by `networkx.greedy_modularity_communities`.
3. **Tail atoms** (remaining heavy atoms) are compressed into two qubits (q10 and q11) by alternating the sum of their Pauling electronegativities.
4. **Apply gates**:
   * `Ry(θi)` on qubit `qi` with `θi = ALPHA * χi²` (α = 0.1, χ = electronegativity).
   * `Rxx(bond_order/2)` for each bond inside the backbone.
5. **Readouts**: 12 `<Z>` moments and 66 `<ZZ>` moments (78 features in total).
6. **Classifier**: a classical SVM with RBF kernel (`C = 10`) trained on the 78‑dimensional feature vectors.

> The circuit is a **feature map**; the SVM makes the final prediction. Community selection takes the single highest-scoring atom of each community (no descriptor averaging); only the tail pools atoms, by alternating sums of Pauling electronegativity. Stereochemistry is retained in the isomeric SMILES but ignored by the gates, so enantiomers currently produce identical features — see `results/README.md` for the mechanism and design options.

The diagram below is **the actual circuit**, not a cartoon: `python -m analysis_features circuit-diagram` reads the angles from the same `build_statevector` function used for the results and draws a Qiskit `QuantumCircuit` with `draw("mpl")`. For aspirin (14 heavy atoms), q0–q9 receive `Ry(0.1·χ²)` (0.65 rad for C, 1.18 rad for O), q10/q11 receive the alternating tail sums (0.300 / 0.172 rad), and the bonds become `Rxx(order/2)` gates — including two interface gates per backbone–tail bond. Every angle and gate is listed in `results/quantum_circuit.json`.

![Qiskit drawing of the 12-qubit feature-map circuit](results/quantum_circuit_qiskit.png)

*Figure 2 – Qiskit 2.6.2 drawing of the real 10+2 circuit for aspirin. The 78 Pauli moments measured after these gates are the SVM input.*

### 2.2 Descriptor weights: deliberately equal

The community‑importance strategy scores each heavy atom with nine descriptors. **All nine carry the same weight, 1/9 ≈ 0.111.** The BBB literature justifies the *choice* of descriptors — they are the per‑atom quantities most often associated with passive permeation — but it does not give a defensible numerical ranking of their per‑atom contributions, and we fit no weight to BBBP labels. Without evidence for a hierarchy, different numbers would be an arbitrary prior, so equal weights are the neutral starting point.

| Descriptor | RDKit source | Weight | Why the descriptor is included |
|---|---|---:|---|
| TPSA contribution | `rdMolDescriptors._CalcTPSAContribs` | 1/9 | The most used BBB filter: CNS compounds typically need PSA < 90 Å² [1, 2, 3]. |
| LogP contribution | `rdMolDescriptors._CalcCrippenContribs` | 1/9 | Lipophilicity is the other dominant term, with an optimum around logP 1.5–2.7 [1, 4]. |
| H‑bond donors | `BaseFeatures.fdef` | 1/9 | Donors carry a larger desolvation penalty than acceptors [1, 5]. |
| H‑bond acceptors | `BaseFeatures.fdef` | 1/9 | Polarity relevant to membrane crossing, but less penalising [1, 5]. |
| Molar refractivity | `_CalcCrippenContribs` | 1/9 | Size/polarizability is an independent term in Abraham‑type BBB equations [4, 6]. |
| Gasteiger partial charge | `AllChem.ComputeGasteigerCharges` | 1/9 | Local charge affects desolvation at the membrane interface [1, 7]. |
| Formal charge | `atom.GetFormalCharge()` | 1/9 | Ionised species cross poorly, though rare in this dataset (34 structures) [1, 7]. |
| Heteroatom flag | `atom.GetAtomicNum() != 6` | 1/9 | Coarse proxy for hydrogen bonding and polarity [1]. |
| Aromaticity | `atom.GetIsAromatic()` | 1/9 | Separates aromatic N from amide N and contributes to membrane affinity [1]. |

Each continuous descriptor is rescaled **within each molecule** as `|x_i − mean| / max_j |x_j − mean|`, so it lies in [0, 1] regardless of its physical unit (Å², log units, e). Binary flags are already in {0, 1}. The weights sum to one, so the score is a dimensionless weighted average: *relative shares of chemical salience*, not physically comparable coefficients. Note that this score selects **which atoms are resolved by the ten qubits** — it is not a permeability prediction and is not added to the classifier features; the SVM on the 78 Pauli moments makes the label decision.

**Does the choice matter?** `python -m analysis_features weight-probe` compares the equal default with a literature‑ranked vector (0.20 TPSA, 0.20 logP, …) and 20 random Dirichlet vectors, on identical folds, circuit and SVM: equal **0.7745**, literature‑ranked **0.7841**, random **0.7584 ± 0.0257** [0.7095, 0.8004]. The literature vector is 0.0096 AUC higher in this run, well inside the random spread — but we deliberately **do not adopt it**, because picking the vector by test AUC would be tuning on the evaluation set. The defensible claim is robustness: weight choice moves AUC by ≈ 0.01–0.03, whereas chemistry‑driven atom selection beats a random backbone by 0.7745 vs 0.6342 (≈ 0.14, about 5.5 SD). Full justification, sensitivity table, objections and references: `results/README.md`.

[1] Pajouhesh & Lenz, *NeuroRx* 2:541–553 (2005). [2] Veber et al., *J. Med. Chem.* 45:2615–2623 (2002). [3] Hitchcock & Pennington, *J. Med. Chem.* 49:7559–7583 (2006). [4] Hansch/Leo analysis in [1]. [5] Kenny, *J. Med. Chem.* 65:14261–14275 (2022). [6] Wildman & Crippen, *J. Chem. Inf. Comput. Sci.* 39:868–873 (1999). [7] Muehlbacher et al., *J. Comput.-Aided Mol. Des.* 25:1095–1106 (2011).

### 2.3 Summary of the three experiments

| Arm | Accuracy | Balanced accuracy | AUC |
|---|---|---|---|
| `paper_smiles_order` | 0.8399 | 0.7603 | 0.8602 |
| `random_backbone` | 0.7228 | 0.5764 | 0.6342 |
| `community_importance` | 0.7887 | 0.6900 | 0.7745 |

The **confusion matrices** for each arm are shown in a single figure below (generated by `python -m analysis_features three-arms`).

![Confusion matrices for the three arms](results/confusion_matrices_three_arms.png)

*Figure 3 – Out‑of‑fold confusion matrices for the three selection strategies.*

### 2.4 Pauling electronegativity vs. the new importance features

The new embedding (`analysis_features/embedding_with_importance.py`) parameterises the ten backbone `Ry` rotations with the per‑atom BBBP importance score `s` instead of the Pauling electronegativity: `θi = 0.1·sᵢ²` rather than `0.1·χᵢ²`. Everything else is unchanged, including the alternating‑electronegativity tail, so the two panels differ only in the backbone angles. The comparison runs both maps on the same three atom‑selection rules, folds and SVM (`python -m analysis_features embedding-comparison`):

| Embedding | SMILES order | Random backbone | Community importance |
|---|---:|---:|---:|
| Pauling `χ²` | AUC 0.860 | AUC 0.634 | AUC 0.775 |
| New features `s²` | AUC 0.867 | AUC 0.675 | AUC 0.732 |

![Pauling vs new features](results/embedding_comparison.png)

*Figure 4 – Two panels of out‑of‑fold confusion matrices: Pauling electronegativity (top) and the new importance‑based features (bottom), each for SMILES order, random backbone and community importance.*

### 2.5 How could we classify better? (evidence first, ideas second)

A reviewer is right to ask why we should change anything if the current map
already reaches AUC ≈ 0.86. The honest answer is that **the strongest number is
partly an artefact** and that a classical baseline already wins:

| Model (same 1955 molecules, scaffold folds) | AUC | Comment |
|---|---:|---|
| Classical RBF SVM on Morgan fingerprints | **0.8820** | beats every quantum arm |
| Quantum, deposited SMILES order | 0.8602 | provenance leak, not chemistry |
| Whole-molecule atom composition | 0.8133 | no circuit at all |
| Quantum, community importance | 0.7745 | honest structure-based arm |
| Quantum, random backbone | 0.6342 | below trivial baseline |

Ranked ideas, with the evidence status stated explicitly:

1. **Fix the comparison before chasing accuracy (already done).** Use the
   cleaned, canonical structures and scaffold-disjoint folds; report balanced
   accuracy and bootstrap CIs. The apparent 0.860 advantage of SMILES order
   disappears under canonicalisation, and a random backbone (0.634) is worse
   than always predicting "permeable" (0.764). Any future "improvement" must
   be measured against **0.8820 (Morgan SVM)**, not against the leaky arm.
2. **Hybrid quantum + classical kernel (best-supported next experiment).** A
   whole-molecule bag of elements already reaches 0.8133, so much of the signal
   is additive chemistry. Sum an RBF kernel on the 78 Pauli moments with an RBF
   kernel on Morgan/composition descriptors (both fitted in-fold) and test
   whether the quantum part adds complementary information. If the hybrid does
   not beat Morgan alone, the honest conclusion is that the quantum map is
   redundant here.
3. **Encode chiral pseudoscalars.** 629 molecules carry `@`/`@@`, but the
   circuit is mirror-blind. Add `Rz(±π/4)` from `GetChiralTag()` on the qubit of
   each selected stereocentre (and an `Rz = 0` ablation). This is a genuine new
   gate family with no extra qubits; see `results/README.md`.
4. **Richer readout / trainable angles.** The current 78 moments are first and
   second order; try third-order ⟨ZZZ⟩ moments, or replace fixed `0.1·χ²` with
   variational angles trained by parameter-shift on the training folds only.
   This increases capacity but also variance — it needs nested CV to be
   defensible.
5. **Handle imbalance explicitly in-fold.** 76 % of labels are permeable;
   `class_weight="balanced"`, in-fold threshold selection and balanced accuracy
   should be reported. The PCA experiment (2 features → balanced accuracy
   0.500) shows how easily the SVM collapses to the majority class.
6. **More data / external validation.** 1955 molecules and 1016 scaffolds are
   small for a kernel method; validate on a larger BBB set (e.g. B3DB) with
   scaffold splitting before claiming generalisation.

Ideas 1–2 are backed by the results above; 3–6 are **proposals, not validated
results** and are listed so the limitations are explicit rather than hidden.

**If I had to pick one change:** combine (1) and (2) — keep the chemistry-driven,
equal-weight score but force the ten backbone atoms to be a **connected
subgraph**: take the community representatives, then join them with a minimum
Steiner tree over the heavy-atom graph. Feed the resulting 78 moments through a
hybrid kernel with Morgan/composition descriptors. The diagnosis supports this
more than adding gates: random backbones have ≈ 4.5 internal backbone bonds and
≈ 29 gates and score 0.634; canonical has 9.14 bonds / 12.24 gates and scores
0.801; chemical coverage has 7.76 bonds / 15.78 gates and scores 0.766.
**Connectivity, not gate count, tracks AUC.** A connected community backbone
should keep community coverage without the duplicated interface gates that
overload q10/q11. Chiral `Rz` gates fix a correctness problem, but with only 6
stereo-conflict rows they are unlikely to move AUC much. Any such change must be
evaluated on the same folds and beat the 0.882 Morgan baseline before it is
reported as an improvement.

### 2.6 Validation protocol: exactly how many molecules train and test

All numbers reported in this README (three arms and the 2×3 comparison) come
from **out-of-fold predictions on the same 1955 molecules**. The splitter is
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with 1016
Bemis–Murcko scaffold groups as grouping variable, so a scaffold never appears
in both train and test of the same fold. The scaler and the SVM are fitted
**inside each training fold only**.

| Item | Value |
|---|---|
| Molecules (cleaned) | **1955** (461 label 0, 1494 label 1) |
| Scaffold groups | 1016 |
| Folds | 5, scaffold-disjoint, seed 42 |
| Train per fold | **1564** (369 or 368 label 0; 1195 or 1196 label 1) |
| Test per fold | **391** (92 or 93 label 0; 299 or 298 label 1) |
| Distinct test scaffolds per fold | 199, 200, 204, 206, 207 |
| Cumulative train exposures / test predictions | 7820 / **1955** |

Every molecule is predicted exactly once (out of fold), so the confusion
matrices sum to 1955 — no molecule is scored by a model trained on itself.
Fold 1: train 1564 (369/1195), test 391 (92/299); fold 2: train 1564
(368/1196), test 391 (93/298); folds 3–5: train 1564 (369/1195), test 391
(92/299). Each fold's test set contains 199–207 distinct scaffolds.

For reference, the paper-style protocol in `random_qubit_test.py` also reports
10 × `StratifiedShuffleSplit(test_size=0.2, random_state=42)` — likewise 1564
train / 391 test per split, but **not scaffold-disjoint**. It is reported
separately and is not the basis of the headline numbers.

---

## 3. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Clean the raw data
python -m analysis_features clean

# Run all sanity checks
python -m analysis_features verify

# Produce the three‑arm experiment artefacts
python -m analysis_features three-arms

# Compare Pauling vs new importance features (2x3 confusion matrices)
python -m analysis_features embedding-comparison

# Draw the graph‑to‑circuit diagram
python -m analysis_features diagram

# Draw the real 12-qubit circuit with Qiskit
python -m analysis_features circuit-diagram

# Sensitivity analysis of descriptor weights
python -m analysis_features weight-probe

# Run the test suite
pytest
```

All commands are idempotent – running them again produces the same outputs (CSV, JSON, PNG).  The outputs live under `results/` and `data/processed/`.

---

## 4. License and citation

This work is released under the MIT license.  Please cite the original paper if you use the dataset or the code.
