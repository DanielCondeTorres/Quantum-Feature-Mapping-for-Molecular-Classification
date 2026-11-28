# Atom-order benchmark

This directory answers a narrow question: when the 10+2 circuit gives a
different score for `canonical`, `chemical` and `source` atom orders, is the
difference caused by *which ten atoms* reach q0–q9, or by *which qubit* each
atom occupies?

## Setup

| Item | Value |
|---|---|
| Molecules | 1955 cleaned BBBP structures (1494 permeable / 461 non-permeable) |
| Groups | 1016 Bemis–Murcko scaffolds; acyclic molecules share one group |
| Folds | 5-fold `StratifiedGroupKFold`, seed 42 |
| Classifier | `StandardScaler` + `SVC(kernel="rbf", C=10)`, fitted inside each training fold |
| Readout | 78 Pauli moments (12 `⟨Z⟩` + 66 `⟨ZZ⟩`) |
| Tail | `global` (the ablation tail, not the paper's alternating sums) |
| Metric | pooled out-of-fold AUC, 1000-sample bootstrap 95% CI |

Every arm sees the same molecules, labels, folds and classifier. Only the atom
ordering changes. Nothing is tuned on the test fold.

## Results

| Arm | What changes | AUC | 95% CI | Balanced acc. |
|---|---|---:|---:|---:|
| `source` | source string, counter-ions kept | **0.8805** | 0.862–0.899 | 0.796 |
| `canonical` | first ten canonical atoms | **0.8015** | 0.775–0.826 | 0.732 |
| `chemical` | greedy Morgan/role coverage | **0.7664** | 0.742–0.794 | 0.689 |
| `chemical_canonical_order` | chemical set, atom-index order | 0.7604 | 0.734–0.787 | 0.690 |
| `source_random` (3 seeds) | source order, random start | 0.8200 ± 0.017 | — | 0.732 |
| `random` (3 seeds) | canonical order, random start | 0.7992 ± 0.037 | — | 0.719 |
| `canonical_shuffled` (3 seeds) | canonical set, random qubit order | 0.7989 ± 0.001 | — | 0.728 |

The three seed means are reported with their standard deviation; the individual
seed values are in `atom_order_benchmark.csv`.

## The three controls that separate set from order

1. **`canonical_shuffled`** keeps the same ten canonical atoms and permutes
   which qubit each one occupies. It scores **0.7989**, statistically identical
   to `canonical` (0.8015). The qubit assignment does not matter.
2. **`chemical_canonical_order`** keeps the chemical-coverage set and only
   changes its order. It scores **0.7604**, statistically identical to
   `chemical` (0.7664). Again, the order does not matter.
3. **`random`** rotates the canonical heavy-atom order to a random starting
   atom. Because the molecule is read cyclically and only the first ten atoms
   are kept, this changes *which* ten-atom window is selected, not just q0. The
   score varies from 0.756 to 0.823 depending on the seed. The window matters;
   the identity of q0 alone does not.

**Conclusion: the canonical-vs-chemical gap is a property of the selected atom
set, not of the qubit numbering.**

## Why the chemical set is less predictive

The chemical selector maximises coverage of Morgan environments and chemical
roles. That objective is task-agnostic, and on BBBP it produces a different kind
of set:

| Backbone diagnostic | `canonical` | `chemical` |
|---|---:|---:|
| Heavy atoms selected | 9.87 | 9.87 |
| Heteroatoms in backbone | 2.53 | **3.27** |
| Backbone–backbone bonds | **9.14** | 7.76 |
| Rxx gates (all, incl. interface) | 12.24 | **15.78** |

The chemical set is more heteroatom-rich and more scattered. It has fewer bonds
*inside* the backbone, and more bonds that cross the backbone–tail boundary;
each interface bond is applied twice, to q10 and to q11. So the extra gates are
not extra independent couplings among the ten atom qubits: they overload the
two summary qubits instead.

A bag-of-elements control makes the same point without the circuit:

| Control (same folds, same SVM) | AUC |
|---|---:|
| `size_only` (heavy-atom count) | 0.5877 |
| `canonical_backbone_composition` (element counts of the ten atoms + size) | 0.7417 |
| `chemical_backbone_composition` (element counts of the ten atoms + size) | 0.7195 |
| `whole_molecule_composition` (element counts + size + bonds + charge) | **0.8133** |

The chemical set starts 0.022 AUC below the canonical set *before any circuit is
applied*. The circuit adds about 0.06 over its own set's composition for canonical
(0.8015 vs 0.7417) and about 0.05 for chemical (0.7664 vs 0.7195), so the map is
doing something, but it cannot recover the gap created by the selector. The chemical selector is
optimising chemical diversity, not the hydrophobic/lipophilic context that BBB
permeability depends on.

## The source-order arm is still a provenance leak

`source` is 0.079 AUC above `canonical`. Rotating the source string to a random
start lowers it from 0.8805 to 0.8200 ± 0.017, but not to the canonical level.
So part of the leak is the starting atom, and part is the source traversal
itself. Neither is chemistry. The modelling table therefore has no `source_num`
column and is sorted by structure hash.

## What this means for the paper

- **Do not report source order as a chemical result.** It is the strongest arm
  and the least meaningful.
- **Use a graph-derived selector for the main quantum arm** (or canonical order
  as a reproducible control), and report `canonical_shuffled` to show that the
  result is not an artefact of a lucky qubit numbering.
- **Report the composition controls.** A whole-molecule bag of elements reaches
  0.8133, statistically indistinguishable from the canonical circuit (0.8015,
  overlapping CIs). Any claim that the 10+2 map adds value over composition
  needs a stronger baseline and a different dataset.
- **Expect seed variance when the ten-atom window is random.** A 0.067 spread
  across three seeds is larger than the canonical-vs-chemical gap.

## Caveats

- `C=10` is fixed, not nested. On the `dct_rdkit` branch nested `C` changed AUC
  by less than 0.01, but this run does not repeat that search.
- The headline metric is pooled out-of-fold AUC. Fold-by-fold values are in
  `atom_order_benchmark.json`; their mean can differ from the pooled value.
- Scaffold groups are coarse (1016 groups for 1955 molecules; acyclic molecules
  share one group). A different grouping would move all arms together.
- The circuit is a feature map; the classifier is a classical RBF SVM.

## Files

| File | Content |
|---|---|
| `atom_order_benchmark.csv` | One row per arm: AUC, CI, balanced accuracy, F1, gates, edges, heteroatoms |
| `atom_order_benchmark.json` | Configuration, fold-by-fold AUCs and composition controls |
| `confusion_matrix.png` | Out-of-fold matrices for `source`, `canonical`, `chemical` |
| `composition_controls.csv` | The bag-of-elements controls above |
| `backbone_diagnostics.csv` | Per-molecule gate and connectivity counts |

Reproduce with:

```bash
python -m analysis_features benchmark
```
