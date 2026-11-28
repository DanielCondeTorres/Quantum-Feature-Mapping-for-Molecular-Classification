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
| `source_random` (3 seeds) | source order, random start, salts kept | 0.8200 ± 0.017 | — | 0.732 |
| `source_random_stripped` (3 seeds) | source order, random start, salts removed | 0.8245 ± 0.020 | — | 0.737 |
| `source_permuted` (3 seeds) | random subset of source-parent atoms | 0.7587 ± 0.009 | — | 0.672 |
| `canonical_permuted` (3 seeds) | random subset of canonical-parent atoms | 0.7534 ± 0.006 | — | 0.671 |
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

### The random-start arm is not a fair chemical control

`source_random` rotates the **source traversal**; it does not destroy it. The
ten atoms are still a contiguous window of the string as deposited, so any
provenance encoded in that sequence survives the random start. Two tests make
this explicit:

- **Removing the salts does not lower it.** `source_random_stripped` scores
  **0.8245 ± 0.020**, slightly above the salt-kept version (0.8200 ± 0.017).
  The counter-ion is not the signal; only 98 of 1955 molecules are
  multi-fragment, and the salt diagnostic shows the counter-ion reaches the
  backbone in 1–5% of cases depending on the seed.
- **Destroying the traversal removes the advantage.** `source_permuted` takes a
  random ten-atom subset of the source parent and scores **0.7587 ± 0.009**;
  `canonical_permuted` does the same from the canonical parent and scores
  **0.7534 ± 0.006**. Both are below `canonical` (0.8015). The advantage lives
  in the **sequence**, not in the molecule or in the start atom.

The seed spread is also large: 0.801–0.838 for `source_random`, and seed 1 is
0.8013, indistinguishable from canonical. The mean 0.820 is not a robust
improvement; it is the average of a window lottery that still samples the
source order.

The backbone diagnostics make the same point. Random subsets have about **29
gates** and **4.5 backbone–backbone bonds**, far more gates than canonical
(12.24) or chemical (15.78), yet the lowest AUC. More gates is not more
information; a contiguous, connected backbone is.

**Practical consequence:** do not present `source_random` as evidence that a
random start helps. It is a weaker version of the source-order leak. The
controls to report are `canonical_shuffled` (same set, random qubits) and
`canonical_permuted` / `source_permuted` (traversal destroyed).

## Comparison with `main` Branch (250-Molecule Subset vs Full Dataset)

In the original repository (`main` branch):
1. **Dataset size and selection**: Only 250 molecules (125 permeable, 125 non-permeable) were sampled randomly (`BBBP_previous_250.csv`), evaluated with a 10-split `StratifiedShuffleSplit`.
2. **SMILES representation**: Molecules were processed as raw SMILES strings. Unstripped counter-ions (such as `[Cl]` in propranolol) occupied qubit 0 (`q0`), and missing elements in `HANNAN_EN` (Br, I, P) silently defaulted to 1.0.
3. **Ordering artifact**: Raw SMILES strings encode human depiction and curation order (`source` order).

When evaluating both protocols directly:
- On the **250-molecule subset** (10-split random CV):
  - `Quantum Main (Source order, uncleaned)`: AUC **0.844 ± 0.032** | BalAcc **0.762**
  - `Classical RF (Morgan FP 256)`: AUC **0.836 ± 0.033** | BalAcc **0.750**
  - `Quantum Cleaned (Canonical order)`: AUC **0.712 ± 0.054** | BalAcc **0.640**
  *(Standardizing atom ordering drops quantum AUC by >0.13 on the same 250 molecules!)*

- On the **Full Cleaned BBBP Dataset** (1,955 molecules, 5-fold scaffold split):
  - `Classical RBF-SVM (Morgan FP)`: AUC **0.8820** | BalAcc **0.7680**
  - `Classical RF (Morgan FP)`: AUC **0.8744** | BalAcc **0.7238**
  - `Quantum Main (Source order, uncleaned)`: AUC **0.8805** | BalAcc **0.7956** *(leaks depository sequence)*
  - `Whole-Molecule Composition Baseline`: AUC **0.8133** | BalAcc **0.7397**
  - `Quantum Cleaned (Canonical order)`: AUC **0.8015** | BalAcc **0.7323**
  - `Quantum Chemical (Coverage order)`: AUC **0.7664** | BalAcc **0.6892**

**Key finding**: The apparent quantum performance on `main` was inflated by two factors: small-subset random splitting (which has high variance) and the uncleaned `source` depiction order. Once cleaned and evaluated on canonical structures, the quantum feature map achieves 0.8015 AUC, which is surpassed by classical Morgan fingerprints (0.8820) and matched by basic whole-molecule atomic composition (0.8133).

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
- **Do not use `source_random` as a random-start control.** It keeps the source
  traversal and still leaks; `source_permuted` and `canonical_permuted` are the
  controls that actually destroy the sequence.
- **Expect seed variance when the ten-atom window is random.** A 0.067 spread
  across three `source_random` seeds is larger than the canonical-vs-chemical
  gap. A single seed is not a result.

## Files

| File | Content |
|---|---|
| `atom_order_benchmark.csv` | One row per arm: AUC, CI, balanced accuracy, F1, gates, edges, heteroatoms |
| `atom_order_benchmark.json` | Configuration, fold-by-fold AUCs, composition controls and salt diagnostics |
| `confusion_matrix.png` | Out-of-fold matrices for `source`, `canonical`, `chemical` on the full dataset |
| `confusion_matrices_main_vs_cleaned.png` | Side-by-side matrices comparing 250-subset vs full dataset across quantum and classical models |
| `main_vs_cleaned_benchmark.csv` | Summary table of metrics comparing `main` protocol against cleaned pipeline |
| `composition_controls.csv` | The bag-of-elements controls above |
| `backbone_diagnostics.csv` | Per-molecule gate and connectivity counts for canonical and chemical |
| `salt_diagnostics.csv` | Counter-ion inclusion and label rates for the random-start arm |
| `salt_probe.csv`, `salt_probe.json` | Salt-kept vs salt-stripped random-start arms |
| `permutation_probe.csv`, `permutation_probe.json` | Random-subset controls and their diagnostics |

Reproduce with:

```bash
python -m analysis_features benchmark
```
