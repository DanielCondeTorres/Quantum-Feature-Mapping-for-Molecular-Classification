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

## Random source q0 vs per-atom RDKit property selectors

The last experiment changes **how q0 is chosen**, not the circuit. Two ideas are
compared on the same 1955 molecules, 1016 scaffold groups, five folds and the
same RBF-SVM on 78 Pauli moments:

1. **`source_random`**: keep the deposited SMILES (including counter-ions), but
   rotate its heavy-atom traversal to a **random starting atom** instead of
   always using the first atom as the `main` branch does.
2. **`property_*`**: ignore the string order completely and rank heavy atoms by
   a fixed RDKit per-atom value, then put the ten highest on q0–q9. The values
   are `TPSA` contribution (`_CalcTPSAContribs`, Å²), Pauling electronegativity,
   heteroatom flag, absolute formal charge, pharmacophore role weight, and a
   fixed 0.4/0.3/0.3 mixture of TPSA/en/role. None is fitted to BBBP labels.

| Arm | Cómo se elige q0–q9 | AUC | Bal. acc. |
|---|---|---:|---:|
| `source` | primer átomo del SMILES depositado, sales incluidas | **0.8805** | 0.796 |
| `source_random` (3 seeds) | inicio aleatorio en el SMILES depositado | 0.8200 ± 0.017 | 0.732 |
| `canonical` | primeros 10 átomos del SMILES canónico | **0.8015** | 0.732 |
| `property_charge` | mayor carga formal absoluta | 0.8000 | 0.725 |
| `property_role` | mayor peso de rol farmacofórico | 0.7908 | 0.738 |
| `property_mixed` | mezcla fija TPSA/EN/rol (0.4/0.3/0.3) | 0.7891 | 0.728 |
| `property_pauling` | mayor electronegatividad de Pauling | 0.7868 | 0.736 |
| `property_tpsa` | mayor contribución TPSA por átomo (RDKit) | 0.7792 | 0.730 |
| `property_hetero` | prioridad de heteroátomos | 0.7714 | 0.712 |
| `chemical` | cobertura de entornos Morgan y roles | 0.7664 | 0.689 |

`source_random` es la media de tres semillas (0.8281, 0.8002, 0.8316); la
semilla 1, que aparece en las figuras, da 0.8002, idéntica al canónico.

`pca_property_selectors.png` contiene seis paneles de PCA (2x3) y
`confusion_matrix_property_selectors.png` las seis matrices de confusión
correspondientes, para `source`, `source_random` (semilla 1), `canonical`,
`property_tpsa`, `property_role` y `property_mixed`. El PCA se ajusta sobre los
78 momentos estandarizados (StandardScaler) y los puntos se colorean por `p_np`.

### Interpretación

- **Ningún selector por propiedad supera al canónico** (0.8015). El mejor es
  `property_charge` (0.8000), que en BBBP es casi indistinguible del canónico
  porque casi todos los átomos tienen carga 0 y el desempate por índice devuelve
  otra vez los primeros diez. No es una ganancia química.
- **Priorizar polaridad es peor**: TPSA 0.7792, heteroátomos 0.7714 y cobertura
  Morgan 0.7664 quedan por debajo. Como ya mostró el diagnóstico de gates, estos
  selectores eligen átomos polares dispersos, reducen los enlaces
  backbone–backbone (7.76 frente a 9.14 del canónico) y absorben la señal en
  gates de interfaz duplicados en q10/q11.
- **`source_random` sí es numéricamente mejor** (0.820 de media) que todos los
  selectores químicos, pero no es una prueba de que el inicio aleatorio ayude:
  al destruir la secuencia (`source_permuted`) cae a 0.7587, y al quitar sales
  no baja (0.8245). Lo que queda es la secuencia de procedencia del archivo, no
  la química.
- **El PCA no muestra separación lineal clara.** PC1+PC2 explica entre 39.2% y
  50.8% de la varianza según el brazo; las nubes de permeables y no permeables
  se solapan en todos los paneles. `property_role` y `property_mixed` explican
  algo más de varianza (50.7%/50.8%) pero no mejor AUC (0.791/0.789). La
  separación la hace la SVM RBF, no las dos primeras componentes.
- **El desequilibrio de clases importa.** `property_role` tiene mayor balanced
  accuracy (0.738) que `canonical` (0.732) pese a menor AUC (0.791 vs 0.802);
  en todos los paneles la matriz tiene muchos más falsos positivos que
  falsos negativos porque el 76% de las moléculas son permeables.

**Conclusión para el equipo:** elegir q0 por propiedades químicas de RDKit no
mejora el mapa 10+2; la ventaja aparente del `source` con inicio aleatorio es
fuga de procedencia, y el mejor modelo real sigue siendo el clásico de Morgan
(AUC 0.882).

## The requested four-feature comparison

`python -m analysis_features four-features` runs **only** these four arms on the
cleaned 1955-row table. Everything else is identical: 1016 scaffold groups,
five folds, `global` tail, StandardScaler fitted in-fold and RBF SVM `C=10`.
For `pca2`, PC1/PC2 come from the 78 moments and the scaler **and** PCA are
fitted inside each training fold, so no test information leaks.

| Arm | Circuit feature | Classifier features | AUC (95% CI) | Bal. acc. |
|---|---|---:|---:|---:|
| `source` | SMILES order + Pauling angles | 78 | **0.8805** (0.862–0.899) | 0.796 |
| `source_random` | same string, random q0 (seed 0) | 78 | 0.8281 (0.805–0.851) | 0.740 |
| `tpsa` | canonical order + per-atom TPSA angles | 78 | 0.8073 (0.781–0.830) | 0.720 |
| `pca2` | canonical circuit → PCA of 78 moments | **2** | 0.5933 (0.565–0.622) | 0.500 |

The four confusion matrices are in `confusion_matrices_four_features.png`
(TN/FP/FN/TP in reading order):

- `source`: 309 / 152 / 118 / 1376
- `source_random`: 259 / 202 / 121 / 1373
- `tpsa`: 241 / 220 / 123 / 1371
- `pca2`: 0 / 461 / 0 / 1494

**Which is best?** By AUC, `source` (0.8805) — but that is the deposited-string
leak, not chemistry. Among the structure-based arms, **TPSA as the per-atom
circuit angle is the best** (0.8073), marginally above the canonical Pauling
encoding (0.8015, CIs overlap); it is an ablation, not the published paper.
**PC1/PC2 as features are a failure**: unsupervised PCA keeps the largest
variance, not the label-bearing variance, so the SVM collapses to predicting
"permeable" for every molecule (balanced accuracy 0.500). Do not use two
unsupervised components as the feature map for this task.

## Why PC1+PC2 fails (the detailed explanation)

`python -m analysis_features pca-diagnosis` separates four effects. The figure
`pca_diagnosis.png` shows all of them.

1. **PCA rotation is not the problem.** Keeping **all 78** PCs gives AUC
   **0.8015**, exactly the same as the raw 78 standardized moments. PCA is an
   orthogonal rotation and the isotropic RBF kernel is invariant to it. A
   linear combination of features is fine.
2. **Truncation is the problem.** The signal is distributed across dozens of
   moments, not concentrated in the highest-variance directions:

   | PCs kept | 1 | 2 | 3 | 5 | 10 | 20 | 40 | 78 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | AUC (unweighted SVM) | 0.573 | 0.593 | 0.623 | 0.652 | 0.696 | 0.783 | 0.802 | 0.802 |
   | AUC (class_weight=balanced) | 0.632 | 0.654 | 0.675 | 0.689 | 0.729 | 0.785 | 0.800 | 0.801 |

   The first two PCs explain 43% of the variance but almost none of the label
   signal. PC1 correlates with the label only **+0.17**, and more strongly with
   **heavy-atom count (-0.30)**: it is mostly a size/composition axis. PC2 has
   label correlation **+0.02**. Unsupervised PCA maximises variance, not class
   separation.
3. **No single moment carries the signal.** The best individual moment reaches
   AUC 0.784, below the full 78-dimensional model (0.802). The discriminative
   information is in the *joint pattern* across atoms and bonds; any 2-dimensional
   projection throws most of it away.
4. **The all-positive prediction is a class-imbalance artefact on top of that.**
   The dataset is 76.4% permeable. With only two weak features, the unweighted
   SVM puts every decision score above zero (std 0.092), so it predicts
   "permeable" for all 1955 molecules: AUC 0.593, balanced accuracy 0.500. With
   `class_weight="balanced"` the same two PCs give AUC 0.654 and balanced
   accuracy 0.609, and 49% positive predictions. The kernel width is *not* the
   cause: default gamma (0.0298) versus the 78-feature gamma (0.0128) changes
   AUC only from 0.593 to 0.597.

**And why the order/chemical comparison looks the other way round:** `source`
(0.8805) is not chemistry, it is the deposited string's curation order leaking
into q0; that is the signal the cleaning removes. `tpsa` as an **angle**
(0.8073) keeps the full canonical graph and only changes the rotation values,
so it slightly improves on Pauling (0.8015, overlapping CIs). `tpsa` or
heteroatoms as an atom **selector** (0.779/0.771) instead choose a scattered
set and lose backbone connectivity. So "the order is better" only holds for the
leaky source order; among honest structure-based arms, keeping the topology
matters more than which per-atom property is used.

## Paper qubit order vs random backbone (full dataset)

The paper states: *"Atoms are indexed in RDKit SMILES-parse order. Backbone
atoms a₀–a₉ map one-to-one onto q₀–q₉"*. `python -m analysis_features
random-qubit-test` reproduces that rule on all **1955 cleaned molecules** and
replaces only that rule with a random ten-atom backbone, over 20 seeds. The
published tail, Pauling angles, RBF SVM and folds are otherwise identical.

Two protocols are reported: five scaffold-disjoint folds and the paper's
ten stratified shuffle splits. Plain accuracy has a trivial baseline of
**0.7642** ("always permeable") on this imbalanced table.

| Protocol | Regla de q0–q9 | Accuracy | Bal. acc. | AUC |
|---|---|---:|---:|---:|
| 5 folds por scaffold | paper: primeros 10 átomos en orden SMILES | **0.8501** | 0.776 | 0.8559 |
| 5 folds por scaffold | aleatoria, media de 20 seeds | 0.7587 ± 0.0180 | 0.6116 | 0.6886 |
| 10 splits estratificados | paper | **0.8558 ± 0.0171** | 0.7827 | 0.8727 |
| 10 splits estratificados | aleatoria, media de 20 seeds | 0.7747 ± 0.0182 | — | — |

La accuracy aleatoria va de 0.7361 a 0.8036 en los 20 seeds; el mejor
(seed 13, accuracy 0.8036, AUC 0.7969) sigue por debajo del paper. Con la
regla del paper no hay ninguna semilla que la iguale.

**El resultado incómodo:** la media aleatoria (0.7587) está incluso ligeramente
**por debajo del baseline trivial** de predecir siempre “permeable” (0.7642).
Es decir: elegir los 10 qubits al azar produce un clasificador peor que no
mirar la molécula. La selección del paper, en cambio, da 0.850 de accuracy.
Esto confirma que el orden de átomos del SMILES contiene una señal muy fuerte
para esta tarea — pero, como mostramos arriba, es señal de **procedencia del
archivo**, no de química: al canonizar el mismo conjunto cae y al destruir la
secuencia (`source_permuted`, 0.7587) se comporta exactamente como una
selección aleatoria.

`random_qubit_accuracy.png` muestra el histograma de accuracies de los 20
seeds frente a la barra del paper y la línea del baseline trivial, la relación
accuracy/balanced accuracy y una matriz de confusión de ejemplo (seed 0:
TN 131, FP 330, FN 177, TP 1317).

## BBBP atom importance and graph communities

`python -m analysis_features importance-test` replaces the random backbone
with a chemically motivated one. A fixed score combines RDKit per-atom TPSA
contribution, Crippen logP/MR contributions, HBD/HBA SMARTS roles, Gasteiger and
formal charges, element identity and aromaticity. Continuous descriptors use
their absolute deviation from the molecular mean and the weights (0.20 TPSA,
0.20 logP, 0.10 MR, 0.15 HBD, 0.10 HBA, 0.10 charge, 0.05 each for formal,
hetero and aromatic) are fixed a priori, never fitted to labels. The
`community_importance` arm first splits the heavy-atom graph into communities
by modularity optimisation and selects the best atom of each community.

| Regla de q0–q9 | Accuracy | Bal. acc. | AUC | Accuracy (10 splits) |
|---|---:|---:|---:|---:|
| Paper: orden SMILES | **0.8501** | 0.776 | 0.8559 | — |
| Aleatorio (media 20 seeds) | 0.7587 | 0.612 | 0.6886 | — |
| `importance` (RDKit) | 0.8061 | 0.704 | 0.7777 | 0.8261 |
| `community_importance` (networkx) | 0.8036 | 0.704 | 0.7679 | 0.8192 |

Matrices de confusión: `importance` TN 235 / FP 226 / FN 153 / TP 1341;
`community_importance` TN 237 / FP 224 / FN 160 / TP 1334.

**Conclusión:** la selección por importancia química sí funciona — mejora la
accuracy de 0.759 a 0.806 y el AUC de 0.689 a 0.778 respecto a elegir átomos al
azar, y supera el baseline trivial de 0.764. Pero sigue por debajo del orden
SMILES del paper (0.850), que como ya vimos contiene fuga de procedencia. Las
comunidades no aportan: 0.804 frente a 0.806 del ranking directo, probablemente
porque obligan a representar regiones que no son las más informativas. La
figura `importance_test.png` compara los cuatro brazos.

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
| `property_selector_benchmark.csv`, `.json` | AUC and PCA variance for random-source q0 vs per-atom property selectors |
| `pca_property_selectors.png` | PCA (2x3) of the 78 moments coloured by label for six selectors |
| `confusion_matrix_property_selectors.png` | Out-of-fold confusion matrices (2x3) for the same six selectors |
| `four_features_benchmark.csv`, `.json` | The four requested arms: source, random q0, PC1/PC2 and TPSA |
| `confusion_matrices_four_features.png` | The four out-of-fold confusion matrices (2x2) |
| `pca_diagnosis.csv`, `.json`, `.png` | Rotation vs truncation, PC correlations, imbalance and score distributions |
| `random_qubit_test.csv`, `.json`, `.png` | Paper SMILES order vs 20 random backbones, accuracy distributions |
| `importance_test.csv`, `.json`, `.png` | RDKit BBBP importance and graph communities vs paper order |

Reproduce with:

```bash
python -m analysis_features benchmark
python -m analysis_features compare
python -m analysis_features property-probe
python -m analysis_features four-features
python -m analysis_features pca-diagnosis
python -m analysis_features random-qubit-test
python -m analysis_features importance-test
```
