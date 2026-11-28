# Full BBBP dataset for the 10+2 quantum feature map

Branch **`analysis_features`**, based on **`main`**.

This branch turns the raw DeepChem BBBP table into a modelling-ready dataset for
the paper, and exposes the 10+2 embedding in a form that can be inspected line
by line. It is deliberately self-contained: the cleaning code and the circuit
simulator depend only on RDKit, NumPy and pandas — not on the experiment
package used in the `dct_rdkit` branch.

The paper notebooks on `main` are untouched.

---

## TL;DR

| Quantity | Result |
|---|---:|
| Raw rows in `data/raw/BBBP.csv` | **2050** |
| Invalid SMILES (RDKit) | **11** |
| Valid rows after parsing | 2039 |
| Multi-fragment rows (salts / solvents / mixtures) | **105** |
| Structures with contradictory labels (excluded) | **11** structures / **22** rows |
| Duplicate rows collapsed | **62** |
| **Final modelling table** | **1955** (1494 permeable / 461 non-permeable) |
| Elements present | B, Br, C, Cl, F, I, N, O, P, S |
| Elements missing from the original table | **Br, I, P** (added explicitly) |
| Previous 250-molecule benchmark reproduced | **250 / 250**, 0 label mismatches |
| Molecules with more than ten heavy atoms | **1851 / 1955** |
| Charged structures kept (not neutralised) | **34** |
| Stereoisomer rows with opposite labels | **6** |
| Rows in tautomer groups with opposite labels | **8** (6 are the stereo pairs, 2 are true tautomers) |

The number that changed relative to the earlier `dct_rdkit` deliverable is the
input count: `data/bbbp_mol/bbbp.csv` on that branch had **already dropped the
11 invalid rows** (2039 rows). Starting from the untouched 2050-row file makes
the failure count visible, which is what the methods section should report.

---

## Quick start

```bash
python -m pip install -r requirements.txt

# Rebuild every deliverable from the raw table
python -m analysis_features clean

# Inspect one molecule: which atom sits on each qubit?
python -m analysis_features embed --smiles "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"

# Compact summary of the cleaned table
python -m analysis_features summary

# Check every requirement against the committed outputs
python -m analysis_features verify

# Reproduce the atom-order benchmark and the confusion matrices
python -m analysis_features benchmark

# Main-vs-cleaned comparison on the 250 subset and the full dataset
python -m analysis_features compare

# Random-source q0 vs per-atom RDKit property selectors + PCA
python -m analysis_features property-probe

# Tests
python -m pytest
```

Everything is deterministic. The RDKit version used for the committed outputs
is stored in `data/processed/BBBP_clean_settings.json` (2024.03.5 at the time of
writing).

---

## What is in the branch

```text
analysis_features/
  config.py        # circuit constants and default paths
  elements.py      # original Hannan table, explicit extensions, coverage audit
  standardize.py   # parse, salt stripping, canonical SMILES, tautomer key
  cleaning.py      # the eight-stage pipeline (pure functions, counted stages)
  atom_order.py    # canonical / source / chemical / random backbone selectors
  embedding.py     # exact 10+2 state vector + Pauli readout, no Qiskit
  benchmark.py     # atom-order comparison on scaffold folds + confusion matrix
  compare.py       # main (250 subset, source) vs cleaned full dataset
  property_probe.py # random q0 vs per-atom RDKit property selectors + PCA
  verify.py        # independent check of every cleaning requirement
  io.py            # read raw tables, write the six deliverable files
  cli.py           # clean / embed / summary / benchmark / compare / property-probe / verify
tests/             # 47 unit, integration and verification tests
data/
  raw/BBBP.csv                  # untouched DeepChem download (2050 rows)
  raw/BBBP_previous_250.csv     # previous balanced benchmark (125 + 125)
  processed/
    BBBP_clean_full.csv         # modelling table (1955 x 10)
    BBBP_clean_audit.csv        # every parsed row + exclusion reason (2039)
    BBBP_invalid_smiles.csv     # the 11 rejected source strings
    BBBP_fragment_audit.csv     # every disconnected component
    BBBP_clean_log.txt          # per-stage counts
    BBBP_clean_settings.json    # versions + decisions in machine-readable form
docs/METHODS.md                 # copy-paste methods paragraph
results/
  atom_order_benchmark.csv      # AUC per ordering rule, same folds
  atom_order_benchmark.json     # fold-by-fold scores and configuration
  confusion_matrix.png          # out-of-fold matrices (source / canonical / chemical)
  composition_controls.csv      # bag-of-elements controls
  backbone_diagnostics.csv      # gates and connectivity per molecule
  salt_probe.csv / .json        # random q0 with and without salts
  permutation_probe.csv / .json # traversal destroyed controls
  confusion_matrices_main_vs_cleaned.png
  main_vs_cleaned_benchmark.csv
  property_selector_benchmark.csv / .json
  pca_property_selectors.png
  confusion_matrix_property_selectors.png
```

---

## The cleaning pipeline, stage by stage

The pipeline is `analysis_features.cleaning.clean_dataset`. Each stage reports
its own count to `BBBP_clean_log.txt`.

| Stage | Rule | BBBP result |
|---|---|---:|
| 1. Parse | `Chem.MolFromSmiles`; reject if RDKit returns `None` | 11 rejected |
| 2. Strip salts | Keep the largest **carbon-containing** fragment; ties by heavy atoms, then heteroatoms, then canonical SMILES | 105 rows |
| 3. Canonicalise | `MolToSmiles(canonical=True, isomericSmiles=True)` | 2039 |
| 4. Elements | Collect every element; compare with the original table and with the explicit extension; raise on any unknown | Br, I, P added |
| 5a. Label conflicts | Group by canonical structure; exclude every structure carrying both labels | 11 structures / 22 rows |
| 5b. Duplicates | Collapse identical canonical structures, keeping the first occurrence | 62 removed |
| 6. Stereo flag | Group by stereo-free canonical structure; flag groups with opposite labels | 6 rows |
| 7. Tautomer flag | Group by RDKit tautomer canonical form; flag groups with opposite labels | 8 rows |
| 8. Previous benchmark | Canonicalise the previous 250 SMILES with the same parent rule; flag membership | 250 / 250 |

### Why the order matters: propranolol

The first raw entry is

```text
[Cl].CC(C)NCC(O)COc1cccc2ccccc12    label 1
```

and the free base also appears later:

```text
CC(C)NCC(O)COc1cccc2ccccc12          label 0
```

If salts are stripped **before** deduplication, both rows collapse to
`CC(C)NCC(O)COc1cccc2ccccc12` and the contradiction becomes visible. The
structure is then excluded. This is the concrete answer to the qubit-0 worry:

- without stripping, the published baseline (`order="source"`) puts **Cl on
  qubit 0**;
- with canonical re-numbering alone, Cl moves but remains in the molecule and
  distorts the tail;
- with stripping, qubit 0 is a carbon of the parent.

```bash
python -m analysis_features embed \
  --smiles "[Cl].CC(C)NCC(O)COc1cccc2ccccc12" \
  --order source --keep-salts
# q0 = Cl

python -m analysis_features embed \
  --smiles "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
# q0 = C, parent = CC(C)NCC(O)COc1cccc2ccccc12
```

---

## Decisions, and why

### Salts and solvents — removed

We keep the largest **carbon-containing** fragment. Carbon comes first so a
counter-ion is never chosen as the parent. Ties are broken by heavy-atom count,
then heteroatom count, then the canonical string. The lexical tie-break is not
chemistry, so every tie is flagged:

- `num 105`: caffeine · caffeine · ethylenediamine (a 1:2:1 adduct);
- `num 1839`: steroid · steroid · water.

Both keep the largest organic fragment. They are mixtures rather than simple
salts and are flagged in `BBBP_fragment_audit.csv` (`manual_review = True`) and
in the audit table (`ambiguous_fragments = True`).

### Charges — kept, never neutralised

A protonated amine and its free base are different species at physiological pH.
Neutralising would change the property being predicted, so `formal_charge` is
recorded and the structure is left as deposited. 34 structures in the final
table carry a non-zero formal charge.

### Tautomers — not merged

RDKit's tautomer enumerator is a heuristic and can move protons in ways that
change the permeating species. We use it **only** to group structures for
conflict detection. No tautomer is rewritten or merged.

The two genuine tautomer conflicts are the xanthine forms:

```text
Cn1c(=O)c2[nH]cnc2n(C)c1=O    label 0
Cn1c(=O)c2nc[nH]c2n(C)c1=O    label 1
```

They remain in the table with `tautomer_label_conflict = True`. The other six
flagged rows are the stereo pairs below: same constitution, different labels.

### Stereochemistry — preserved in the file, invisible to the circuit

RDKit keeps the `@/@@` tags: 629 of the 1955 cleaned molecules carry stereo.
They are **not** removed or averaged. But the feature map only reads Pauling
electronegativity per atom and bond order per edge; both are invariant under a
chiral inversion. A dedicated audit on this same 1955-molecule table found:

| Measurement | Count |
|---|---:|
| Stereoisomer forms in skeletons with more than one form | 43 |
| Forms with identical features under SMILES order + Pauling | **42 / 43** |
| Forms with identical features under chemical coverage + TPSA | **41 / 43** |
| Forms in skeletons carrying opposite labels | **6** |

For those six rows, two molecules produce the same 78-number vector but have
different `p_np`. No classifier on top of the map can separate them; this is
irreducible representation noise, not a model failure. They are flagged as
`stereo_label_conflict` so the main experiment can exclude them or report them
as a limitation.

### Invalid SMILES — rejected, not repaired

The 11 failures are listed in `BBBP_invalid_smiles.csv` with their original
`num`, name, SMILES and label. Most share a systematic guanidine motif
(`[NH]=C(\N)N`, `[NH2]` with four bonds) that RDKit rejects on valence grounds.
They could be repaired by hand, but a silent repair would be a chemical edit
masquerading as parsing. They are therefore excluded and counted.

### "Standardize" — what we do and do not do

We parse with RDKit (sanitisation on), strip disconnected fragments and write
canonical isomeric SMILES. We do **not** call `rdMolStandardize.Cleanup` by
default, because it rewrites charge-separated groups (for example sulfoxide
`S=O` → `[S+][O-]`) and would change 15 structures in this dataset. The flag is
available as `--cleanup` for a sensitivity check, but it is off by default.

---

## Column dictionary

### `BBBP_clean_full.csv` — modelling table

| Column | Meaning |
|---|---|
| `mol_id` | `bbbp_` + first 12 hex of SHA-1(canonical SMILES). Structure-derived, carries no source-order information. |
| `canonical_smiles` | Canonical isomeric SMILES of the parent. |
| `label` | BBBP `p_np` (1 permeable, 0 non-permeable). |
| `heavy_atom_count` | Non-hydrogen atoms in the parent. |
| `bond_count` | Bonds between two heavy atoms. |
| `atoms_beyond_10` | `max(0, heavy_atom_count - 10)`; atoms compressed into the tail. |
| `formal_charge` | Net formal charge of the parent. |
| `in_previous_250` | Structure belongs to the previous balanced benchmark. |
| `stereo_label_conflict` | Skeleton has stereoisomers with opposite labels. |
| `tautomer_label_conflict` | Tautomer group has opposite labels. |

Rows are sorted by `mol_id`, not by source order, so row position cannot encode
curation provenance.

### `BBBP_clean_audit.csv` — provenance table

Same columns plus `source_num`, `source_name`, `source_smiles`,
`salt_stripped`, `ambiguous_fragments` and `excluded_reason`
(`included`, `label_conflict`, `duplicate`). It keeps all 2039 parsed rows,
including excluded ones, so every removal is inspectable.

> **Do not use `source_num` or `source_smiles` as model features.** The source
> file is grouped by provenance and the row index alone reaches AUC 0.940 under
> scaffold cross-validation. The audit table exists for tracing, not modelling.

---

## The 10+2 embedding, precisely

### What the circuit sees

The circuit does not see a drawing, a conformer or a SMILES string. It sees
three kinds of local numbers computed by RDKit:

1. **One angle per selected atom.** The published encoding uses Pauling
   electronegativity `χ`:

   ```text
   θ_i = α · χ_i²,        α = 0.1
   ```

   `Ry(θ_i)` is applied to qubit `i` for the first ten selected heavy atoms.
2. **One angle per bond.** `Rxx(φ_ij)` with

   ```text
   φ_ij = bond_order_ij / 2
   ```

   applied between the qubits of the two bonded atoms.
3. **Two tail angles.** Every heavy atom after the tenth is compressed into
   qubits 10 and 11. In the published rule the tail is split alternately and
   each half contributes an electronegativity sum:

   ```text
   θ_10 = (α/2) · Σ χ_i   (tail positions 0, 2, 4, ...)
   θ_11 = (α/2) · Σ χ_i   (tail positions 1, 3, 5, ...)
   ```

   Interface bonds (one atom in the backbone, one in the tail) are coupled to
   **both** q10 and q11 in the published circuit.

The state is built from `|0⟩^⊗12`:

```text
|ψ⟩ = (∏_{bonds} Rxx(φ)) · (∏_{atoms} Ry(θ)) |0⟩^⊗12
```

and measured in the computational basis:

```text
p(s) = |⟨s|ψ⟩|²,    s ∈ {0,1}^12
```

There are 4096 probabilities. The paper feeds all 4096 to an RBF-kernel SVM.
This branch defaults to the compact **78 Pauli moments** (`12` single-qubit
`⟨Z_i⟩` plus `66` pair `⟨Z_i Z_j⟩`), which are estimable from the same
computational-basis shots and are easier to inspect.

### Propranolol after salt stripping

`CC(C)NCC(O)COc1cccc2ccccc12` has 13 heavy atoms: 10 on the backbone, 3 in the
tail. The printed table from `python -m analysis_features embed ...` is:

| Qubit | Atom index | Element | Angle (rad) |
|---:|---:|---|---:|
| q0 | 0 | C | 0.65025 |
| q1 | 1 | C | 0.65025 |
| q2 | 2 | C | 0.65025 |
| q3 | 3 | N | 0.92416 |
| q4 | 4 | C | 0.65025 |
| q5 | 5 | C | 0.65025 |
| q6 | 6 | O | 1.18336 |
| q7 | 7 | C | 0.65025 |
| q8 | 8 | O | 1.18336 |
| q9 | 9 | C | 0.65025 |
| q10 | tail A | — | 0.63750 |
| q11 | tail B | — | 0.51000 |

The chlorine is gone. Qubit 0 is a methyl carbon of the parent, with
`θ = 0.1 × 2.55² = 0.65025`.

### Does the circuit classify?

No. The circuit is a **feature map**. The classifier is always classical:

```text
SMILES → RDKit numbers → 12-qubit rotations → 4096 amplitudes
       → 4096 probabilities or 78 moments → classical SVM → label
```

The same `StandardScaler + SVC(RBF)` can consume RDKit descriptors or circuit
moments; it does not know which is which. Any claim of a "quantum classifier"
in this project would be inaccurate.

### Did we use TPSA?

**Not in the published encoding.** The paper's atom angle is `0.1 · χ²`
(Pauling), and that is the default here. TPSA appears in the `dct_rdkit`
ablation branch in two different roles, which are easy to conflate:

| Variant | Atom angle | Atom selection | Readout |
|---|---|---|---|
| **Paper / this branch default** | `0.1 · χ²` | first ten atoms of the string | 4096 `p(s)` |
| `dct_rdkit` main arm | `0.1 · χ²` | `CHEMICAL_COVERAGE` (graph-derived) | 78 `Z/ZZ` |
| "chem coverage + TPSA" | `π · clip(TPSA_i / 35 Å², 0, 1)` | `CHEMICAL_COVERAGE` | 78 `Z/ZZ` |
| Other ablations | Gasteiger charge, Crippen logP, composite BBB, angle mixtures, multi-physchem | several | `X/Y/Z`, fidelity kernel, re-uploading |

So: **the paper uses Pauling electronegativity; TPSA was an optional atom
property and a selector feature in later ablations, not part of the original
circuit.** `embed(..., property="tpsa")` is provided to reproduce that variant,
but it is not the default.

### Reproducing the paper exactly

Two switches matter:

```bash
# published tail rule + source atom order
python -m analysis_features embed --smiles "..." --order source --tail paper

# improved summary tail
python -m analysis_features embed --smiles "..." --order canonical --tail global
```

`--order source` keeps the atom order of the string as written, which is what
the original notebook did (`atoms[:10]`). `--order canonical` re-parses RDKit's
canonical SMILES first and is reproducible across input spellings. `--tail
paper` uses the alternating electronegativity sums above; `--tail global` uses
the chemical/structural summaries from the newer ablations.

---

## Advice on atom ordering

Canonical SMILES makes atom numbering **reproducible**, not **chemically
meaningful**. Atom 0 is still whatever RDKit's canonical ranking puts first.
The published baseline is worse: it uses the order of the source string, which
encodes who deposited the record.

Measured on the same 1955 molecules with scaffold-disjoint outer folds
(`dct_rdkit`, `analysis/full_dataset/`):

| Representation | AUC (fixed C) | AUC (nested C) |
|---|---:|---:|
| `source_row` (audit only) | 0.940 | 0.951 |
| SMILES order, source string | 0.889 | 0.895 |
| Canonical traversal | 0.810 | 0.815 |
| Chemical coverage + Pauling | 0.784 | 0.795 |

The source-order arm is not reading chemistry; it is reading curation order.
The drop from 0.889 to 0.810 when canonicalising is the leak disappearing.

### Reproduced on this branch, with the order/set controls

`python -m analysis_features benchmark` reruns the comparison on the 1955-row
table with 1016 scaffold groups, five folds, fixed `C=10`, 78 Pauli moments and
the `global` tail. The numbers are pooled out-of-fold AUC (the `dct_rdkit`
table above reports the mean of fold AUCs, which is slightly higher).

| Arm | What changes | AUC | 95% CI |
|---|---|---:|---:|
| `source` | source string, counter-ions kept | **0.8805** | 0.862–0.899 |
| `canonical` | first ten canonical atoms | **0.8015** | 0.775–0.826 |
| `chemical` | greedy Morgan/role coverage | **0.7664** | 0.742–0.794 |
| `chemical_canonical_order` | chemical set, atom-index order | 0.7604 | 0.734–0.787 |
| `source_random` (3 seeds) | source order, random start | 0.8200 ± 0.017 | — |
| `random` (3 seeds) | canonical order, random start | 0.7992 ± 0.037 | — |
| `canonical_shuffled` (3 seeds) | canonical set, random qubit order | 0.7989 ± 0.001 | — |

Two controls settle the canonical-vs-chemical question:

- `canonical_shuffled` keeps the canonical set and permutes the qubits: **0.7989**,
  identical to `canonical`. The qubit assignment does not matter.
- `chemical_canonical_order` keeps the chemical set and only changes the order:
  **0.7604**, identical to `chemical`. Again, the order does not matter.

So the gap is a property of the **selected atom set**, not of the numbering.
The chemical selector maximises Morgan-environment and chemical-role coverage,
which on BBBP yields a scattered, heteroatom-rich backbone: 3.27 heteroatoms
versus 2.53, 7.76 backbone–backbone bonds versus 9.14, and 15.78 gates versus
12.24 (the surplus are interface bonds duplicated onto q10 and q11). Its own
bag-of-elements is less predictive too: 0.7195 versus 0.7417 for the canonical
set, before any circuit is applied. A whole-molecule composition baseline
reaches **0.8133**, statistically indistinguishable from the canonical circuit
(0.8015, overlapping CIs). The map adds 0.05–0.06 over the composition of its
own selected atoms, but it does not beat a simple whole-molecule composition
model, and it is well below RDKit descriptors (0.870).

Rotating the source string to a random start lowers it from 0.8805 to
0.8200 ± 0.017, so part of the leak is the starting atom and part is the source
traversal. Rotating the canonical order changes the ten-atom window and gives
0.7992 ± 0.037: the window matters, the identity of q0 does not, and the seed
spread (0.756–0.823) is larger than the canonical-vs-chemical gap.

**Recommendation for the paper:** use a graph-derived selector for the main
quantum arm (or canonical order as a reproducible control), report
`canonical_shuffled` to show that the result is not a lucky numbering, report
source order only as the historical baseline with the leakage caveat, and
include the composition controls. Do not tune on the source-order score.

### Random q0 vs per-atom RDKit property selectors

A separate probe asks whether a chemically chosen q0 beats a random one. The
`property_*` arms rank heavy atoms by a fixed RDKit per-atom value — TPSA
contribution (`_CalcTPSAContribs`, Å²), Pauling electronegativity, heteroatom
flag, absolute formal charge, pharmacophore role, and a 0.4/0.3/0.3 mixture —
and put the ten highest on q0–q9. They ignore the deposited string entirely and
use no label information. The comparison is on the same 1955 molecules, folds
and SVM as above.

| Selector | Cómo elige q0–q9 | AUC |
|---|---|---:|
| `source` | primer átomo del string depositado (con sales) | 0.8805 (fuga) |
| `source_random` (3 seeds) | inicio aleatorio en ese string | 0.8200 ± 0.017 |
| `canonical` | primeros 10 del SMILES canónico | **0.8015** |
| `property_charge` | mayor carga formal absoluta | 0.8000 |
| `property_role` | mayor peso de rol farmacofórico | 0.7908 |
| `property_mixed` | mezcla fija TPSA/EN/rol | 0.7891 |
| `property_pauling` | mayor electronegatividad | 0.7868 |
| `property_tpsa` | mayor contribución TPSA atómica (RDKit) | 0.7792 |
| `property_hetero` | prioridad de heteroátomos | 0.7714 |
| `chemical` | cobertura de entornos Morgan | 0.7664 |

Ningún selector químico supera al canónico. El que empata (`property_charge`,
0.8000) lo hace porque casi todos los átomos tienen carga 0 y el desempate
devuelve los primeros diez. Los que priorizan polaridad (TPSA, heteroátomos)
quedan por debajo: eligen átomos dispersos y pierden conectividad del backbone.
`source_random` es numéricamente mejor que todos ellos, pero es fuga: al
destruir la secuencia (`source_permuted`, 0.7587) cae por debajo del canónico,
y quitar sales no lo reduce (0.8245). El PCA no separa las clases en ningún
brazo (PC1+PC2 explica 39–51% de la varianza y las nubes se solapan); la
separación la hace la SVM RBF.

`results/pca_property_selectors.png` shows the PCA of the 78 moment vectors
(coloured by `p_np`) and the matching out-of-fold confusion matrices for
`source`, `source_random` (seed 1), `canonical`, `property_tpsa`,
`property_role` and `property_mixed`.

---

## Advice on stereoisomers

Keeping `@/@@` is chemically correct, but the current encoding cannot use them.
Options, in order of effort:

1. **Exclude the six flagged rows** from the main benchmark and report them as
   a known limitation. This is the cleanest choice for the paper.
2. **Add a chiral channel** to the feature map. A minimal version assigns a
   local rotation of `+δ` or `−δ` to a stereocentre depending on tetrahedral
   parity, so enantiomers no longer produce identical states. This is new
   work, but it is the only way to make a structure-based claim about
   enantioselective permeation.
3. **Report stereo-blindness explicitly** and avoid claiming that the circuit
   captures 3D structure. It does not: there is no conformer, no distance, no
   chirality.

The audit that produced the 42/43 and 41/43 numbers is on `dct_rdkit`
(`experiments/audit_stereo_collisions.py`, `analysis/stereo_collision/`). The
flag in this branch is the same grouping rule.

---

## Chemically questionable entries

- **11 invalid source SMILES.** Systematic valence errors around guanidine and
  `[NH2]`; excluded and listed. Most are recoverable by hand, but not by a
  rule that would be safe to apply silently.
- **2 ambiguous parent selections.** The caffeine/ethylenediamine adduct and
  the steroid/water mixture. The largest organic fragment is kept, but these
  are mixtures, not salts. Flagged.
- **11 label conflicts.** Propranolol, aspirin, L-DOPA and eight others. The
  duplicate free base / salt pair is the common pattern; the label is curation
  noise, and there is no external evidence in the file to choose one.
- **One genuine tautomer conflict.** The two xanthine forms, labels 0 and 1.
  Kept and flagged.
- **34 charged structures.** Kept deliberately. If a downstream model assumes
  neutral molecules, that assumption is wrong for BBBP.
- **1851 / 1955 molecules have more than ten heavy atoms** (median 23, maximum
  132). The tail compression is therefore the normal case, not an edge case.
  Tail–tail bonds are not emitted as individual gates; this is an architectural
  bottleneck to state in the methods.

---

## Reproducing the previous 250

`data/raw/BBBP_previous_250.csv` is the old balanced benchmark (125 permeable,
125 non-permeable). After applying the same salt-stripping and canonicalisation
rule:

- **250 / 250** structures are present in the cleaned table;
- **0** label mismatches;
- `in_previous_250 = True` marks them, so the old results can be recomputed on
  exactly the same rows.

The previous 250 are a subset of the 1955, not a separate dataset. `--order
source` with their original strings reproduces the old atom order; `--order
canonical` gives the reproducible version.

---

## Tests and reproducibility

```bash
python -m pytest
# 42 passed

python -m analysis_features verify
# 34/34 checks passed
```

`verify` re-reads the committed CSVs and checks each requirement independently:
raw row count, invalid count, parsed balance, final size, label counts, no
duplicates, no conflicting labels, required columns, the `atoms_beyond_10`
formula, element coverage, audit reasons, previous-250 reproduction, the
propranolol qubit-0 regression, the recorded RDKit version, the stage log,
charge/stereo/tautomer flags and the untouched Hannan table.

The suite covers salt stripping, canonical invariance, charge and stereo
preservation, ambiguous fragments, tautomer grouping, the missing elements,
the embedding (normalisation, 78 moments, qubit-0 chloride regression,
enantiomer collision, tail modes) and the CLI end to end. `test_cleaning.py`
also checks the committed outputs (1955 rows, 250 previous, 34 charged, 1851
tail, no `source_num` in the modelling table).

To regenerate:

```bash
python -m analysis_features clean
```

The output is deterministic given the raw file and the recorded RDKit version.

---

## Limitations to state in the paper

1. The circuit is stereo-blind; six rows are provably unlearnable from the
   current features.
2. The tail compresses all atoms after the tenth into two scalars; 1851
   molecules are affected.
3. The circuit uses one scalar per atom and one per bond, which is poorer input
   than 13 whole-molecule RDKit descriptors. Classical descriptors lead on this
   dataset; the quantum arm is not expected to beat them. A bag-of-elements +
   size + charge baseline reaches 0.8133, statistically indistinguishable from
   the canonical circuit (0.8015); the map adds 0.05–0.06 over the composition
   of its own ten atoms, but nothing over whole-molecule composition.
4. Any result that uses source atom order or `source_row` is contaminated by
   curation provenance. The modelling table removes both.
5. "Quantum" here means a simulated feature map, not a quantum classifier and
   not hardware advantage.

---

## Data provenance

- Raw table: DeepChem BBBP (`BBBP.csv`), 2050 rows. The file is kept verbatim in
  `data/raw/BBBP.csv`.
- Toolkit: RDKit 2024.03.5 (recorded in `BBBP_clean_settings.json`).
- Previous benchmark: `BBBP_balanced_50_50.csv` from the paper repository,
  copied to `data/raw/BBBP_previous_250.csv`.
- Methods text: `docs/METHODS.md`.

## Original paper material

The notebooks from `main` remain at the repository root
(`10_splits_7_datasets_experiment_ibm_kingston.ipynb`, the hardware runs, the
ablation study, the depth/decay study and the classical benchmark). This branch
adds the dataset and embedding inspection; it does not modify them.
