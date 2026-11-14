# molecular_encoding
### Chemically Grounded Quantum Feature Mapping for Molecular Classification

A self-contained Python module designed for the **QAMP 2027** follow-up study to evaluate exponential expressivity and representation bias in the 10+2 Quantum Feature Map on NISQ hardware.

---

## 1. Core Mathematical and Chemical Concepts

### 1.1 Bemis-Murcko Molecular Scaffold
In medicinal chemistry, **Bemis & Murcko (J. Med. Chem. 1996, 39, 2887–2893)** introduced a systematic decomposition of drug molecules into four structural components:
1. **Ring Systems:** All cyclic substructures (aromatic, heteroaromatic, aliphatic rings).
2. **Linkers:** Acyclic chains of bonds connecting two or more ring systems.
3. **Framework (Scaffold):** The union of all ring systems and linkers.
4. **Side Chains:** All non-ring, non-linker terminal substituents.

```
       [ Side Chain ]                       [ Side Chain ]
       -CH2-CH(CH3)2                        -CH(CH3)-COOH
             \                                    /
              [===== BEMIS-MURCKO RING SYSTEM =====]
              [           (Benzene Core)           ]
```

**Why it matters for Qubit Mapping:**
In typical drug datasets (e.g. BBBP, BACE), the core scaffold carries the critical pharmacophoric elements and aromatic electron density that govern membrane permeability and receptor binding. 
* *Previous limitation:* Truncating atoms by raw SMILES text index (`atoms[:10]`) frequently shoved the scaffold ring into the auxiliary tail if the SMILES string began with a side chain.
* *Our approach:* Atoms belonging to the Bemis-Murcko scaffold are given **highest priority** to occupy the individual backbone qubits (`q0` to `q5`–`q9`).

---

### 1.2 Betweenness Centrality in Molecular Graphs
A molecule is represented as an undirected graph $G = (V, E)$, where vertices $V$ are heavy atoms and edges $E$ are covalent bonds weighted by bond order.

The **Betweenness Centrality** $C_B(v)$ of an atom $v \in V$ is defined as the fraction of all shortest topological paths between all pairs of atoms that pass through $v$:

$$C_B(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

where:
* $\sigma_{st}$ is the total number of shortest paths from atom $s$ to atom $t$.
* $\sigma_{st}(v)$ is the number of those shortest paths that pass through atom $v$.

**Chemical Significance:**
Atoms with high betweenness centrality are the **topological hubs** or "bridges" connecting functional groups and rings. Assigning high-centrality atoms to lower qubit indices ensures that the most structurally influential atoms receive the primary entangling gates ($R_{xx}$) in the quantum circuit.

---

### 1.3 Readout: What is Being Measured?

#### Current Readout in the 10+2 Framework:
* The circuit ends with `qc.measure_all()`, measuring all 12 qubits in the computational basis ($\{|0\rangle, |1\rangle\}$).
* The output feature vector is the **full measurement probability distribution**:
  $$\vec{p} = \Big[ p(000000000000),\, p(000000000001),\, \dots,\, p(111111111111) \Big] \in \mathbb{R}^{4096}$$
* **Limitation:** Sampling $2^{12} = 4096$ states with 1024 or 4096 shots introduces severe shot-noise sparsity.

#### Planned Readout Alternatives:
1. **Single-Qubit Pauli Z Expectations ($\langle Z_i \rangle$):**
   $$\langle Z_i \rangle = p(q_i = 0) - p(q_i = 1) \in [-1, 1], \quad i \in \{0, \dots, 11\} \implies \mathbb{R}^{12}$$
2. **Two-Qubit Correlators ($\langle Z_i Z_j \rangle$):**
   $$\langle Z_i Z_j \rangle = p(q_i \oplus q_j = 0) - p(q_i \oplus q_j = 1), \quad \binom{12}{2} = 66 \implies \mathbb{R}^{66}$$
3. **Joint Vector:** $[\langle Z_i \rangle, \langle Z_i Z_j \rangle] \in \mathbb{R}^{78}$.

---

## 2. Directory Structure

```
molecular_encoding/
├── __init__.py           # Public API: encode_molecule()
├── preprocessing.py      # RDKit desalting, valence check, canonicalization
├── atom_selection.py     # Bemis-Murcko + Betweenness Centrality assignment
├── properties.py         # Property extractors (Pauling EN, Crippen LogP, Gasteiger)
├── circuit_builder.py    # QuantumCircuit construction with uncoupled tail fix
├── collision_analysis.py # Fingerprinting & collision detection across datasets
├── visualization.py      # Molecular graph and circuit heatmap plotting
└── tests/
    └── test_atom_selection.py
```

---

## 3. Benchmarks on BBBP (Stratified 10-Split CV)

| Method / Atom Selection Strategy | Accuracy | AUC-ROC | F1-Score | Status / Insight |
|---|:---:|:---:|:---:|---|
| **Classical (Random Forest + ECFP4)** | 0.625 ± 0.068 | 0.654 ± 0.091 | 0.632 ± 0.089 | Classical reference |
| **Quantum SMILES Original (Paper baseline)** | 0.570 ± 0.103 | 0.613 ± 0.110 | 0.524 ± 0.165 | SMILES-syntax dependent |
| **Quantum Murcko Scaffold + Centrality** | 0.595 ± 0.106 | 0.673 ± 0.070 | 0.587 ± 0.125 | Scaffold invariant, but suffers carbon-angle degeneracy |
| **Quantum Pharmacophore-First** | **0.675 ± 0.119** | **0.736 ± 0.095** | **0.697 ± 0.115** | Prioritizes heteroatoms & H-bonding |
| **Quantum Canonical Morgan Rank** 🚀 | **0.695 ± 0.079** | **0.760 ± 0.093** | **0.725 ± 0.072** | **Massive leap (+12.5% Acc, +14.7% AUC)** |
| Negative Control (Random atom assignment) | 0.575 ± 0.060 | 0.559 ± 0.112 | 0.543 ± 0.106 | Topology scrambled (drops to random baseline) |
