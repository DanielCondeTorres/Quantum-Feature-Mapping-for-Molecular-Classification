# molecular_encoding

A self-contained Python module for the QAMP 2027 molecular representation study.

This module implements and compares atom-to-qubit assignment strategies for the
10+2 quantum feature map, independent of the original QAMP 2026 notebooks.

## Structure

```
molecular_encoding/
├── __init__.py
├── preprocessing.py      # RDKit molecular cleaning pipeline
├── atom_selection.py     # Atom ordering & qubit assignment strategies
├── properties.py         # Atomic property extractors (EN, Gasteiger, etc.)
├── circuit_builder.py    # Build QuantumCircuit from assignment
├── collision_analysis.py # Circuit fingerprinting and collision detection
├── visualization.py      # Molecular graph + qubit assignment plots
└── tests/
    ├── test_preprocessing.py
    ├── test_atom_selection.py
    └── test_collision.py
```

## Usage

```python
from molecular_encoding import encode_molecule

# Default: scaffold + centrality ordering, Pauling EN
qc = encode_molecule("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen

# With options
from molecular_encoding.atom_selection import Strategy
qc = encode_molecule(smiles, strategy=Strategy.SCAFFOLD_CENTRALITY)
```

## Does NOT modify anything in the root notebooks.
