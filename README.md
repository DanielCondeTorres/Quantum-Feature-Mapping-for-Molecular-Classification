# Full BBBP + 10+2 quantum feature map

Rama **`analysis_features`**, creada desde `main`. Todo el trabajo está en esta rama y subido al fork.

Este README explica **qué se limpió**, **cómo se construye el embedding** y **qué dan los tres experimentos finales**. El código está pensado para ser leído: `analysis_features/three_arms.py` es autocontenido y solo hace los tres brazos.

---

## 1. Los datos (limpios)

Fuente: `data/raw/BBBP.csv`, el crudo de DeepChem con **2050 filas**.

| Etapa | Qué pasa | Filas |
|---|---|---:|
| Raw | Tabla original | 2050 |
| Parseo RDKit | 11 SMILES inválidos (errores de valencia en guanidinas/`[NH2]`), listados en `BBBP_invalid_smiles.csv` | −11 → 2039 |
| Sales/solventes | Se conserva el fragmento carbonado mayor; propranolol pierde `[Cl]` antes de asignar qubits | 105 filas |
| Conflictos de etiqueta | 11 estructuras con labels 0 y 1 se excluyen (propranolol, aspirina, L-DOPA…) | −22 filas |
| Duplicados | Misma estructura con la misma etiqueta → primera aparición | −62 filas |
| **Final** | **1955 moléculas únicas**, 1494 permeables / 461 no permeables | **1955** |

Comprobaciones (`python -m analysis_features verify`): **34/34**. Tests: **55 pasando**.

Decisiones químicas:

- **Cargas:** se conservan (34 estructuras cargadas). Neutralizar cambia la especie que permea.
- **Tautómeros:** no se fusionan; solo se marcan 8 filas en conflicto.
- **Estéreo:** se conserva en SMILES isoméricos, pero el circuito es ciego a quiralidad; 6 filas marcadas.
- **Electronegatividad:** la tabla de Hannan cubría H, Li, Be, B, C, N, O, F, S, Cl. BBBP contiene **Br, I y P**, que antes recibían un valor por defecto silencioso. Añadidos: P 2.19, Br 2.96, I 2.66; un elemento desconocido ahora lanza error.
- Columnas `heavy_atom_count`, `bond_count`, `atoms_beyond_10 = max(0, heavy_atoms − 10)`. El 94.7% supera 10 átomos, así que la cola es el caso normal.
- Los 250 del benchmark anterior sobreviven **250/250** con la misma etiqueta (`in_previous_250`).
- RDKit **2024.03.5**, log por etapas en `data/processed/BBBP_clean_log.txt`.

---

## 2. Métodos: del grafo molecular al circuito

### 2.1 Dibujo del embedding

```mermaid
flowchart LR
  S[SMILES depositado] --> R[RDKit: molécula<br/>grafo de átomos pesados]
  R --> SEL{¿qué 10 átomos<br/>van a q0-q9?}
  SEL -->|paper_smiles_order| P[Primeros 10 átomos<br/>en orden de parseo]
  SEL -->|random_backbone| A[10 átomos al azar]
  SEL -->|community_importance| C[Score RDKit por átomo<br/>+ comunidades networkx]
  P --> Q[q0 ... q9]
  A --> Q
  C --> Q
  R --> T[Resto de átomos = cola]
  T --> TAIL[Suma alternada de χ<br/>A: 0,2,4... → q10<br/>B: 1,3,5... → q11]
  Q --> CIR[Ry(θi) por átomo<br/>Rxx(bond_order/2) por enlace]
  TAIL --> CIR
  CIR --> PSI[Estado de 12 qubits<br/>|ψ⟩]
  PSI --> M[78 momentos Pauli<br/>12 ⟨Z⟩ + 66 ⟨ZZ⟩]
  M --> SVM[SVM RBF, C=10]
  SVM --> Y[Permeable / no permeable]
```

Distribución de cables:

```text
q0 ─Ry(0.1·χ0²)─ q1 ─Ry(0.1·χ1²)─ ... ─ q9 ─Ry(0.1·χ9²)
        └── Rxx(orden_de_enlace / 2) entre átomos del backbone que estén unidos ──┘

q10 ← (0.1/2) · Σ χ(átomos de la cola en posiciones pares)
q11 ← (0.1/2) · Σ χ(átomos de la cola en posiciones impares)

Un enlace backbone–cola emite DOS compuertas Rxx: una a q10 y otra a q11.
Un enlace cola–cola no emite Rxx (sus átomos ya entraron en las sumas).
```

### 2.2 ¿Es una suma para clasificar?

**No.** La clasificación no es una suma de propiedades atómicas. El circuito produce un estado cuántico de 12 qubits y de él se leen **78 números** (momentos de Pauli). Esos 78 números van a una **SVM con kernel RBF**, que es la que decide la etiqueta. Las únicas sumas son internas al circuito y tienen una escala fija:

1. **Suma de la cola:** `θ10 = 0.05·Σχ_A` y `θ11 = 0.05·Σχ_B`. Se usan los valores de Pauling directamente, sin z-score. El factor `0.1/2` es una constante fija del paper, no un parámetro ajustado.
2. **Score de selección (solo en `community_importance`):** sí es una suma ponderada, pero **se normaliza dentro de cada molécula** para que las propiedades sean comparables:

```text
score_i = Σ_k  w_k · |x_i,k − media_k| / max_j |x_j,k − media_k|
```

Los `w_k` son fijos y suman 1. Así cada propiedad queda en `[0,1]` antes de sumarse. No se ajustan a etiquetas.

Además, la SVM estandariza los 78 momentos con `StandardScaler` (z-score), pero **ajustado solo dentro del fold de entrenamiento**, nunca con datos de test. Eso es lo único que se normaliza por z-score.

### 2.3 Propiedades que usa `community_importance`

| Propiedad | Función RDKit / acceso | Peso | Por qué importa para BBBP |
|---|---|---:|---|
| Contribución atómica al TPSA | `rdMolDescriptors._CalcTPSAContribs(mol)` | 0.20 | Polaridad; TPSA alto penaliza permeabilidad pasiva |
| Contribución al logP | `rdMolDescriptors._CalcCrippenContribs(mol)` | 0.20 | Lipofilia; facilita el paso por la bicapa |
| Refractividad molar (MR) | `_CalcCrippenContribs(mol)` | 0.10 | Polarizabilidad / volumen estérico |
| Donor de H (HBD) | Roles SMARTS de RDKit (`BaseFeatures.fdef`) | 0.15 | Desolvatación; cuesta más que un aceptor |
| Aceptor de H (HBA) | Roles SMARTS | 0.10 | Interacción con agua |
| Carga parcial Gasteiger | `AllChem.ComputeGasteigerCharges(mol)` | 0.10 | Momento dipolar local |
| Carga formal | `atom.GetFormalCharge()` | 0.05 | Especies ionizadas permean peor |
| Heteroátomo | `atom.GetAtomicNum() != 6` | 0.05 | N, O, S, halógenos |
| Aromaticidad | `atom.GetIsAromatic()` | 0.05 | Nitrógeno aromático vs amida/básico |

Luego `networkx.algorithms.community.greedy_modularity_communities` divide el grafo de átomos pesados en comunidades. Se coge el átomo de mayor score de cada comunidad y se rellenan los huecos con el ranking global. Así los 10 qubits representan regiones distintas en vez de concentrarse en un mismo grupo funcional. Los átomos restantes siguen comprimiéndose en q10/q11: en un diseño fijo de 12 qubits no se puede eliminar, pero sí se reparte mejor.

**Los otros dos brazos:** `paper_smiles_order` reproduce la regla literal del paper (primeros 10 átomos del SMILES depositado; sin selección química) y `random_backbone` elige 10 átomos al azar como control.

---

## 3. ¿Qué es el AUC?

AUC = *Area Under the ROC Curve*. Mide qué tan bien el modelo **ordena** las moléculas: es la probabilidad de que una molécula permeable aleatoria reciba una puntuación más alta que una no permeable aleatoria.

- **0.5** = aleatorio.
- **1.0** = separación perfecta.
- Es **independiente del umbral**, por eso se usa junto con la matriz de confusión (que sí depende del umbral 0 de la SVM).

En este dataset el 76% de las moléculas son permeables: predecir siempre "permeable" da **accuracy 0.764** pero AUC 0.5. Por eso también se reporta *balanced accuracy* (media de la sensibilidad y la especificidad). En los experimentos completos del repo el AUC se acompaña de un intervalo bootstrap al 95%.

---

## 4. Resultados: tres brazos, mismas condiciones

Dataset completo limpio (1955 moléculas), 5 folds por scaffold, circuito del paper con la cola alternada, 78 momentos, SVM RBF C=10. Solo cambia la selección de q0–q9.

| Brazo | Accuracy | Balanced acc. | AUC | IC 95% AUC |
|---|---:|---:|---:|---:|
| `paper_smiles_order` | **0.8501** | 0.7760 | **0.8559** | — |
| `random_backbone` | 0.7407 | 0.5828 | 0.6399 | — |
| `community_importance` | 0.8036 | 0.7035 | 0.7679 | — |

### Matrices de confusión

Filas = etiqueta real, columnas = predicción. Orden: no permeable (0), permeable (1).

| Brazo | | Pred. no perm. | Pred. perm. |
|---|---|---:|---:|
| **paper_smiles_order** | Real no perm. | **TN 293** | FP 168 |
| | Real perm. | FN 125 | **TP 1369** |
| **random_backbone** | Real no perm. | **TN 131** | FP 330 |
| | Real perm. | FN 177 | **TP 1317** |
| **community_importance** | Real no perm. | **TN 237** | FP 224 |
| | Real perm. | FN 160 | **TP 1334** |

Imagen: `results/confusion_matrices_three_arms.png` (tres paneles, una matriz por brazo).

**Lectura:** `random_backbone` (0.741) está incluso por debajo del baseline trivial de 0.764: elegir átomos al azar no funciona. `community_importance` (0.804) sí aprende química: gana 0.063 de accuracy y 0.128 de AUC sobre el azar. `paper_smiles_order` da la cifra más alta (0.850), pero esa ventaja es **fuga de procedencia**: el orden en que los SMILES fueron depositados agrupa moléculas por origen, no por química. Al canonizar o destruir la secuencia, esa ventaja desaparece.

---

## 5. Código

```text
analysis_features/
  cleaning.py       # pipeline de limpieza por etapas
  elements.py       # tabla de Hannan + extensiones + auditoría
  standardize.py    # parseo, sales, SMILES canónico
  atom_order.py     # score de importancia + comunidades (networkx)
  embedding.py      # simulador exacto del circuito 10+2
  three_arms.py     # LOS TRES BRAZOS: build_features → evaluate → plot → run
  verify.py         # 34 comprobaciones del dataset
  io.py / cli.py    # lectura/escritura y comandos
```

`three_arms.py` es el único módulo necesario para reproducir la tabla y la figura final: cuatro funciones pequeñas (`build_features`, `evaluate`, `plot`, `run`) y tres dataclasses. No hay Qiskit: el estado se simula con NumPy.

---

## 6. Conclusiones para el equipo

1. El dataset limpio es **1955** moléculas; antes se usaba un fichero que ya había perdido 11 inválidas.
2. Br, I y P recibían electronegatividad inventada; ya están corregidos.
3. El propranolol ya no pone Cl en q0 y sus dos entradas con etiquetas opuestas se excluyen.
4. El circuito usa **electronegatividad de Pauling**, no TPSA. TPSA aparece solo en la ablación de selección atómica.
5. Elegir q0 al azar no funciona (AUC 0.640). Elegirlo por propiedades RDKit + comunidades funciona mejor (0.768) pero no supera al orden del paper (0.856), que es fuga.
6. La clasificación la hace una **SVM clásica** sobre 78 momentos; el circuito es un mapa de features, no un clasificador cuántico.
7. Limitaciones: ciego al estéreo, cola comprimida para el 94.7% de las moléculas, y sin ventaja demostrada sobre descriptores clásicos (SVM Morgan: AUC 0.882).

---

## 7. Reproducir

```bash
python -m pip install -r requirements.txt

python -m analysis_features clean          # regenera data/processed/
python -m analysis_features verify         # 34/34
python -m analysis_features three-arms     # CSV, JSON y PNG de las 3 matrices
python -m pytest                           # 55 tests
```
