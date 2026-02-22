# Motion Analytics Toolkit

A comprehensive, simulator-agnostic toolkit for analyzing motion data from robotics simulations and real motion capture. Built on principles from biomechanics, gesture recognition, dance notation, and sports science, and directly inspired by the theoretical work in the [rosetta-motion](https://github.com/KathrynC/rosetta-motion) project.

## Features

- **Core**: Standardized telemetry schemas, base analyzer classes, signal processing utilities, image schema detectors (PATH, CYCLE, CONTACT, BALANCE, FORCE), SystemViz entity tagging.
- **Kinematics**: Path curvature, workspace analysis, smoothness metrics.
- **Biomechanics**: Gait phase detection, symmetry indices, energy expenditure.
- **Archetypes**: 48 behavioral archetypes from 116 gaits across 6 source types (simulation-grounded, literary/cultural, motion-word, character/celebrity, math-seeded, interpolation/meta). Lakoff grounding with feature layer classification (grounded vs linking), ICM violation auditing, and metaphor violation detection.
- **Scenarios**: Define environmental perturbations, run stress tests, compute robustness and vulnerability. Weight perturbation, ablation, interpolation, and cross-topology suites.
- **Semantic CA**: Build a lattice over motion dictionary entries, apply local rules, detect clusters and phase transitions. Wolfram CA classification (Classes 1-4).
- **Patterns**: Alexander Pattern Language DAG with validation, topological sequencing, and coverage analysis.
- **IO**: Loaders for Evolutionary-Robotics telemetry (JSONL) and motion dictionary JSON.

### Lakoff Architecture

The toolkit implements a grounded conceptual metaphor pipeline following Lakoff's experiential structures:

```
Image Schema Detection → Behavioral Feature Extraction → Feature Layer Classification → Archetype Matching → Violation Auditing
```

- **Image schemas** (PATH, CYCLE, CONTACT, BALANCE, FORCE) are detected from raw telemetry and wired directly into the feature extraction pipeline, producing ~40 features (10 canonical + Beer-compatible + 16 schema-prefixed + 8 promoted top-level).
- **Feature layers** classify every feature as `grounded` (directly observable sensorimotor) or `linking` (cross-domain abstraction), enforcing Lakoff Maxim 7: ground first, link second.
- **MetaphorAuditor** checks grounding criteria, ICM assumptions, and reports `layer_warnings` when grounding criteria reference linking features.

### Wolfram CA Classification

The semantic CA module includes a `WolframClassifier` that evolves a lattice under a rule for N generations and classifies the dynamics into one of Wolfram's four classes:

| Class | Behavior | Signature |
|-------|----------|-----------|
| 1 | Fixed point | Entropy collapses, order converges to 1.0 |
| 2 | Periodic | Entropy stable, low tail variance, clusters stable |
| 3 | Chaotic | High entropy, high variance, unstable clusters |
| 4 | Complex | Intermediate entropy, structured boundary activity |

### Archetype Atlas

48 archetypes organized into 6 source families, each with grounding criteria, ICM definitions, and violation conditions. Notable archetype groups include:

- **The Triptych** (Revelation/Ecclesiastes/Noether): Three gaits from the same anti-phase sign family, separated only by magnitude balance — spanning from maximum displacement (pale_horse, DX=29.17m) to maximum efficiency (whirling_wind, 0.00495) to perfect stasis (conservation_law, DX=0.031m). Demonstrates that structural principles survive substrate transfer from scripture to LLM weights to robot synapses.
- **Gallop variants** (charging/tumbling/drifting): A single motion concept that splits into three distinct behavioral modes depending on LLM interpretation.
- **Simulation-grounded** (11 archetypes): Derived from the 116-gait zoo with real feature vectors from PyBullet telemetry.
- **Movement vocabulary** (8 archetypes): Semantic clusters from cross-language, cross-model LLM experiments (5 LLMs, 5 languages, 495+ gaits).

### Visualizations

Interactive D3.js visualizations in `hero-concepts/`:

- **Gait Archetype Atlas** (`06_archetype_atlas.html`): All 48 archetypes positioned by PCA of their 6D weight vectors, with Stoyko SystemViz category shapes, source-family coloring, and interactive tooltips showing grounding criteria and ICM definitions.
- **Weight Hypergraph** (`07_weight_hypergraph.html`): Each archetype rendered as its actual 3-sensor → 2-motor bipartite weight graph (green=excitatory, pink=inhibitory, width=magnitude). Four hyperedge modes reveal structural relationships: sign-pattern family, strongest channel match, magnitude balance similarity, and triptych anti-phase family. Tooltips show ICM names, source titles, sign patterns, and per-channel balance analysis.
- **Persona Weight Hypergraph** (`08_persona_weight_hypergraph.html`): 2,315 personas (2,000 fictional characters, 132 celebrities, 79 politicians, 104 mathematicians) positioned by PCA of their 6D weight vectors. K-means clustering (k=17) with convex hull overlays, 6-category color coding, name search, category filters, notable-only mode with edge visualizations (cluster, sign-pattern, cross-category twin, story family), and hover detail panel with bipartite weight glyph and behavioral metrics.

### SystemViz Entity Tagging

Entities (archetypes, lattice nodes, scenarios, image schemas, Wolfram classes) can be annotated with terms from the SystemViz Visual Vocabulary (Stoyko, v1.1). Six categories: Driver, Signal, State, Boundary, Relation, Domain. 72 curated elements with default tagging rules for each entity type.

## Installation

```bash
pip install motion-analytics-toolkit
