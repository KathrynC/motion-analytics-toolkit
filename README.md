# Motion Analytics Toolkit

A comprehensive, simulator-agnostic toolkit for analyzing motion data from robotics simulations and real motion capture. Built on principles from biomechanics, gesture recognition, dance notation, and sports science, and directly inspired by the theoretical work in the [rosetta-motion](https://github.com/KathrynC/rosetta-motion) project.

## Features

- **Core**: Standardized telemetry schemas, base analyzer classes, signal processing utilities, image schema detectors (PATH, CYCLE, CONTACT, BALANCE, FORCE), SystemViz entity tagging.
- **Kinematics**: Path curvature, workspace analysis, smoothness metrics.
- **Biomechanics**: Gait phase detection, symmetry indices, energy expenditure.
- **Archetypes**: Match motions to conceptual personas (e.g., Deleuze, Borges) using weight-space or behavioral similarity. Lakoff grounding with feature layer classification (grounded vs linking), ICM violation auditing, and metaphor violation detection.
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

### SystemViz Entity Tagging

Entities (archetypes, lattice nodes, scenarios, image schemas, Wolfram classes) can be annotated with terms from the SystemViz Visual Vocabulary (Stoyko, v1.1). Six categories: Driver, Signal, State, Boundary, Relation, Domain. 72 curated elements with default tagging rules for each entity type.

## Installation

```bash
pip install motion-analytics-toolkit
