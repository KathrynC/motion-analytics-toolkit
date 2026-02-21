# Motion Analytics Toolkit

A comprehensive, simulator‑agnostic toolkit for analyzing motion data from robotics simulations and real motion capture. Built on principles from biomechanics, gesture recognition, dance notation, and sports science, and directly inspired by the theoretical work in the [rosetta-motion](https://github.com/KathrynC/rosetta-motion) project.

## Features

- **Core**: Standardized telemetry schemas, base analyzer classes, signal processing utilities.
- **Kinematics**: Path curvature, workspace analysis, smoothness metrics.
- **Biomechanics**: Gait phase detection, symmetry indices, energy expenditure.
- **Archetypes**: Match motions to conceptual personas (e.g., Deleuze, Borges) using weight‑space or behavioral similarity.
- **Scenarios**: Define environmental perturbations, run stress tests, compute robustness and vulnerability.
- **Semantic CA**: Build a lattice over motion dictionary entries, apply local rules, detect clusters and phase transitions.
- **IO**: Loaders for Evolutionary‑Robotics telemetry (JSONL) and motion dictionary JSON.
- **Utils**: Forward kinematics for the 3‑link robot.

## Installation

```bash
pip install motion-analytics-toolkit
