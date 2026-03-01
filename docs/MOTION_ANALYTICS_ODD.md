# ODD Protocol: Motion Analytics Toolkit — Standardized Behavior Analysis

**Following Grimm et al. (2020), "The ODD Protocol for Describing Agent-Based and Other Simulation Models: A Second Update."**

**Model version:** `Core Analytics v1.0`
**Date:** 2026-02-28
**Authors:** Kathryn Cramer, Claude Opus 4.6
**Repository:** `motion-analytics-toolkit/`

---

## I. OVERVIEW

### 1. Purpose and Patterns

#### 1.1 Purpose
The purpose of this toolkit is to provide a simulator-agnostic framework for analyzing robot behavioral telemetry. It bridges raw kinematic data (x, y, z, rotation) with high-level conceptual metaphors and semantic archetypes. It acts as the analytical backbone for the Rosetta Motion project and other sibling simulators.

#### 1.2 Patterns
The toolkit identifies and validates against the following behavioral and theoretical patterns:
*   **Lakoff Maxim 7 (Ground First, Link Second):** The requirement that abstract metaphors must be grounded in direct sensorimotor observations.
*   **Image Schema Structures:** Canonical patterns of experience (PATH, CYCLE, CONTACT, BALANCE, FORCE) observed in both physical movement and linguistic expression.
*   **Wolfram Class Signatures:** The classification of semantic cellular automata dynamics into four distinct classes (Fixed Point, Periodic, Chaotic, Complex).
*   **Biomechanical Symmetry:** Empirical symmetry indices and energy expenditure metrics from sports science.

### 2. Entities, State Variables, and Scales

#### 2.1 Entities and State Variables
*   **Telemetry Stream (Entity):**
    *   *State:* Multi-variate time series of positions, velocities, and joint angles.
*   **Feature Vector (Entity):**
    *   *State:* 40+ derived features including Curvature, Beer-compatible Dx, Smoothness, and Schema-prefixed metrics.
*   **Lattice (Collective):**
    *   *State:* An N-dimensional grid of motion entries for semantic CA evolution.

#### 2.2 Scales
*   **Temporal:** High-resolution telemetry sampling (typically 10-100Hz).
*   **Conceptual:** Mapping from low-level "Grounded" features to high-level "Linking" abstractions.

### 3. Process Overview and Scheduling
The analytical pipeline follows a strictly ordered schedule:
1.  **Ingestion:** Standardize raw telemetry into JSONL schemas.
2.  **Signal Processing:** Apply smoothing and normalization.
3.  **Image Schema Detection:** Identify primitive patterns (e.g., a repeating CYCLE).
4.  **Feature Extraction:** Calculate kinematic and biomechanical metrics.
5.  **Layer Classification:** Tag features as `grounded` or `linking`.
6.  **Archetype Matching:** Compute behavioral similarity to conceptual personas.
7.  **Violation Auditing:** Check for ICM (Idealized Cognitive Model) warning flags.

---

## II. DESIGN CONCEPTS

### 4. Design Concepts

#### 4.1 Basic Principles (ODD+D)
*   **Theoretical Background:** Based on **Embodied Cognition (Lakoff/Johnson)** and **Biosemiotics**. It assumes that "Motion carries topological weight."
*   **Decision-Making Objectives:** The toolkit attempts to minimize the "ICM Violation Rate"—the divergence between a conceptual model and the physical behavior.

#### 4.2 Emergence
*   **Semantic Phase Transitions:** Emergent Class 4 (Complex) dynamics in the semantic lattice, indicating regions of high behavioral novelty.

#### 4.3 Adaptation
The system classifies "Taxonomy Drift" by comparing how motion entries move across the behavioral lattice over time.

#### 4.4 Sensing
The toolkit senses "Lakoff Grounding Violations" where linking features lack sufficient grounded evidence.

#### 4.5 Interaction
Coordinates with **Rosetta Motion** for dictionary refinement and **PyBullet** for raw data generation.

#### 4.6 Observation
Outputs include `symmetry_index`, `curvature_mean`, `metaphor_warning_logs`, and Wolfram class signatures.

---

## III. DETAILS

### 5. Initialization
Initialized with standardized telemetry objects and a configuration file (`configs/`) defining the feature extraction weights.

### 6. Input Data
Raw JSONL telemetry from Evolutionary Robotics or motion capture; SystemViz visual vocabulary (Stoyko v1.1).

### 7. Submodels
*   **Lakoff Pipeline:** Grounds metaphors in sensorimotor features.
*   **Wolfram Classifier:** Measures the complexity of semantic rule evolution.
*   **SystemViz Tagger:** Annotates behavioral entities with structured system terminology.
