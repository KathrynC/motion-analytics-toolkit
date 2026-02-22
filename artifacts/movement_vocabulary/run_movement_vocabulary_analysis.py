#!/usr/bin/env python3
"""
Comprehensive behavioral analysis of the movement vocabulary experiment.
700 trials: 28 concepts x 5 languages x 5 models.
Outputs: movement_vocabulary_analysis.json
"""
import json
import numpy as np
import os
import sys
from collections import defaultdict

# ── NumpyEncoder ──────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ── Load data ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "motion_seed_experiment_v2.json")) as f:
    main_data = json.load(f)

with open(os.path.join(BASE,
          "motion_seed_experiment_v2.singlemodel.20260221_160301.json")) as f:
    backup_data = json.load(f)

all_results = main_data["results"] + backup_data["results"]
print(f"Loaded {len(main_data['results'])} + {len(backup_data['results'])} = {len(all_results)} trials")

# ── Filter to successful trials with analytics ────────────────────────
trials = [r for r in all_results if r.get("success") and r.get("analytics")]
print(f"Successful trials with analytics: {len(trials)}")

# ── Helper: extract key metrics from a trial ──────────────────────────
KEY_METRICS = ["mean_speed", "dx", "phase_lock_score", "contact_entropy_bits",
               "path_straightness", "yaw_net_rad"]

def get_metrics(t):
    """Extract the 6 key behavioral metrics from a trial."""
    a = t["analytics"]
    o = a.get("outcome", {})
    co = a.get("coordination", {})
    ct = a.get("contact", {})
    return {
        "mean_speed": o.get("mean_speed", 0.0),
        "dx": o.get("dx", 0.0),
        "phase_lock_score": co.get("phase_lock_score", 0.0),
        "contact_entropy_bits": ct.get("contact_entropy_bits", 0.0),
        "path_straightness": o.get("path_straightness", 0.0),
        "yaw_net_rad": o.get("yaw_net_rad", 0.0),
    }

def get_all_metrics(t):
    """Extract all numeric metrics from a trial."""
    a = t["analytics"]
    o = a.get("outcome", {})
    co = a.get("coordination", {})
    ct = a.get("contact", {})
    ra = a.get("rotation_axis", {})
    m = {}
    # Outcome metrics
    for k in ["dx", "dy", "yaw_net_rad", "mean_speed", "speed_cv",
              "work_proxy", "distance_per_work", "path_length",
              "path_straightness", "heading_consistency"]:
        m[k] = o.get(k, 0.0)
    # Contact metrics
    for k in ["duty_torso", "duty_back", "duty_front", "contact_entropy_bits"]:
        m[k] = ct.get(k, 0.0)
    # Coordination
    m["phase_lock_score"] = co.get("phase_lock_score", 0.0)
    m["delta_phi_rad"] = co.get("delta_phi_rad", 0.0)
    j0 = co.get("joint_0", {})
    j1 = co.get("joint_1", {})
    m["freq_back"] = j0.get("dominant_freq_hz", 0.0)
    m["amp_back"] = j0.get("dominant_amplitude", 0.0)
    m["freq_front"] = j1.get("dominant_freq_hz", 0.0)
    m["amp_front"] = j1.get("dominant_amplitude", 0.0)
    # Rotation axis
    ad = ra.get("axis_dominance", [0, 0, 0])
    m["roll_dominance"] = ad[0] if len(ad) > 0 else 0.0
    m["pitch_dominance"] = ad[1] if len(ad) > 1 else 0.0
    m["yaw_dominance"] = ad[2] if len(ad) > 2 else 0.0
    m["axis_switching_rate_hz"] = ra.get("axis_switching_rate_hz", 0.0)
    per = ra.get("periodicity", {})
    m["roll_freq_hz"] = per.get("roll_freq_hz", 0.0)
    m["pitch_freq_hz"] = per.get("pitch_freq_hz", 0.0)
    m["yaw_freq_hz"] = per.get("yaw_freq_hz", 0.0)
    return m

def get_weights(t):
    """Extract weight vector as a 6-element numpy array."""
    w = t["weights"]
    return np.array([w.get("w03", 0), w.get("w04", 0), w.get("w13", 0),
                     w.get("w14", 0), w.get("w23", 0), w.get("w24", 0)],
                    dtype=np.float64)

# ── Statistical helpers (no scipy) ────────────────────────────────────
def mann_whitney_u(x, y):
    """
    Manual Mann-Whitney U test.
    Returns U statistic, z-score, and approximate two-sided p-value.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return {"U": None, "z": None, "p_approx": None}

    # Combine and rank
    combined = np.concatenate([x, y])
    # Use average ranks for ties
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)

    # Handle ties: average ranks for tied values
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    R1 = np.sum(ranks[:nx])
    U1 = R1 - nx * (nx + 1) / 2.0
    U2 = nx * ny - U1
    U = min(U1, U2)

    # Normal approximation
    mu = nx * ny / 2.0
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sigma == 0:
        return {"U": float(U), "z": 0.0, "p_approx": 1.0}
    z = (U - mu) / sigma

    # Approximate p-value using normal CDF approximation
    # Using the Abramowitz and Stegun approximation for Phi(x)
    p_approx = 2.0 * _norm_cdf(z)  # two-sided

    return {"U": float(U), "z": float(z), "p_approx": float(p_approx)}

def _norm_cdf(z):
    """Approximation of the standard normal CDF for negative z values (left tail)."""
    # We want the left-tail probability, which for the U-test is P(Z <= z) where z is negative
    z = abs(z)
    # Abramowitz and Stegun 26.2.17
    p = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + p * z)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cdf = pdf * (b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
    return float(cdf)

def cohens_d(x, y):
    """Cohen's d effect size."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def skewness(arr):
    """Manual skewness calculation."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return 0.0
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return 0.0
    return float(np.mean(((arr - mu) / sigma) ** 3))

def metrics_to_vec(metrics_dict):
    """Convert key metrics dict to vector for cosine similarity."""
    return np.array([metrics_dict.get(k, 0.0) for k in KEY_METRICS], dtype=np.float64)

# ── Index trials ──────────────────────────────────────────────────────
concept_trials = defaultdict(list)
language_trials = defaultdict(list)
model_trials = defaultdict(list)
concept_lang_trials = defaultdict(list)
concept_model_trials = defaultdict(list)

for t in trials:
    c = t["concept"]
    l = t["language"]
    m = t["model"]
    concept_trials[c].append(t)
    language_trials[l].append(t)
    model_trials[m].append(t)
    concept_lang_trials[(c, l)].append(t)
    concept_model_trials[(c, m)].append(t)

all_concepts = sorted(set(t["concept"] for t in trials))
all_languages = sorted(set(t["language"] for t in trials))
all_models = sorted(set(t["model"] for t in trials))

print(f"Concepts: {len(all_concepts)} - {all_concepts}")
print(f"Languages: {len(all_languages)} - {all_languages}")
print(f"Models: {len(all_models)} - {all_models}")

# ── Compute grand statistics ──────────────────────────────────────────
all_metric_keys = list(get_all_metrics(trials[0]).keys())
grand_metrics = {k: [] for k in all_metric_keys}
for t in trials:
    m = get_all_metrics(t)
    for k in all_metric_keys:
        grand_metrics[k].append(m[k])
grand_mean = {k: float(np.mean(grand_metrics[k])) for k in all_metric_keys}
grand_std = {k: float(np.std(grand_metrics[k])) for k in all_metric_keys}

print("\n=== Grand means ===")
for k in KEY_METRICS:
    print(f"  {k}: {grand_mean[k]:.4f} +/- {grand_std[k]:.4f}")

# ======================================================================
# SECTION 1: SEMANTIC COHERENCE
# ======================================================================
print("\n=== Section 1: Semantic Coherence ===")

semantic_clusters = {
    "speed_fast": ["sprint", "dash", "gallop", "scurry", "charge"],
    "speed_slow": ["crawl", "plod", "tiptoe", "drag", "drift"],
    "rotation": ["twirl", "pivot", "circle", "turn_left", "turn_right"],
    "oscillation": ["rock", "sway", "wobble"],
    "stability": ["freeze", "march", "stomp"],
    "locomotion_style": ["hop", "slide", "stagger", "zigzag", "retreat", "patrol", "headstand"],
}

cluster_stats = {}
for cluster_name, concepts in semantic_clusters.items():
    cluster_trials_list = []
    for c in concepts:
        cluster_trials_list.extend(concept_trials.get(c, []))

    if not cluster_trials_list:
        continue

    # Compute mean+std of key metrics
    cluster_metrics = {k: [] for k in KEY_METRICS}
    for t in cluster_trials_list:
        m = get_metrics(t)
        for k in KEY_METRICS:
            cluster_metrics[k].append(m[k])

    stats = {}
    for k in KEY_METRICS:
        arr = np.array(cluster_metrics[k])
        stats[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "n": len(arr),
        }

    cluster_stats[cluster_name] = {
        "concepts": concepts,
        "n_trials": len(cluster_trials_list),
        "metrics": stats,
    }
    print(f"  {cluster_name}: {len(cluster_trials_list)} trials, "
          f"mean_speed={stats['mean_speed']['mean']:.4f}, dx={stats['dx']['mean']:.4f}")

# Mann-Whitney U between fast vs slow
fast_trials = []
for c in semantic_clusters["speed_fast"]:
    fast_trials.extend(concept_trials.get(c, []))
slow_trials = []
for c in semantic_clusters["speed_slow"]:
    slow_trials.extend(concept_trials.get(c, []))

comparisons = {}
for metric in ["mean_speed", "dx", "contact_entropy_bits"]:
    fast_vals = [get_metrics(t)[metric] for t in fast_trials]
    slow_vals = [get_metrics(t)[metric] for t in slow_trials]
    u_result = mann_whitney_u(fast_vals, slow_vals)
    d = cohens_d(fast_vals, slow_vals)
    comparisons[metric] = {
        "fast_mean": float(np.mean(fast_vals)),
        "fast_std": float(np.std(fast_vals)),
        "slow_mean": float(np.mean(slow_vals)),
        "slow_std": float(np.std(slow_vals)),
        "mann_whitney_U": u_result["U"],
        "mann_whitney_z": u_result["z"],
        "mann_whitney_p_approx": u_result["p_approx"],
        "cohens_d": d,
        "n_fast": len(fast_vals),
        "n_slow": len(slow_vals),
    }
    print(f"  Fast vs Slow [{metric}]: d={d:.3f}, U={u_result['U']:.0f}, "
          f"p~{u_result['p_approx']:.4f}")

semantic_coherence = {
    "cluster_statistics": cluster_stats,
    "fast_vs_slow_comparisons": comparisons,
    "description": "Semantic clusters grouped by movement type. Mann-Whitney U tests "
                   "and Cohen's d effect sizes compare fast vs slow speed concepts on "
                   "key behavioral metrics."
}

# ======================================================================
# SECTION 2: CROSS-LANGUAGE FUNCTOR
# ======================================================================
print("\n=== Section 2: Cross-Language Functor ===")

cross_lang = {}
all_cosines = []

for concept in all_concepts:
    lang_means = {}
    for lang in all_languages:
        ts = concept_lang_trials.get((concept, lang), [])
        if not ts:
            continue
        vecs = [metrics_to_vec(get_metrics(t)) for t in ts]
        lang_means[lang] = np.mean(vecs, axis=0)

    if len(lang_means) < 2:
        cross_lang[concept] = {"n_languages": len(lang_means), "insufficient_data": True}
        continue

    # CV across languages for each metric
    lang_metric_vals = {k: [] for k in KEY_METRICS}
    for lang, vec in lang_means.items():
        for i, k in enumerate(KEY_METRICS):
            lang_metric_vals[k].append(vec[i])

    cv_per_metric = {}
    for k in KEY_METRICS:
        arr = np.array(lang_metric_vals[k])
        mu = np.mean(arr)
        if abs(mu) > 1e-10:
            cv_per_metric[k] = float(np.std(arr) / abs(mu))
        else:
            cv_per_metric[k] = float(np.std(arr))  # can't normalize by near-zero mean

    # Cosine similarity between all language pairs
    langs_present = sorted(lang_means.keys())
    pair_cosines = {}
    for i in range(len(langs_present)):
        for j in range(i+1, len(langs_present)):
            l1, l2 = langs_present[i], langs_present[j]
            cs = cosine_similarity(lang_means[l1], lang_means[l2])
            pair_cosines[f"{l1}-{l2}"] = cs
            all_cosines.append(cs)

    mean_cosine = float(np.mean(list(pair_cosines.values()))) if pair_cosines else 0.0
    mean_cv = float(np.mean(list(cv_per_metric.values())))

    cross_lang[concept] = {
        "n_languages": len(lang_means),
        "cv_per_metric": cv_per_metric,
        "mean_cv": mean_cv,
        "language_pair_cosines": pair_cosines,
        "mean_cosine_similarity": mean_cosine,
    }

# Rank concepts by translation invariance
ranked_concepts = sorted(
    [(c, d.get("mean_cosine_similarity", 0)) for c, d in cross_lang.items()
     if not d.get("insufficient_data")],
    key=lambda x: -x[1]
)

functor_faithfulness = float(np.mean(all_cosines)) if all_cosines else 0.0

cross_language_functor = {
    "per_concept": cross_lang,
    "ranking_most_invariant": [{"concept": c, "mean_cosine": s} for c, s in ranked_concepts[:10]],
    "ranking_least_invariant": [{"concept": c, "mean_cosine": s} for c, s in ranked_concepts[-10:]],
    "functor_faithfulness_score": functor_faithfulness,
    "total_language_pair_cosines_computed": len(all_cosines),
    "description": "For each concept, behavioral vectors (speed, dx, PLV, entropy, "
                   "straightness, yaw) are averaged per language, then compared via CV "
                   "and cosine similarity across languages. Functor faithfulness is the "
                   "mean cosine similarity across all concept x language-pair combinations."
}

print(f"  Functor faithfulness: {functor_faithfulness:.4f}")
print(f"  Most invariant: {ranked_concepts[0][0]} ({ranked_concepts[0][1]:.4f})")
print(f"  Least invariant: {ranked_concepts[-1][0]} ({ranked_concepts[-1][1]:.4f})")

# ======================================================================
# SECTION 3: MODEL FINGERPRINTS
# ======================================================================
print("\n=== Section 3: Model Fingerprints ===")

model_fingerprints = {}
weight_names = ["w03", "w04", "w13", "w14", "w23", "w24"]

for model in all_models:
    mt = model_trials.get(model, [])
    if not mt:
        continue

    # Mean behavioral profile
    all_m_metrics = {k: [] for k in all_metric_keys}
    for t in mt:
        am = get_all_metrics(t)
        for k in all_metric_keys:
            all_m_metrics[k].append(am[k])

    mean_profile = {k: float(np.mean(all_m_metrics[k])) for k in all_metric_keys}
    std_profile = {k: float(np.std(all_m_metrics[k])) for k in all_metric_keys}

    # Which concepts have lowest within-concept variance?
    concept_variances = {}
    for concept in all_concepts:
        cts = concept_model_trials.get((concept, model), [])
        if len(cts) < 2:
            continue
        # Mean variance across key metrics
        variances = []
        for k in KEY_METRICS:
            vals = [get_metrics(t)[k] for t in cts]
            variances.append(np.var(vals))
        concept_variances[concept] = float(np.mean(variances))

    sorted_cv = sorted(concept_variances.items(), key=lambda x: x[1])
    best_concepts = [{"concept": c, "mean_variance": v} for c, v in sorted_cv[:5]]
    worst_concepts = [{"concept": c, "mean_variance": v} for c, v in sorted_cv[-5:]]

    # Weight distribution statistics
    weights_arr = np.array([get_weights(t) for t in mt])
    weight_stats = {}
    for i, wn in enumerate(weight_names):
        w_col = weights_arr[:, i]
        weight_stats[wn] = {
            "mean": float(np.mean(w_col)),
            "std": float(np.std(w_col)),
            "min": float(np.min(w_col)),
            "max": float(np.max(w_col)),
            "skew": skewness(w_col),
        }

    # Collapse rate: fraction with cosine sim > 0.95 to another trial
    n_trials_model = len(mt)
    if n_trials_model > 1:
        # Compute pairwise cosine similarities
        norms = np.linalg.norm(weights_arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = weights_arr / norms
        cos_matrix = normed @ normed.T

        # Count trials that have at least one other trial with cos > 0.95
        np.fill_diagonal(cos_matrix, 0)
        has_near_duplicate = np.any(cos_matrix > 0.95, axis=1)
        collapse_rate = float(np.mean(has_near_duplicate))

        # Find clusters
        visited = set()
        clusters = []
        for i in range(n_trials_model):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i+1, n_trials_model):
                if j not in visited and cos_matrix[i, j] > 0.95:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(len(cluster))
        n_collapsed_clusters = len(clusters)
    else:
        collapse_rate = 0.0
        n_collapsed_clusters = 0

    model_fingerprints[model] = {
        "n_trials": n_trials_model,
        "mean_behavioral_profile": mean_profile,
        "std_behavioral_profile": std_profile,
        "best_handled_concepts": best_concepts,
        "worst_handled_concepts": worst_concepts,
        "weight_distribution": weight_stats,
        "collapse_rate": collapse_rate,
        "n_collapsed_clusters": n_collapsed_clusters,
    }
    print(f"  {model}: {n_trials_model} trials, collapse_rate={collapse_rate:.3f}, "
          f"best={best_concepts[0]['concept'] if best_concepts else 'N/A'}")

# ======================================================================
# SECTION 4: CONCEPT BEHAVIORAL PROFILES
# ======================================================================
print("\n=== Section 4: Concept Behavioral Profiles ===")

concept_profiles = {}
for concept in all_concepts:
    cts = concept_trials.get(concept, [])
    if not cts:
        continue

    # Mean +/- std of all key metrics
    concept_metrics = {k: [] for k in all_metric_keys}
    for t in cts:
        am = get_all_metrics(t)
        for k in all_metric_keys:
            concept_metrics[k].append(am[k])

    mean_metrics = {k: float(np.mean(concept_metrics[k])) for k in all_metric_keys}
    std_metrics = {k: float(np.std(concept_metrics[k])) for k in all_metric_keys}

    # Behavioral signature: metrics >0.5 std from grand mean (using all metrics)
    distinctive = []
    for k in all_metric_keys:
        if grand_std[k] > 1e-10:
            z_score = (mean_metrics[k] - grand_mean[k]) / grand_std[k]
            if abs(z_score) > 0.5:
                distinctive.append({
                    "metric": k,
                    "z_score": float(z_score),
                    "direction": "high" if z_score > 0 else "low"
                })
    distinctive.sort(key=lambda x: -abs(x["z_score"]))

    # Best exemplar trial
    # For speed concepts: highest speed; for rotation: highest yaw; for rhythmic: highest PLV
    # General: pick the one closest to the concept mean vector in key metrics
    speed_concepts = ["sprint", "dash", "gallop", "scurry", "charge"]
    rotation_concepts = ["twirl", "pivot", "circle", "turn_left", "turn_right"]
    oscillation_concepts = ["rock", "sway", "wobble"]

    if concept in speed_concepts:
        best_idx = int(np.argmax([get_metrics(t)["mean_speed"] for t in cts]))
        best_criterion = "highest_speed"
    elif concept in rotation_concepts:
        best_idx = int(np.argmax([abs(get_metrics(t)["yaw_net_rad"]) for t in cts]))
        best_criterion = "highest_abs_yaw"
    elif concept in oscillation_concepts:
        best_idx = int(np.argmax([get_metrics(t)["phase_lock_score"] for t in cts]))
        best_criterion = "highest_phase_lock"
    else:
        # Closest to concept mean vector
        mean_vec = np.array([mean_metrics[k] for k in KEY_METRICS])
        dists = [np.linalg.norm(metrics_to_vec(get_metrics(t)) - mean_vec) for t in cts]
        best_idx = int(np.argmin(dists))
        best_criterion = "closest_to_concept_mean"

    best_trial = cts[best_idx]
    best_exemplar = {
        "criterion": best_criterion,
        "model": best_trial["model"],
        "language": best_trial["language"],
        "weights": best_trial["weights"],
        "key_metrics": get_metrics(best_trial),
    }

    # Cross-model agreement
    model_vecs = {}
    for model in all_models:
        mts = concept_model_trials.get((concept, model), [])
        if mts:
            vecs = [metrics_to_vec(get_metrics(t)) for t in mts]
            model_vecs[model] = np.mean(vecs, axis=0)

    if len(model_vecs) >= 2:
        model_names = sorted(model_vecs.keys())
        cross_model_cosines = []
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                cs = cosine_similarity(model_vecs[model_names[i]], model_vecs[model_names[j]])
                cross_model_cosines.append(cs)
        cross_model_agreement = float(np.mean(cross_model_cosines))
    else:
        cross_model_agreement = None

    concept_profiles[concept] = {
        "n_trials": len(cts),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "distinctive_signature": distinctive,
        "best_exemplar": best_exemplar,
        "cross_model_agreement": cross_model_agreement,
    }
    sig_str = ", ".join([f"{d['metric']}({d['direction']})" for d in distinctive[:3]])
    agr_str = f"{cross_model_agreement:.3f}" if cross_model_agreement is not None else "N/A"
    print(f"  {concept}: n={len(cts)}, agreement={agr_str}, signature=[{sig_str}]")

# ======================================================================
# SECTION 5: WEIGHT-SPACE GEOMETRY
# ======================================================================
print("\n=== Section 5: Weight-Space Geometry ===")

# Collect all weight vectors
all_weights = np.array([get_weights(t) for t in trials])
n_total = all_weights.shape[0]
print(f"  Weight matrix: {all_weights.shape}")

# PCA of 6D -> 2D
weights_centered = all_weights - np.mean(all_weights, axis=0)
cov = np.cov(weights_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

total_var = np.sum(eigenvalues)
variance_explained = eigenvalues / total_var if total_var > 0 else eigenvalues
cumulative_variance = np.cumsum(variance_explained)

# Project to 2D
pc_2d = weights_centered @ eigenvectors[:, :2]

pca_result = {
    "eigenvalues": eigenvalues.tolist(),
    "variance_explained_ratio": variance_explained.tolist(),
    "cumulative_variance_explained": cumulative_variance.tolist(),
    "pc1_loadings": eigenvectors[:, 0].tolist(),
    "pc2_loadings": eigenvectors[:, 1].tolist(),
    "loading_labels": weight_names,
    "pc1_range": [float(pc_2d[:, 0].min()), float(pc_2d[:, 0].max())],
    "pc2_range": [float(pc_2d[:, 1].min()), float(pc_2d[:, 1].max())],
}

print(f"  PCA variance explained: PC1={variance_explained[0]:.3f}, "
      f"PC2={variance_explained[1]:.3f}, cumulative 2D={cumulative_variance[1]:.3f}")

# Weight-space attractors (cosine sim > 0.9 clustering)
norms = np.linalg.norm(all_weights, axis=1, keepdims=True)
norms = np.where(norms < 1e-10, 1e-10, norms)
normed_weights = all_weights / norms

# Simple greedy clustering
ATTRACTOR_THRESHOLD = 0.9
visited = set()
attractors = []

for i in range(n_total):
    if i in visited:
        continue
    cluster = [i]
    visited.add(i)
    for j in range(i+1, n_total):
        if j in visited:
            continue
        cs = float(np.dot(normed_weights[i], normed_weights[j]))
        if cs > ATTRACTOR_THRESHOLD:
            cluster.append(j)
            visited.add(j)
    if len(cluster) >= 3:
        centroid = np.mean(all_weights[cluster], axis=0)
        # What concepts are in this cluster?
        cluster_concepts = [trials[k]["concept"] for k in cluster]
        concept_counts = {}
        for c in cluster_concepts:
            concept_counts[c] = concept_counts.get(c, 0) + 1
        attractors.append({
            "size": len(cluster),
            "centroid": centroid.tolist(),
            "concept_counts": dict(sorted(concept_counts.items(), key=lambda x: -x[1])),
            "top_concepts": sorted(concept_counts.items(), key=lambda x: -x[1])[:5],
        })

attractors.sort(key=lambda x: -x["size"])
print(f"  Found {len(attractors)} attractors (cosine > {ATTRACTOR_THRESHOLD}, size >= 3)")
for att in attractors[:5]:
    top = att["top_concepts"]
    top_str = ", ".join([f"{c}({n})" for c, n in top[:3]])
    print(f"    size={att['size']}: {top_str}")

# Correlation between weight positions and behavioral outcomes
weight_behavior_correlations = {}
for i, wn in enumerate(weight_names):
    w_col = all_weights[:, i]
    corrs = {}
    for k in KEY_METRICS:
        vals = np.array([get_metrics(t)[k] for t in trials])
        # Pearson correlation
        if np.std(w_col) > 1e-10 and np.std(vals) > 1e-10:
            r = np.corrcoef(w_col, vals)[0, 1]
            corrs[k] = float(r)
        else:
            corrs[k] = 0.0
    weight_behavior_correlations[wn] = corrs

# Which weight carries most behavioral info? (mean abs correlation)
weight_info_scores = {}
for wn in weight_names:
    mean_abs_r = np.mean([abs(v) for v in weight_behavior_correlations[wn].values()])
    weight_info_scores[wn] = float(mean_abs_r)

weight_info_ranked = sorted(weight_info_scores.items(), key=lambda x: -x[1])
print(f"  Weight info ranking: {[(w, f'{s:.3f}') for w, s in weight_info_ranked]}")

weight_space_geometry = {
    "pca": pca_result,
    "n_attractors": len(attractors),
    "attractors": attractors[:20],  # top 20
    "weight_behavior_correlations": weight_behavior_correlations,
    "weight_information_ranking": [{"weight": w, "mean_abs_correlation": s} for w, s in weight_info_ranked],
    "description": "PCA of 700 6D weight vectors; attractor analysis via greedy cosine-similarity "
                   "clustering (threshold 0.9); Pearson correlations between each weight position "
                   "and key behavioral metrics."
}

# ======================================================================
# SECTION 6: ARCHETYPE CANDIDATES
# ======================================================================
print("\n=== Section 6: Archetype Candidates ===")

# Identify archetypes from the data by behavioral clustering
# Strategy: use the behavioral vectors (KEY_METRICS) and cluster

all_beh_vecs = np.array([list(get_metrics(t).values()) for t in trials])

# Normalize behavioral vectors
beh_mean = np.mean(all_beh_vecs, axis=0)
beh_std = np.std(all_beh_vecs, axis=0)
beh_std = np.where(beh_std < 1e-10, 1, beh_std)
beh_normed = (all_beh_vecs - beh_mean) / beh_std

# K-means-like clustering (manual implementation, no scipy/sklearn)
def kmeans_manual(X, k, max_iter=100, n_init=10):
    """Simple k-means with multiple random initializations."""
    best_labels = None
    best_inertia = float('inf')
    n = X.shape[0]

    for init in range(n_init):
        # Random initialization (k-means++ style: pick first random, then weighted)
        rng = np.random.RandomState(42 + init)
        centers = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array([min(np.sum((X[i] - c)**2) for c in centers) for i in range(n)])
            probs = dists / (dists.sum() + 1e-10)
            idx = rng.choice(n, p=probs)
            centers.append(X[idx])
        centers = np.array(centers)

        for _ in range(max_iter):
            # Assign
            dists = np.array([[np.sum((X[i] - centers[j])**2) for j in range(k)] for i in range(n)])
            labels = np.argmin(dists, axis=1)

            # Update
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = centers[j]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = sum(np.sum((X[i] - centers[labels[i]])**2) for i in range(n))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia

# Try k=6 archetypes (data-driven)
K = 6
labels, beh_centers, inertia = kmeans_manual(beh_normed, K, max_iter=50, n_init=10)
print(f"  K-means (k={K}): inertia={inertia:.1f}")

# Also try k=4..8 and pick best silhouette-like score
def pseudo_silhouette(X, labels, k):
    """Simple silhouette-like score without scipy."""
    n = X.shape[0]
    scores = []
    for i in range(min(n, 500)):  # subsample for speed
        ci = labels[i]
        same = X[labels == ci]
        if len(same) <= 1:
            scores.append(0)
            continue
        a = np.mean([np.sqrt(np.sum((X[i] - same[j])**2)) for j in range(len(same)) if not np.array_equal(same[j], X[i])])
        b_vals = []
        for cj in range(k):
            if cj == ci:
                continue
            other = X[labels == cj]
            if len(other) == 0:
                continue
            b_vals.append(np.mean([np.sqrt(np.sum((X[i] - other[j])**2)) for j in range(len(other))]))
        if not b_vals:
            scores.append(0)
            continue
        b = min(b_vals)
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))

# Test different k values
k_scores = {}
for k in [4, 5, 6, 7, 8]:
    lbl, ctr, iner = kmeans_manual(beh_normed, k, max_iter=50, n_init=5)
    s = pseudo_silhouette(beh_normed, lbl, k)
    k_scores[k] = {"silhouette": s, "inertia": float(iner)}
    print(f"    k={k}: silhouette={s:.3f}, inertia={iner:.1f}")

# Use k=6 clusters to define archetypes
# Map each cluster back to its properties
archetype_candidates = []
archetype_names_templates = [
    "strider", "spinner", "loafer", "wobbler", "dasher", "crawler",
    "marcher", "drifter"
]

for ci in range(K):
    mask = labels == ci
    cluster_indices = np.where(mask)[0]
    cluster_trials = [trials[i] for i in cluster_indices]

    if not cluster_trials:
        continue

    # Centroid in original weight space
    cluster_weights = np.array([get_weights(t) for t in cluster_trials])
    weight_centroid = np.mean(cluster_weights, axis=0)

    # Behavioral profile
    cluster_beh = {k: [] for k in KEY_METRICS}
    for t in cluster_trials:
        m = get_metrics(t)
        for k in KEY_METRICS:
            cluster_beh[k].append(m[k])

    beh_profile = {k: {"mean": float(np.mean(cluster_beh[k])), "std": float(np.std(cluster_beh[k]))}
                   for k in KEY_METRICS}

    # Distinguishing metrics (z-score from grand mean)
    grounding_criteria = []
    for k in KEY_METRICS:
        if grand_std[k] > 1e-10:
            z = (beh_profile[k]["mean"] - grand_mean[k]) / grand_std[k]
            if abs(z) > 0.5:
                grounding_criteria.append({"metric": k, "z_score": float(z),
                                           "direction": "high" if z > 0 else "low"})
    grounding_criteria.sort(key=lambda x: -abs(x["z_score"]))

    # Source concepts
    concept_counts = {}
    for t in cluster_trials:
        c = t["concept"]
        concept_counts[c] = concept_counts.get(c, 0) + 1
    top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])

    # Cross-language support
    language_counts = {}
    for t in cluster_trials:
        l = t["language"]
        language_counts[l] = language_counts.get(l, 0) + 1
    n_languages = len(language_counts)

    # Auto-name based on behavioral signature using a priority cascade
    mean_speed = beh_profile["mean_speed"]["mean"]
    mean_dx = beh_profile["dx"]["mean"]
    mean_yaw = beh_profile["yaw_net_rad"]["mean"]
    mean_plv = beh_profile["phase_lock_score"]["mean"]
    mean_entropy = beh_profile["contact_entropy_bits"]["mean"]
    mean_straightness = beh_profile["path_straightness"]["mean"]

    # z-scores for each dimension
    z_speed = (mean_speed - grand_mean["mean_speed"]) / max(grand_std["mean_speed"], 1e-10)
    z_dx = (mean_dx - grand_mean["dx"]) / max(grand_std["dx"], 1e-10)
    z_yaw = (abs(mean_yaw) - abs(grand_mean["yaw_net_rad"])) / max(grand_std["yaw_net_rad"], 1e-10)
    z_plv = (mean_plv - grand_mean["phase_lock_score"]) / max(grand_std["phase_lock_score"], 1e-10)
    z_entropy = (mean_entropy - grand_mean["contact_entropy_bits"]) / max(grand_std["contact_entropy_bits"], 1e-10)
    z_straight = (mean_straightness - grand_mean["path_straightness"]) / max(grand_std["path_straightness"], 1e-10)

    # Name by most distinctive behavioral feature
    name_candidates = [
        ("spinner", abs(z_yaw), abs(mean_yaw) > 0.3),
        ("dasher", z_speed, z_speed > 0.5),
        ("strider", z_dx, z_dx > 0.5 and z_straight > 0),
        ("marcher", z_plv, z_plv > 0.3 and z_speed > 0),
        ("wobbler", z_entropy, z_entropy > 0.5),
        ("drifter", -z_straight, z_straight < -0.3 and abs(z_speed) < 0.5),
        ("crawler", -z_speed, z_speed < -0.5 and z_dx < 0),
        ("loafer", -z_speed, z_speed < -0.3 and abs(z_dx) < 0.3),
        ("pacer", z_plv, z_plv > 0 and z_speed < 0),
    ]
    name_candidates.sort(key=lambda x: -x[1])

    name = None
    existing = [a["name"] for a in archetype_candidates]
    for candidate_name, score, condition in name_candidates:
        if condition and candidate_name not in existing:
            name = candidate_name
            break
    if name is None:
        # Fallback: use the highest-scoring name with a qualifier
        for candidate_name, score, condition in name_candidates:
            qualified = f"{candidate_name}_alt"
            if qualified not in existing:
                name = qualified
                break
        if name is None:
            name = f"archetype_{ci}"

    desc_parts = []
    for gc in grounding_criteria[:3]:
        desc_parts.append(f"{gc['direction']} {gc['metric']}")
    description = f"Behavioral archetype with {', '.join(desc_parts) if desc_parts else 'average profile'}"

    archetype_candidates.append({
        "name": name,
        "cluster_id": int(ci),
        "description": description,
        "n_trials": len(cluster_trials),
        "representative_weight_vector": {wn: float(weight_centroid[i]) for i, wn in enumerate(weight_names)},
        "behavioral_profile": beh_profile,
        "grounding_criteria": grounding_criteria,
        "source_concepts": [{"concept": c, "count": n} for c, n in top_concepts[:8]],
        "cross_language_support": {
            "n_languages": n_languages,
            "language_counts": language_counts,
        },
    })

    top_c_str = ", ".join([f"{c}({n})" for c, n in top_concepts[:3]])
    print(f"  Archetype '{name}': {len(cluster_trials)} trials, {n_languages} langs, "
          f"top concepts: {top_c_str}")

archetype_section = {
    "method": "K-means clustering on normalized behavioral vectors (6 key metrics), "
              "k selected from [4-8] by pseudo-silhouette score.",
    "k_selection": k_scores,
    "archetypes": archetype_candidates,
}

# ======================================================================
# SECTION 7: SUMMARY STATISTICS
# ======================================================================
print("\n=== Section 7: Summary Statistics ===")

total_trials = len(all_results)
successful = len(trials)
failed = total_trials - successful
success_rate = successful / total_trials if total_trials > 0 else 0.0

per_model_counts = {}
for model in all_models:
    model_total = sum(1 for r in all_results if r["model"] == model)
    model_success = len(model_trials.get(model, []))
    per_model_counts[model] = {
        "total": model_total,
        "successful": model_success,
        "success_rate": model_success / model_total if model_total > 0 else 0.0,
    }

# Top findings
findings = []

# Finding 1: semantic coherence
if comparisons.get("mean_speed"):
    d_speed = comparisons["mean_speed"]["cohens_d"]
    findings.append(f"Fast vs slow concepts show Cohen's d = {d_speed:.3f} on mean_speed "
                   f"(fast mean: {comparisons['mean_speed']['fast_mean']:.4f}, "
                   f"slow mean: {comparisons['mean_speed']['slow_mean']:.4f})")

# Finding 2: functor faithfulness
findings.append(f"Overall functor faithfulness (cross-language behavioral consistency): "
               f"{functor_faithfulness:.4f} mean cosine similarity across "
               f"{len(all_cosines)} language-pair comparisons")

# Finding 3: most/least invariant
if ranked_concepts:
    findings.append(f"Most translation-invariant concept: '{ranked_concepts[0][0]}' "
                   f"(cosine {ranked_concepts[0][1]:.4f})")
    findings.append(f"Least translation-invariant concept: '{ranked_concepts[-1][0]}' "
                   f"(cosine {ranked_concepts[-1][1]:.4f})")

# Finding 4: PCA
findings.append(f"PCA of weight space: PC1 explains {variance_explained[0]:.1%}, "
               f"PC2 explains {variance_explained[1]:.1%} of variance "
               f"(cumulative 2D: {cumulative_variance[1]:.1%})")

# Finding 5: attractors
findings.append(f"Found {len(attractors)} weight-space attractors (cosine > 0.9, size >= 3). "
               f"Largest attractor has {attractors[0]['size'] if attractors else 0} trials")

# Finding 6: model collapse
collapse_rates = {m: mf["collapse_rate"] for m, mf in model_fingerprints.items()}
max_collapse_model = max(collapse_rates, key=collapse_rates.get) if collapse_rates else None
findings.append(f"Highest model collapse rate: {max_collapse_model} "
               f"({collapse_rates.get(max_collapse_model, 0):.1%})")

# Finding 7: weight information
findings.append(f"Most behaviorally informative weight: {weight_info_ranked[0][0]} "
               f"(mean |r| = {weight_info_ranked[0][1]:.3f})")

# Finding 8: effect sizes for fast/slow
for metric, comp in comparisons.items():
    d = comp["cohens_d"]
    label = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
    findings.append(f"Fast vs slow on {metric}: Cohen's d = {d:.3f} ({label} effect)")

summary_statistics = {
    "total_trials": total_trials,
    "successful_trials": successful,
    "failed_trials": failed,
    "success_rate": success_rate,
    "n_concepts": len(all_concepts),
    "n_languages": len(all_languages),
    "n_models": len(all_models),
    "concepts": all_concepts,
    "languages": all_languages,
    "models": all_models,
    "per_model_counts": per_model_counts,
    "grand_means": grand_mean,
    "grand_stds": grand_std,
    "top_findings": findings,
}

for f in findings:
    print(f"  - {f}")

# ======================================================================
# ASSEMBLE AND WRITE OUTPUT
# ======================================================================
print("\n=== Writing output ===")

output = {
    "metadata": {
        "analysis_date": "2026-02-21",
        "data_sources": [
            "motion_seed_experiment_v2.json (560 trials, 4 Ollama models)",
            "motion_seed_experiment_v2.singlemodel.20260221_160301.json (140 trials, 1 LM Studio model)",
        ],
        "total_trials_analyzed": len(trials),
        "experiment": "Movement vocabulary experiment: 28 concepts x 5 languages x 5 models",
        "key_metrics_used": KEY_METRICS,
        "all_metrics_tracked": all_metric_keys,
    },
    "semantic_coherence": semantic_coherence,
    "cross_language_functor": cross_language_functor,
    "model_fingerprints": model_fingerprints,
    "concept_behavioral_profiles": concept_profiles,
    "weight_space_geometry": weight_space_geometry,
    "archetype_candidates": archetype_section,
    "summary_statistics": summary_statistics,
}

out_path = os.path.join(BASE, "movement_vocabulary_analysis.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)

print(f"\nWrote: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")
print("Done.")
