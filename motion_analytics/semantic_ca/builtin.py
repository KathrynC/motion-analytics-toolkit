"""Built‑in configurations and helper functions."""

from .lattice import extract_beer_features, cosine_similarity

# Default Beer analytics metrics used for similarity
BEER_METRICS = [
    'displacement', 'speed_mean', 'efficiency',
    'contact_entropy', 'duty_cycle_back', 'duty_cycle_front',
    'phase_lock', 'freq_back', 'freq_front',
    'roll_dominance', 'pitch_dominance', 'yaw_dominance', 'axis_switching_rate'
]

# A default similarity metric: cosine on Beer feature vectors
DEFAULT_SIMILARITY = lambda a, b: cosine_similarity(a, b)

# Lakoff feature layer annotation for Beer analytics features.
# Grounded = directly observable sensorimotor features.
# Linking = cross-domain abstractions requiring interpretation.
# Indices correspond to extract_beer_features() output order.
BEER_FEATURE_LAYERS = {
    'grounded': [0, 1, 2, 3, 4, 5, 9, 10, 11, 12],
    # 0-2: outcome (displacement, speed, efficiency)
    # 3-5: contact (entropy, duty_cycle_back, duty_cycle_front)
    # 9-12: rotation axis (roll/pitch/yaw dominance, axis_switching_rate)
    'linking': [6, 7, 8],
    # 6-8: coordination (phase_lock, freq_back, freq_front)
}
