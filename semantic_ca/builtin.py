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
