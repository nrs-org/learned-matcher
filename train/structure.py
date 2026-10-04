"""Structure head inference: for track pairs, P(sibling) = neither side is
derived from the other (two versions of one song). Trained by
train_relation.py (structure.txt, 3 classes: left derived / right derived /
sibling, swap-augmented); evaluated symmetrically."""

import numpy as np
import lightgbm as lgb

from features import DATA

_cache = {}


def _model(name):
    if name not in _cache:
        path = DATA / "learned-matcher/models" / name / "structure.txt"
        _cache[name] = lgb.Booster(model_file=str(path)) if path.exists() else None
    return _cache[name]


def structure_probs(F, types, name="rel-v6"):
    if not name:
        return None
    m = _model(name)
    if m is None:
        return None
    cols = m.feature_name()
    X = F.reindex(columns=cols)
    signed = [c for c in cols if c.startswith("s_")]
    Xs = X.copy()
    Xs[signed] = -Xs[signed]
    p1 = m.predict(X)
    p2 = m.predict(Xs)
    p_sib = (p1[:, 2] + p2[:, 2]) / 2
    return [float(s) if t == "track" else None for s, t in zip(p_sib, np.asarray(types))]
