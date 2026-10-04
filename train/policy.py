"""Class layout and the verdict policy shared by training, pool scoring and
gold prediction. Models trained before the sibling class (v1-v9) have 3
output columns; later ones have 4."""

CLASSES = ["same", "related", "sibling", "unrelated"]
CLASSES_V1 = ["same", "related", "unrelated"]


def class_names(n_cols):
    return CLASSES if n_cols == 4 else CLASSES_V1


def merge_guard(features):
    """Pairs that must never auto-merge whatever the score: a placeholder
    title ("Private video") or a generic artist name ("Release - Topic",
    "Various Artists") on either side. They DEFER instead."""
    g = None
    for col in ("placeholder", "generic_name"):
        if col in features:
            m = features[col].fillna(0).to_numpy() > 0
            g = m if g is None else (g | m)
    return g


def verdicts(types, P, thresholds, guard=None, structure=None):
    """MERGE above the per-type same threshold; else the strongest of
    related / sibling / unrelated, with DEFER when same still leads.
    SIBLING means "two versions of one song": no direct edge (each version
    links to the common original instead)."""
    names = class_names(P.shape[1])
    out = []
    for t, row in zip(types, P):
        p = dict(zip(names, row))
        if p["same"] >= thresholds.get(t, 1.0):
            out.append("DEFER" if guard is not None and guard[len(out)] else "MERGE")
            continue
        rest = {k: v for k, v in p.items() if k != "same"}
        best = max(rest, key=rest.get)
        if p["same"] > rest[best]:
            out.append("DEFER")
        else:
            v = {"related": "RELATE", "sibling": "SIBLING", "unrelated": "DISTINCT"}[best]
            # structure head (track RELATE only): P(neither side derives from the other)
            if v == "RELATE" and structure is not None and structure[len(out)] is not None and structure[len(out)] > 0.5:
                v = "SIBLING"
            out.append(v)
    return out
