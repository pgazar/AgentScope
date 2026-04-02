GREEN  = "#22863a"
ORANGE = "#d97706"
RED    = "#dc2626"
GRAY   = "#888780"

THRESHOLDS = {
    "ir":        {"green": 0.70, "orange": 0.40},
    "agent":     {"green": 0.80, "orange": 0.60},
    "geval":     {"green": 0.80, "orange": 0.60},
    "cost_usd":  {"green": 0.020, "orange": 0.050, "invert": True},
    "latency_s": {"green": 2.0,   "orange": 5.0,   "invert": True},
    "hallu":     {"green": 0.10,  "orange": 0.25,  "invert": True},
}


def color_for_metric(value: float, metric_type: str) -> str:
    if value is None:
        return GRAY
    cfg = THRESHOLDS.get(metric_type, THRESHOLDS["geval"])
    if cfg.get("invert"):
        return GREEN if value <= cfg["green"] else ORANGE if value <= cfg["orange"] else RED
    return GREEN if value >= cfg["green"] else ORANGE if value >= cfg["orange"] else RED
