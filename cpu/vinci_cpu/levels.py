"""Challenge levels: TRAINING and DEV, the same public parameters as the GPU workshop (seeds 0-7 rank the DEV run)."""
LEVELS = {
    "training": {
        "pieces": ["bolt_box"],
        "table_region": {"x": [0.22, 0.28], "y": [0.47, 0.53]},  # the old stand area, +/- 3 cm
        "yaw_deg": 20,
        "drawer_ajar": {"p": 0.0, "range": [0.0, 0.0]},
        "drawer_damping": [2.5, 2.5],
        "friction": [1.2, 1.2],
        "mass_scale": [1.0, 1.0],
        "min_gap": 0.03,
        "perturbations": {},
        "rules": {},
        "sim_time_budget_s": 90,
    },
    "dev": {
        "pieces": ["bolt_box", "sensor", "bracket", "gauge"],
        # >= 12 cm from the open drawer's front-left corner (x 0.40, y 0.38): the 20 cm wide Franka hand lifting a piece
        # at drawer height (0.66-0.78 m) must not sweep it
        "table_region": {"x": [0.12, 0.32], "y": [0.50, 0.68]},
        "yaw_deg": 45,
        "drawer_ajar": {"p": 0.0, "range": [0.0, 0.0]},
        "drawer_damping": [1.0, 6.0],
        "friction": [0.8, 1.2],
        "mass_scale": [0.6, 2.0],
        "min_gap": 0.03,
        # moderate, VISIBLE failures so that verification / recovery has something to fix (DAgger lesson)
        "sticky_drawer": {"p": 0.10, "damping": 12.0},
        "perturbations": {"transport_slip": {"p": 0.10, "open_steps": 18}},
        "rules": {},
        "sim_time_budget_s": 240,
    },
}


def load_level(name: str) -> dict:
    if name not in LEVELS:
        raise ValueError(f"unknown level {name!r}: training | dev")
    return {"name": name, **LEVELS[name]}
