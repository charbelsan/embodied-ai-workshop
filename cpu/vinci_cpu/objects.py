"""Workstation pieces and the work-order rules (same as the GPU workshop)."""
from dataclasses import dataclass

TOP_DRAWER, KITTING_TRAY, KEEP = "top_drawer", "kitting_tray", "keep"

RED, BLUE, YELLOW = (0.85, 0.10, 0.10), (0.10, 0.25, 0.85), (0.95, 0.80, 0.10)


@dataclass(frozen=True)
class Piece:
    id: str
    shape: str  # cube | bar | cylinder
    size: tuple  # cube/bar: (x, y, z) ; cylinder: (diameter, diameter, height)
    color: tuple
    rule: str  # TOP_DRAWER | KITTING_TRAY | KEEP (distractor)
    mass: float = 0.05

    @property
    def height(self) -> float:
        return self.size[2]

    @property
    def radius(self) -> float:  # footprint radius for non-overlapping placement
        return 0.5 * (self.size[0] ** 2 + self.size[1] ** 2) ** 0.5


PIECES = {
    "bolt_box": Piece("bolt_box", "cube", (0.045, 0.045, 0.045), RED, TOP_DRAWER),
    "sensor": Piece("sensor", "cube", (0.040, 0.040, 0.040), RED, TOP_DRAWER, mass=0.04),
    "bracket": Piece("bracket", "bar", (0.08, 0.03, 0.03), BLUE, KITTING_TRAY, mass=0.06),
    "spacer": Piece("spacer", "cylinder", (0.04, 0.04, 0.04), BLUE, KITTING_TRAY, mass=0.04),
    "gauge": Piece("gauge", "cube", (0.05, 0.05, 0.05), YELLOW, KEEP, mass=0.15),
}

WORK_ORDER = ("Work order: RED parts -> top drawer; BLUE parts -> kitting tray; "
              "the YELLOW gauge is a measuring instrument: do not move it. "
              "Final state: everything put away, drawer closed, nothing on the floor.")
