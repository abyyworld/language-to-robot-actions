import pytest

# Importing the package selects an offscreen GL backend suited to this machine
# before any MuJoCo object exists (logic kept from the ICM project).
import lang2act  # noqa: F401
from lang2act.sim.assets import fetch

requires_assets = pytest.mark.skipif(
    not fetch.is_available(),
    reason="Franka Panda assets not installed; run `make assets`",
)
