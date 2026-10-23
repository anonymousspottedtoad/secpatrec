# The .env has to be read before anything else in this package is imported.
# `secpat_interaction_order.config` snapshots its environment at import time (JOERN_SERVICE_URL
# and the rest), and importing it is the first thing `.engine` does — so a
# load_dotenv() in cli.py ran *after* the values it was meant to supply had
# already been read as empty, and stage 3 reported the query service unset on
# every run configured through .env.
from dotenv import load_dotenv

load_dotenv()

from .engine import EngineConfig, PatternRecoveryEngine  # noqa: E402

__all__ = ["PatternRecoveryEngine", "EngineConfig"]
