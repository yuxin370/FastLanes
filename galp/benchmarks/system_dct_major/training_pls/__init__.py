"""Physical-load-segment crop/shuffle model-effect experiment support.

The formal path is the frozen-layout, four-condition, 300-epoch runner exposed
by :mod:`training_pls.train`.  The older schedule exports below remain available
for bounded diagnostics and compatibility; they are not the formal result path.
"""

from .schedule import (
    CROP_POLICIES,
    ORDER_POLICIES,
    ORGANIZATIONS,
    PlsScheduleConfig,
    SampleRecord,
    build_schedule,
    load_training_samples,
)

__all__ = [
    "CROP_POLICIES",
    "ORDER_POLICIES",
    "ORGANIZATIONS",
    "PlsScheduleConfig",
    "SampleRecord",
    "build_schedule",
    "load_training_samples",
]
