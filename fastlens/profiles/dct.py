"""CNN output semantics for the existing source512 Direct-DCT paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator


@dataclass(frozen=True, slots=True)
class DctModelProfile:
    """Projected float32 NCHW output from source512 JPEG coefficients.

    Channels are ordered ``(component, natural_frequency_index)`` pairs;
    components 0/1/2 denote Y/Cb/Cr. Normalization contains one
    ``(subtract, divide)`` pair per channel, applied after online geometry,
    round-to-even and int16 saturation. Omit it for unnormalized output.

    Source frequencies use storage-column indices, independently of output
    channels. ``"all"`` preserves the existing CNN reference; a tuple masks
    omitted source frequencies before resize and can change model inputs.
    Projected PLS training currently supports only ``"all"``.
    """

    output_grid_size: int
    output_channels: tuple[tuple[int, int], ...]
    normalization: tuple[tuple[float, float], ...] | None = None
    source_frequency_policy: str | tuple[int, ...] = "all"

    def __post_init__(self) -> None:
        grid = operator.index(self.output_grid_size)
        if isinstance(self.output_grid_size, bool) or grid <= 0:
            raise ValueError("output_grid_size must be a positive integer")
        channels = tuple((operator.index(c), operator.index(f)) for c, f in self.output_channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("output_channels must be nonempty and unique")
        if any(c not in (0, 1, 2) or not 0 <= f < 64 for c, f in channels):
            raise ValueError("output channels require component 0..2 and natural frequency 0..63")
        normalization = (
            ((0.0, 1.0),) * len(channels)
            if self.normalization is None
            else tuple((float(subtract), float(divide)) for subtract, divide in self.normalization)
        )
        if len(normalization) != len(channels) or any(
            not math.isfinite(subtract) or not math.isfinite(divide) or divide == 0
            for subtract, divide in normalization
        ):
            raise ValueError("normalization requires one finite (subtract, nonzero divide) pair per channel")
        selection = self.source_frequency_policy
        if isinstance(selection, str):
            if selection != "all":
                raise ValueError("source_frequency_policy must be 'all' or a sequence of column indices")
        else:
            selection = tuple(operator.index(value) for value in selection)
            if not 1 <= len(selection) <= 64 or len(set(selection)) != len(selection) or any(
                not 0 <= value < 64 for value in selection
            ):
                raise ValueError("source frequencies must be 1..64 unique column indices in [0, 64)")
        object.__setattr__(self, "output_grid_size", grid)
        object.__setattr__(self, "output_channels", channels)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "source_frequency_policy", selection)

    def _native_channels(self) -> list[list[float]]:
        return [
            [component, frequency, subtract, divide]
            for (component, frequency), (subtract, divide) in zip(self.output_channels, self.normalization)
        ]
