"""MetaUAS model: EfficientNet encoder + SMP UNet decoder + soft alignment. Lazy-imports to avoid pulling SMP at init."""

from utils.meta_utils import (
    apply_ad_scoremap,
    normalize,
    read_image_as_tensor,
    safely_load_state_dict,
    set_random_seed,
)

__all__ = [
    "AlignmentLayer",
    "AlignmentModule",
    "MetaUAS",
    "set_random_seed",
    "normalize",
    "apply_ad_scoremap",
    "read_image_as_tensor",
    "safely_load_state_dict",
]


def __getattr__(name: str):
    if name in ("AlignmentLayer", "AlignmentModule"):
        from . import alignment as _alignment
        return getattr(_alignment, name)
    if name == "MetaUAS":
        from .Metauas import MetaUAS
        return MetaUAS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
