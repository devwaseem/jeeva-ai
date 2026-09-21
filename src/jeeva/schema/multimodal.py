"""
MultimodalContext — typed container for image+text inputs to Jeeva's decision engine.

Usage::

    from jeeva import MultimodalContext

    ctx = MultimodalContext(
        text="Classify this invoice.",
        images=["invoice.png"],          # local file paths or PIL Images
    )
    result = engine.choose(ctx, "Classify the document.", ("Invoice", "Receipt", "Contract"))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jeeva.core.prompts import validate_context_budget

if TYPE_CHECKING:
    # PIL is optional; import for static type checkers so types are preserved.
    from PIL.Image import Image as PILImage
else:
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        PILImage = Any

# Type alias accepted in the `images` list
ImageInput = str | Path | PILImage


@dataclass
class MultimodalContext:
    """
    A typed container that bundles a text prompt with one or more images (and,
    in future, audio) for use as the ``context`` argument of any Jeeva API.

    Parameters
    ----------
    text:
        The textual part of the context (required, non-empty).
    images:
        Zero or more images, each supplied as a local file-system path
        (``str`` or ``pathlib.Path``) or an already-loaded
        ``PIL.Image.Image`` object. Remote URLs are rejected.
    audio:
        Reserved for future audio support.  Passing any value currently raises
        ``NotImplementedError``.

    Raises
    ------
    ValueError
        If ``text`` is empty or ``images`` contains an item that is neither a
        local path-like value nor a PIL Image.
    NotImplementedError
        If ``audio`` is non-empty (not yet supported).
    """

    text: str
    images: list[ImageInput] = field(default_factory=list)
    audio: list[Any] = field(default_factory=list)
    max_image_size: tuple[int, int] | int | None = None
    max_images: int = 16
    max_decoded_bytes_per_image: int = 64 * 1024 * 1024
    max_total_decoded_bytes: int = 128 * 1024 * 1024

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                "MultimodalContext.text must be a non-empty string. "
                "Provide the textual portion of your context."
            )

        validate_context_budget(self.text)
        if self.audio:
            raise NotImplementedError(
                "Audio inputs are not yet supported in MultimodalContext. "
                "This field is reserved for a future release."
            )
        if self.max_images <= 0:
            raise ValueError("max_images must be positive.")
        if len(self.images) > self.max_images:
            raise ValueError(f"At most {self.max_images} images are allowed per context.")
        if self.max_decoded_bytes_per_image <= 0:
            raise ValueError("max_decoded_bytes_per_image must be positive.")
        if self.max_total_decoded_bytes <= 0:
            raise ValueError("max_total_decoded_bytes must be positive.")
        if self.max_image_size is not None:
            dimensions = (
                (self.max_image_size, self.max_image_size)
                if isinstance(self.max_image_size, int)
                else self.max_image_size
            )
            if len(dimensions) != 2 or any(size <= 0 for size in dimensions):
                raise ValueError("max_image_size dimensions must be positive.")

        for idx, img in enumerate(self.images):
            if not _is_valid_image_input(img):
                raise ValueError(
                    f"images[{idx}] must be a local file path (str/Path) "
                    f"or a PIL.Image.Image. Got: {type(img).__name__!r}."
                )
            if isinstance(img, str) and img.lower().startswith(("http://", "https://")):
                raise ValueError(
                    "Remote image URLs are not supported. Fetch and validate remote "
                    "content in the application, then pass a local path or PIL image."
                )

    # ------------------------------------------------------------------
    # Image loading helpers
    # ------------------------------------------------------------------

    def load_pil_images(self) -> list[PILImage]:
        """
        Resolves all entries in ``self.images`` to ``PIL.Image.Image`` objects.

        Requires ``Pillow`` to be installed::

            pip install "jeeva-ai[multimodal]"

        Returns
        -------
        list[PIL.Image.Image]
            One PIL Image per entry in ``self.images``, in the same order.

        Raises
        ------
        ImportError
            If ``Pillow`` is not installed.
        FileNotFoundError
            If a local path-based image cannot be found on disk.
        """
        try:
            from PIL import Image as PILImageModule
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to load images in MultimodalContext.\n"
                'Install it with:  pip install "jeeva-ai[multimodal]"\n'
                "or:               pip install Pillow"
            ) from exc

        loaded: list[PILImage] = []
        total_decoded_bytes = 0
        for img in self.images:
            if _is_pil_image(img):
                pil_obj: Any = img
            elif isinstance(img, (str, Path, os.PathLike)):
                path = Path(img)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path.resolve()}")
                pil_obj = PILImageModule.open(path)
            else:  # pragma: no cover - constructor validation makes this unreachable
                continue

            width, height = pil_obj.size
            decoded_bytes = width * height * 3
            if width <= 0 or height <= 0:
                raise ValueError("Images must have positive dimensions.")
            if decoded_bytes > self.max_decoded_bytes_per_image:
                raise ValueError(
                    f"Decoded image requires {decoded_bytes} bytes, exceeding the "
                    f"{self.max_decoded_bytes_per_image} byte per-image limit."
                )
            total_decoded_bytes += decoded_bytes
            if total_decoded_bytes > self.max_total_decoded_bytes:
                raise ValueError(
                    "Decoded images exceed the aggregate "
                    f"{self.max_total_decoded_bytes} byte limit."
                )

            pil_obj.load()
            if pil_obj.mode != "RGB":
                pil_obj = pil_obj.convert("RGB")
            if self.max_image_size is not None:
                max_wh = (
                    (self.max_image_size, self.max_image_size)
                    if isinstance(self.max_image_size, int)
                    else self.max_image_size
                )
                pil_obj.thumbnail(max_wh)
            loaded.append(pil_obj)
        return loaded

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_images = len(self.images)
        text_preview = self.text[:60] + "…" if len(self.text) > 60 else self.text
        return f"MultimodalContext(text={text_preview!r}, images=[{n_images} image(s)])"

    def has_images(self) -> bool:
        """Returns True if at least one image is attached."""
        return len(self.images) > 0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _is_pil_image(obj: Any) -> bool:
    """Checks whether *obj* is a PIL Image without hard-importing Pillow."""
    cls = type(obj)
    return cls.__module__.startswith("PIL") and cls.__name__ == "Image"


def _is_valid_image_input(obj: Any) -> bool:
    """Accepts local path-like values and PIL Images."""
    if isinstance(obj, (str, Path, os.PathLike)):
        return True
    if _is_pil_image(obj):
        return True
    return False
