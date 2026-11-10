"""Operator decisions for a manipulation approval request."""

from dataclasses import dataclass
from typing import Optional, Tuple

Pixel = Tuple[int, int]


@dataclass
class ApprovalDecision:
    """What the operator asked for in response to a set of candidate images.

    Exactly one of these holds:
      * approved: grasp at `pixel` in candidate `image_index`.
      * walk_to: do not grasp yet; walk toward the ground/object point `pixel`
        in candidate `image_index`, then capture new images and ask again.
        `stand_pixel`, when given, is the ground pixel (same image) the robot
        should stand on, facing the target. Otherwise the robot stops a
        configured standoff short of the target, facing it.
      * neither: abandon the grasp attempt.
    """

    approved: bool
    pixel: Optional[Pixel]
    image_index: int
    walk_to: bool = False
    stand_pixel: Optional[Pixel] = None

    @classmethod
    def approve(cls, pixel: Pixel, image_index: int) -> "ApprovalDecision":
        return cls(approved=True, pixel=pixel, image_index=image_index)

    @classmethod
    def reject(cls) -> "ApprovalDecision":
        return cls(approved=False, pixel=None, image_index=0)

    @classmethod
    def walk(
        cls, pixel: Pixel, image_index: int, stand_pixel: Optional[Pixel] = None
    ) -> "ApprovalDecision":
        return cls(
            approved=False,
            pixel=pixel,
            image_index=image_index,
            walk_to=True,
            stand_pixel=stand_pixel,
        )

    @property
    def rejected(self) -> bool:
        return not self.approved and not self.walk_to
