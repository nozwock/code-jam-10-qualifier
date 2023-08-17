from functools import reduce
from typing import Any, Iterable


def prod(seq: Iterable[Any]) -> Any:  # meh, bounds feels like a mess
    return reduce(lambda acc, i: acc * i, seq)


def valid_input(
    image_size: tuple[int, int], tile_size: tuple[int, int], ordering: list[int]
) -> bool:
    """
    Return True if the given input allows the rearrangement of the image, False otherwise.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once.
    """

    tiles, remainder = divmod(prod(image_size), prod(tile_size))

    return (
        remainder == 0
        and tiles == len(ordering)
        and len(set(ordering)) == len(ordering)
    ) or False


def rearrange_tiles(
    image_path: str, tile_size: tuple[int, int], ordering: list[int], out_path: str
) -> None:
    """
    Rearrange the image.

    The image is given in `image_path`. Split it into tiles of size `tile_size`, and rearrange them by `ordering`.
    The new image needs to be saved under `out_path`.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once. If these conditions do not hold, raise a ValueError with the message:
    "The tile size or ordering are not valid for the given image".
    """

    ...
