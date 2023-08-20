from math import prod

import cv2


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
        and len(o := set(ordering)) == len(ordering)
        and o == set(range(tiles))
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

    References:
    - https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

    Relevent:
    - https://realpython.com/image-processing-with-the-python-pillow-library
    """

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if not valid_input(img.shape[:2], tile_size, ordering):
        raise ValueError("The tile size or ordering are not valid for the given image")

    img_height, img_width, channels = img.shape
    tile_height, tile_width = tile_size

    rows = img_height // tile_height
    cols = img_width // tile_width
    tiles = img.reshape(
        rows,
        tile_height,
        cols,
        tile_width,
        channels,
    ).swapaxes(1, 2)

    arranged_img = (
        tiles.reshape(rows * cols, tile_height, tile_width, channels)[ordering]
        .reshape(
            rows,
            cols,
            tile_height,
            tile_width,
            channels,
        )
        .swapaxes(1, 2)
        .reshape(img.shape)
    )

    cv2.imwrite(out_path, arranged_img)


if __name__ == "__main__":
    test_cases = ((((4, 4), (2, 2), [0, 1, 2, 4]), False),)
    for args, expected in test_cases:
        assert valid_input(*args) == expected
