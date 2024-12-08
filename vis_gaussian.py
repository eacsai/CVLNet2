from jaxtyping import Float, Shaped
from dataclasses import dataclass
from torch import Tensor
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, Optional, Union
import torch
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from string import ascii_letters, digits, punctuation
import numpy as np
from torchvision import transforms
from gaussian.decoder import DecoderOutput
from gaussian.diagonal_gaussian_distribution import DiagonalGaussianDistribution
from gaussian.latent_splat import render_cuda_orthographic, RenderOutput
# from gaussian.nopo_cuda_splatting import render_cuda_orthographic
to_pil_image = transforms.ToPILImage()


Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]
EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    opacities: Float[Tensor, "batch gaussian"]
    color_harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    feature_harmonics: Float[Tensor, "batch gaussian channels d_feature_sh"] | None = None 

def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)

def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result

def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]

def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "*#batch 3"],
    maxima: Float[Tensor, "*#batch 3"],
    margin: float = 0.1,
) -> tuple[
    Float[Tensor, "*batch 3"],  # minima of the scene
    Float[Tensor, "*batch 3"],  # maxima of the scene
]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima

def cat(
    main_axis: Axis,
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def vcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")

def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align="left",
        gap=4,
    )

def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results

def render_projections(
    gaussians: Gaussians,
    resolution: tuple[int, int],
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
) -> Float[Tensor, "batch 3 3 height width"]:
    device = gaussians.means.device
    b, _, _ = gaussians.means[:1].shape
    
    # Compute the minima and maxima of the scene.
    minima = gaussians.means[:1].min(dim=1).values
    maxima = gaussians.means[:1].max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )

    look = ["x", "y", "z"]
    # for look_axis in range(3):
    look_axis = 1
    right_axis = (look_axis + 1) % 3
    down_axis = (look_axis + 2) % 3

    # Define the extrinsics for rendering.
    extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
    extrinsics[:, right_axis, 0] = 1
    extrinsics[:, down_axis, 1] = 1
    extrinsics[:, look_axis, 2] = 1
    # extrinsics[:, right_axis, 3] = 0.5 * (
    #     scene_minima[:, right_axis] + scene_maxima[:, right_axis]
    # )
    # extrinsics[:, down_axis, 3] = 0.5 * (
    #     scene_minima[:, down_axis] + scene_maxima[:, down_axis]
    # )

    extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
    extrinsics[:, 3, 3] = 1

    # Define the intrinsics for rendering.
    extents = scene_maxima - scene_minima
    far = extents[:, look_axis]
    near = torch.zeros_like(far)
    # width = extents[:, right_axis]
    # height = extents[:, down_axis]
    width = torch.tensor(resolution[0] * 0.2, dtype=torch.float32, device=device)
    height = torch.tensor(resolution[1] * 0.2, dtype=torch.float32, device=device)
    # extrinsics[:, right_axis, 3] = 0
    # extrinsics[:, down_axis, 3] = 0

    projection: RenderOutput = render_cuda_orthographic(
        extrinsics,
        width,
        height,
        near,
        far,
        resolution,
        torch.zeros((b, 3), dtype=torch.float32, device=device),
        gaussians.means[:1],
        gaussians.covariances[:1],
        gaussians.opacities[:1],
        gaussians.color_harmonics[:1],
        gaussians.feature_harmonics[:1],
        fov_degrees=1.0,
    )
    color = projection.color
    if draw_label:
        right_axis_name = "XYZ"[right_axis]
        down_axis_name = "XYZ"[down_axis]
        label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"
        color = torch.stack([add_label(x, label) for x in color])
    out = render_to_decoder_output(projection, b)
    return out

def render_to_decoder_output(
    render_output: RenderOutput,
    b: int,
) -> DecoderOutput:
    if render_output.feature is not None:
        features = render_output.feature
        # NOTE background feature = 0 = mean = logvar (of normal distribution)
        mean, logvar = (features, (1-rearrange(render_output.mask.detach(), "b h w -> b () h w", b=b)).log().expand_as(features))
        feature_posterior = DiagonalGaussianDistribution(mean, logvar)
    else:
        feature_posterior = None
    return DecoderOutput(
        color=render_output.color if render_output.color is not None else None,
        feature_posterior=feature_posterior,
        mask=render_output.mask,
        depth=render_output.depth
    )