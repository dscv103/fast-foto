import numpy as np


def rgb_to_hsl(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    l = (maxc + minc) / 2.0

    delta = maxc - minc
    s = np.zeros_like(l)
    mask = delta != 0
    s[mask] = delta[mask] / (1 - np.abs(2 * l[mask] - 1))

    h = np.zeros_like(l)
    # Avoid division by zero by masking with delta
    nonzero = delta != 0
    r_eq = (maxc == r) & nonzero
    g_eq = (maxc == g) & nonzero
    b_eq = (maxc == b) & nonzero

    h[r_eq] = ((g[r_eq] - b[r_eq]) / delta[r_eq]) % 6
    h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 2
    h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 4

    h = h * 60.0
    h = np.mod(h, 360.0)

    return h, s, l


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    c = (1 - np.abs(2 * l - 1)) * s
    h_ = h / 60.0
    x = c * (1 - np.abs(np.mod(h_, 2) - 1))

    zeros = np.zeros_like(h)
    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    conds = [
        (0 <= h_) & (h_ < 1),
        (1 <= h_) & (h_ < 2),
        (2 <= h_) & (h_ < 3),
        (3 <= h_) & (h_ < 4),
        (4 <= h_) & (h_ < 5),
        (5 <= h_) & (h_ < 6),
    ]
    values = [
        (c, x, zeros),
        (x, c, zeros),
        (zeros, c, x),
        (zeros, x, c),
        (x, zeros, c),
        (c, zeros, x),
    ]

    r[:] = 0
    g[:] = 0
    b[:] = 0

    for cond, (rc, gc, bc) in zip(conds, values):
        r = np.where(cond, rc, r)
        g = np.where(cond, gc, g)
        b = np.where(cond, bc, b)

    m = l - c / 2.0
    r += m
    g += m
    b += m

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _sector_weight(hue: np.ndarray, center: float, width: float) -> np.ndarray:
    # shortest angular distance
    dist = np.abs(((hue - center + 180.0) % 360.0) - 180.0)
    weight = np.zeros_like(hue)
    inside = dist <= width
    weight[inside] = np.cos(np.pi * dist[inside] / (2 * width))
    return np.clip(weight, 0.0, 1.0)


def apply_hsl_sectors(rgb_srgb: np.ndarray, sectors: list) -> np.ndarray:
    h, s, l = rgb_to_hsl(rgb_srgb)

    for sector in sectors:
        weight = _sector_weight(h, sector["center"], sector["width"])

        h = np.mod(h + weight * sector.get("hue_shift", 0.0), 360.0)
        sat_scale = sector.get("sat_scale", 1.0)
        s = np.clip(s * (1.0 + (sat_scale - 1.0) * weight), 0.0, 1.0)
        l = np.clip(l + weight * sector.get("light_delta", 0.0), 0.0, 1.0)

    return hsl_to_rgb(h, s, l)
