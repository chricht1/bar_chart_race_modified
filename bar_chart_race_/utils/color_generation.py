import colorsys
import numpy as np
import math

def _rgb_to_hex(r, g, b):
        return "#{:02X}{:02X}{:02X}".format(r, g, b)


def generate_large_palette(n,
                        sat_values=(0.9, 0.7, 0.55),
                        val_values=(0.95, 0.78),
                        hue_spacing_strategy='even'):
    """
    Generate n distinct colors by combining several saturation/value variations
    with a set of evenly spaced hues.
    - sat_values: tuple of saturation levels to use (0..1)
    - val_values: tuple of value/brightness levels to use (0..1)
    - hue_spacing_strategy: 'even' or 'golden' (golden reduces perceptual clustering)
    Returns list of hex strings length n.
    """
    variations = [(s, v) for v in val_values for s in sat_values]
    per_hue = len(variations)
    num_hues = math.ceil(n / per_hue)

    # generate hue list
    hues = []
    if hue_spacing_strategy == 'golden':
        golden = 0.618033988749895
        h = 0.0
        for i in range(num_hues):
            hues.append(h % 1.0)
            h += golden
    else:  # evenly spaced
        for i in range(num_hues):
            hues.append(i / float(num_hues))

    # interleave variations to avoid adjacent very-similar colors
    colors = []
    is_dark = []
    for i, h in enumerate(hues):
        # For variety, rotate variation order every hue
        for j in range(per_hue):
            s, v = variations[(j + i) % per_hue]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            colors.append(_rgb_to_hex(r, g, b))
            is_dark.append(np.sum(np.array([r, g, b]) * np.array([299, 587, 114]))/1000 < 123) 
            # estimate perceived brightness of color for 
            #https://stackoverflow.com/questions/49437263/contrast-between-label-and-background-determine-if-color-is-light-or-dark
            if len(colors) == n:
                perm = np.random.permutation(np.array(range(n)))
                colors = np.array(colors)[perm]
                is_dark = np.array(is_dark)[perm]
                return colors, is_dark