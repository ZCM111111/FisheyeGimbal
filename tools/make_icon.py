"""Generates the app icon.

Design: the whole frame is a warped mesh — the raw fisheye picture — and the
single straight accent bar across the middle is the horizon the app locks level.
No circle around it, which also avoids the glyph reading as a prohibition sign.
"""

from PIL import Image, ImageDraw, ImageFilter

S = 1024
CX = CY = S / 2
ACCENT = (255, 214, 10)
MESH = (96, 108, 124)

img = Image.new("RGB", (S, S), (9, 10, 12))
draw = ImageDraw.Draw(img)

# Vertical gradient, lifted at the top so the icon does not read flat.
for y in range(S):
    t = y / (S - 1)
    draw.line(
        [(0, y), (S, y)],
        fill=(
            int(31 + (9 - 31) * t),
            int(35 + (10 - 35) * t),
            int(42 + (12 - 42) * t),
        ),
    )

# Warped mesh: 5 x 5, bowing outwards towards the edges like a fisheye.
mesh = Image.new("L", (S, S), 0)
md = ImageDraw.Draw(mesh)
steps = 140
span = 0.5 * S * 1.02
bulge = 0.30
for index in range(-2, 3):
    if index == 0:
        continue
    base = index / 3.0 * span
    pts = []
    for k in range(steps + 1):
        t = -1 + 2 * k / steps
        pts.append((CX + base * (1 + bulge * t * t), CY + t * span))
    md.line(pts, fill=255, width=13, joint="curve")
    pts = []
    for k in range(steps + 1):
        t = -1 + 2 * k / steps
        pts.append((CX + t * span, CY + base * (1 + bulge * t * t)))
    md.line(pts, fill=255, width=13, joint="curve")

mesh = mesh.filter(ImageFilter.GaussianBlur(1.6))

# Fade the mesh towards the corners so the centre stays legible at small sizes.
fade = Image.new("L", (S, S), 0)
fd = ImageDraw.Draw(fade)
rings = 60
for i in range(rings, 0, -1):
    k = i / rings
    radius = S * 0.80 * k
    value = int(255 * (1 - k) ** 1.15)
    fd.ellipse([CX - radius, CY - radius, CX + radius, CY + radius], fill=value)
mesh = Image.composite(mesh, Image.new("L", (S, S), 0), fade)
img.paste(Image.new("RGB", (S, S), MESH), (0, 0), mesh.point(lambda v: int(v * 0.78)))

# Vignette.
vignette = Image.new("L", (S, S), 0)
vd = ImageDraw.Draw(vignette)
for i in range(60, 0, -1):
    k = i / 60
    vd.ellipse(
        [CX - S * 0.80 * k, CY - S * 0.80 * k, CX + S * 0.80 * k, CY + S * 0.80 * k],
        fill=int(255 * (1 - k) ** 0.5),
    )
img.paste(Image.new("RGB", (S, S), (5, 6, 7)), (0, 0), vignette)

# The locked horizon: the one straight line in the picture.
BAR_HALF = 372
BAR = 32
draw.rounded_rectangle(
    [CX - BAR_HALF, CY - BAR, CX + BAR_HALF, CY + BAR],
    radius=BAR,
    fill=ACCENT,
)

img.save(r"C:\Users\93543\Desktop\SteadyFisheye\icon-1024.png")
print("written", img.size)
