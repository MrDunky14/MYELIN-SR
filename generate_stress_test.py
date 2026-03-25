from PIL import Image, ImageDraw, ImageFont
import math

width, height = 1280, 720
img = Image.new('RGB', (width, height), color=(20, 20, 20))
draw = ImageDraw.Draw(img)

# DLSS Failure 1: Thin high-contrast sub-pixel lines against flat background (Power-lines)
for i in range(0, 15):
    y = 50 + i * 15
    # Drawing lines that are exactly 1 pixel thick, slightly slanted
    draw.line((100, y, 600, y + i*2), fill=(255, 255, 255), width=1)

# DLSS Failure 2: Moire / Fine grid patterns (Chainlink Fences)
# Drawing a very dense overlapping grid where pixels inevitably alias
for x in range(700, 1100, 4):
    for y in range(50, 300, 4):
        draw.point((x, y), fill=(200, 200, 200))
        draw.point((x+1, y+1), fill=(100, 100, 100))

# DLSS Failure 3: High-frequency checkerboards (Checkerboard rendering break)
for x in range(100, 400, 2):
    for y in range(400, 600, 2):
        if (x//2 + y//2) % 2 == 0:
            draw.point((x, y), fill=(255, 255, 255))
            draw.point((x+1, y), fill=(255, 255, 255))
            draw.point((x, y+1), fill=(255, 255, 255))
            draw.point((x+1, y+1), fill=(255, 255, 255))

# Radiating sub-pixel lines (Menger sponge extreme edge testing)
cx, cy = 900, 500
for angle in range(0, 360, 5):
    rad = math.radians(angle)
    x2 = cx + math.cos(rad) * 150
    y2 = cy + math.sin(rad) * 150
    draw.line((cx, cy, x2, y2), fill=(0, 255, 0), width=1)

img.save('data/stress_test_720p.png')
print("Generated DLSS Stress Test at data/stress_test_720p.png")
