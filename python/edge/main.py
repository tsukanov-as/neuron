import os
from PIL import Image, ImageDraw
from functools import reduce

def NOT(x):
	return 1 - x

def AND(x, y):
	return x * y

def OR(x, y):
	return (x + y) - (x * y)

d = os.path.abspath(os.path.dirname(__file__))
image = Image.open(os.path.join(d, 'Lenna.jpg'))

draw = ImageDraw.Draw(image)
width = image.size[0]
height = image.size[1]
pix = image.load()

# kernel 2x2:
# 1 0
# 1 0
def detect(i, j):
	ch = 1
	a = pix[i+0, j+0][ch] / 255
	b = pix[i+1, j+0][ch] / 255
	c = pix[i+0, j+1][ch] / 255
	d = pix[i+1, j+1][ch] / 255
	return reduce(AND, [a, NOT(b),
						c, NOT(d)])

for i in range(width-1):
	for j in range(height-1):
		x = detect(i, j)
		draw.point((i, j), (int(x*255), int(x*255), int(x*255)))

image = image.crop((0, 0, width-1, height-1))

image.show()
