import math

def example_fn(name, radius):
	area = math.pi * radius ** 2
	return "The area of {} is {}".format(name, area)

print(example_fn("Sricharan", 5))
