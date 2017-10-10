from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable


# x = torch.rand(5, 3)
# y = torch.rand(5, 3)

# help(x)

# result = torch.add(x, y)
# numpy_result = torch.ones(5)
# print(result)
# print(numpy_result)

# a = np.ones(10)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(b)


x = Variable(torch.ones(4, 4), requires_grad=True)
# print(x)

y = x + 4
# Since y is created as an operation, it would have grad_fn
# print(y.mean(), y.grad_fn)


z = y * y + 3
out = z.mean()
out.backward()
# print(x.grad)



x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
	y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
