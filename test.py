import copy
import variable

a = variable.Variable(2)
b = variable.Variable(3)
c = variable.Variable(4)
d = variable.Variable(5)

print(a == a)
print(a == b)
print(a == 2)
print(2 == a)
print(a + b)
print(a * b)
print(a - b)
print(a / b)
print(a ** b)
print(a + 1)
print(a * 2)
print(a - 3)
print(a / 4)
print(a ** 5)
print(1 + a)
print(2 * a)
print(3 - a)
print(4 / a)
print(5 ** a)
print(abs(+-a))

# h1 = a + b
# h2 = c + d
# out = h1 + h2

# h1 = a + a
# h2 = h1 + a
# h3 = h2 + h1
# h4 = h3 + h2

# h1 = 1 + a
# h2 = h1 + a
# h3 = h2 + a
# h4 = h3 + a

# h4.backward_pass()

# print(f'h4 gd1: {h4.grad}, should be {0.0}')
# print(f'h3 gd1: {h3.grad}, should be {1.0}')
# print(f'h2 gd1: {h2.grad}, should be {2.0}')
# print(f'h1 gd1: {h1.grad}, should be {3.0}')
# print(f'a gd1: {a.grad}, should be {8.0}')

# h1 = a - b
# h2 = c - d
# out = h1 * h2

# out.backward_pass()

# h1 = a * a
# h1.backward_pass()
