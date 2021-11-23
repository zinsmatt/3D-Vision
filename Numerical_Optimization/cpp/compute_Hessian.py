from sympy import Symbol, exp, diff, simplify
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
x = Symbol('x')
y = Symbol('y')

F = (y - exp(a*x*x+b*x+c))**2/2
print(F)
print(diff(F, a))
print(diff(F, b))
print(diff(F, c))

print("\n")

print(diff(F, a, a))
print(diff(F, b, b))
print(diff(F, c, c))

print("\n")

print(diff(F, a, b))
print(diff(F, a, c))

print("\n")

print(diff(F, b, a))
print(diff(F, b, c))

print("\n")

print(diff(F, c, a))
print(diff(F, c, b))


