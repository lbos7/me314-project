import sympy as sym
import numpy as np
from helper_funcs import unhat4x4, SE3inv

t,r,d,m,M,J = sym.symbols('t,r,d,m,M,J')

x = sym.Function('x')(t)
y = sym.Function('y')(t)
theta = sym.Function('theta')(t)
X = sym.Function('X')(t)
Y = sym.Function('Y')(t)
psi = sym.Function('psi')(t)

q = sym.Matrix([x, y, theta, X, Y, psi])
qdot = q.diff(t)
qddot = qdot.diff(t)

g_wa = sym.Matrix([[1, 0, 0, X],
                   [0, 1, 0, Y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

g_ab = sym.Matrix([[sym.cos(psi), -sym.sin(psi), 0, 0],
                   [sym.sin(psi), sym.cos(psi), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

g_bc = sym.Matrix([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

g_cd = sym.Matrix([[sym.cos(theta), -sym.sin(theta), 0, 0],
                   [sym.sin(theta), sym.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

g_wb = g_wa @ g_ab
g_wd = g_wa @ g_ab @ g_bc @ g_cd


