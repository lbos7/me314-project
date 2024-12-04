import sympy as sym
import numpy as np
from helper_funcs import unhat4x4, SE3inv

t,g,r,d,m,M,J = sym.symbols('t,g,r,d,m,M,J')

x = sym.Function('x')(t)
y = sym.Function('y')(t)
theta = sym.Function('theta')(t)
X = sym.Function('X')(t)
Y = sym.Function('Y')(t)
psi = sym.Function('psi')(t)

# q = sym.Matrix([x, y, theta, X, Y, psi])
q = sym.Matrix([X, Y, psi])
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

I_box = sym.Matrix([[M, 0, 0, 0, 0, 0],
                    [0, M, 0, 0, 0, 0],
                    [0, 0, M, 0, 0, 0],
                    [0, 0, 0, J, 0, 0],
                    [0, 0, 0, 0, J, 0],
                    [0, 0, 0, 0, 0, J]])

Vb_box_hat = sym.simplify(SE3inv(g_wb) @ (g_wb.diff(t)))
Vb_box = unhat4x4(Vb_box_hat)

KE_box = sym.simplify(.5*(I_box @ Vb_box).dot(Vb_box))
V_box = sym.simplify(M*g*((g_wb @ sym.Matrix([0, 0, 0, 1])).dot(sym.Matrix([0, 1, 0, 0]))))

L = sym.simplify(KE_box - V_box)

