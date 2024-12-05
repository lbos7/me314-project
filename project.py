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

q = sym.Matrix([x, y, theta, X, Y, psi])
qdot = q.diff(t)
qddot = qdot.diff(t)

q_jack = sym.Matrix([x, y, theta])
qdot_jack = q_jack.diff(t)
qddot_jack = qdot_jack.diff(t)

q_box = sym.Matrix([X, Y, psi])
qdot_box = q_box.diff(t)
qddot_box = qdot_box.diff(t)

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

L_box = sym.simplify(KE_box - V_box)

KE_jack = sym.simplify(((g_wd @ sym.Matrix([r, 0, 0, 1])).diff(t)).dot((g_wd @ sym.Matrix([r, 0, 0, 1])).diff(t)) + \
          ((g_wd @ sym.Matrix([0, r, 0, 1])).diff(t)).dot((g_wd @ sym.Matrix([0, r, 0, 1])).diff(t)) + \
          ((g_wd @ sym.Matrix([-r, 0, 0, 1])).diff(t)).dot((g_wd @ sym.Matrix([-r, 0, 0, 1])).diff(t)) + \
          ((g_wd @ sym.Matrix([0, -r, 0, 1])).diff(t)).dot((g_wd @ sym.Matrix([0, -r, 0, 1])).diff(t)))

V_jack = sym.simplify(m*g*(g_wd @ sym.Matrix([r, 0, 0, 1])).dot(sym.Matrix([0, 1, 0, 0])) + \
         m*g*(g_wd @ sym.Matrix([0, r, 0, 1])).dot(sym.Matrix([0, 1, 0, 0])) + \
         m*g*(g_wd @ sym.Matrix([-r, 0, 0, 1])).dot(sym.Matrix([0, 1, 0, 0])) + \
         m*g*(g_wd @ sym.Matrix([0, -r, 0, 1])).dot(sym.Matrix([0, 1, 0, 0])))

L_jack = sym.simplify(KE_jack - V_jack)

L = sym.simplify(L_box + L_jack)

eqn_mat_lhs = sym.simplify((L.diff(qdot)).diff(t) - L.diff(q))
eqn_mat = sym.Eq(eqn_mat_lhs, sym.Matrix([0, 0, 0, 0, 0, 0]))

soln = sym.solve(eqn_mat, qddot, dict=True)[0]