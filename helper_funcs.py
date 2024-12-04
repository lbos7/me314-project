import sympy as sym

def unhat4x4(mat):
  return sym.Matrix([mat[3], mat[7], mat[11], mat[9], mat[2], mat[4]])

def SE3inv(mat):
  R = sym.Matrix([[mat[0], mat[1], mat[2]],
                  [mat[4], mat[5], mat[6]],
                  [mat[8], mat[9], mat[10]]])
  p = sym.Matrix([mat[3], mat[7], mat[11]])
  R_trans = R.T
  p_trans = -R_trans @ p
  return sym.Matrix([[R_trans[0], R_trans[1], R_trans[2], p_trans[0]],
                     [R_trans[3], R_trans[4], R_trans[5], p_trans[1]],
                     [R_trans[6], R_trans[7], R_trans[8], p_trans[2]],
                     [0, 0, 0, 1]])