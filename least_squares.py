import numpy as np
def back_sub(U,y):
  x = np.linalg.inv(U).dot(y)
  return x

def fwd_sub_q(Q, b):
    y = np.transpose(Q).dot(b)
    return y
def norm(col_a):
  """returns the norm of the column of A"""
  return np.sqrt(np.sum(np.square(col_a)))

def householder_QR(A):
    # in orderb to avoid errors the array needs to be a float
    A = A.astype('float64')
    # defining  shape

    # number rows
    m = A.shape[0]
    # number columns
    n = A.shape[1]

    R = np.copy(A)
    Q = np.identity(m)
    # loop over the columns
    for i in range(n):
        # vector v equal to the ith column of R
        v = np.copy(R[:, i])

        # update with zeros elemnet above the iter to
        # lower the dimension of the vector e1
        q = 0
        while q < i:
            v[q] = 0
            q = q + 1
        # as we change only the firs elemnt of the vector e1 is possible to
        # change the first elemt as follow
        v[i] = v[i] + np.sign(v[q]) * norm(v)
        v = (np.array([v]))
        H = np.identity(m) - 2 * (np.transpose(v) * v) / (norm(v)) ** 2
        R = np.dot(H, R)
        Q = np.dot(H, Q)
    Q = np.linalg.inv(Q)
    # reduce dimensions of Q and R as required
    # Q has n columns and R is an nXn matrix
    Q_reduced = np.zeros((m, n))
    R_reduced = np.zeros((n, n))
    for i in range(n):
        Q_reduced[:, i] = Q[:, i]
        R_reduced[i] = R[i]
    return Q_reduced, R_reduced, R, Q


def fwd_sub_q(Q, b):
    y = np.transpose(Q).dot(b)
    return y


# using fwd and back substitution
def least_squares(M, b):
    Q_reduced, R_reduced, R, Q = householder_QR(M)

    y = fwd_sub_q(Q_reduced, b)

    return back_sub(R_reduced, y)