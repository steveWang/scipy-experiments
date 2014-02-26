import itertools
import numpy, scipy, scipy.sparse, scipy.linalg

"""

These three functions are actually generators that allow you to run one
iteration at a time. Usage:

gen = jacobi(A, b)

i, x = gen.next()
assert i == 1

i, x = gen.next()
assert i == 2
...

"""

def jacobi(A, b):
    assert len(A) == len(b)
    indices = range(len(A))
    x = numpy.zeros(len(A))
    for k in itertools.count():
        old = numpy.copy(x)
        for j in indices:
            x[j] = (b[j] - sum([A.item(j, i) * old[i] for i in indices if i is not j])) / A.item(j, j)
        yield k, x

def sor(w):
    # First function call parameterizes SOR and returns a new function.
    def parameterized_sor(A, b):
        assert len(A) == len(b)
        indices = range(len(A))
        x = numpy.zeros(len(A))
        for k in itertools.count():
            for j in indices:
                x[j] = (1-w) * x[j] + w * (b[j] - sum([A.item(j, i) * x[i] for i in indices if i is not j])) / A.item(j, j)
            yield k, x
    return parameterized_sor

# Gauss Seidel is a special case of SOR. This is me being lazy, but oh well.
gauss_seidel = sor(1)

def tol(fn, A, b, tolerance, actual, absolute=True):
    last = numpy.zeros(len(A))
    for (k, x) in fn(A, b):
        error = numpy.linalg.norm(actual - x) if absolute else numpy.linalg.norm(x - last) / numpy.linalg.norm(x)
        if error < tolerance:
            return k, x
        last = numpy.copy(x)

n = 5
diag = [4] * n
off_diag = [-1] * n
A = scipy.sparse.spdiags([diag, off_diag, off_diag], [0, -1, 1], n, n).todense()
b = numpy.ones(n).T
actual = scipy.linalg.solve(A, b)

def prob3a(abstol):
    for (name, fn) in [("Jacobi", jacobi), ("Gauss Seidel", gauss_seidel), ("SOR", sor(1.1))]:
        i, soln = tol(fn, A, b, abstol, actual)
        print "%s converged to %s^T in %d iterations." % (name, soln.T.tolist(), i)

def prob3b(reltol):
    for (name, fn) in [("Jacobi", jacobi), ("Gauss Seidel", gauss_seidel), ("SOR", sor(1.1))]:
        i, soln = tol(fn, A, b, reltol, actual, False)
        print "%s converged in %d iterations." % (name, i)

def prob3c(num):
    def error(w):
        gen = sor(w)(A,b)
        for i in range(num):
            j, x = gen.next()
        return numpy.linalg.norm(actual - x) / numpy.linalg.norm(x)

    w = 1
    dx = 100
    while dx > 10 ** -20:
        e = error(w)
        left_e = error(w - dx)
        right_e = error(w + dx)
        if left_e < e and right_e > left_e:
            w = w - dx
        elif right_e < e:
            w = w + dx
        else:
            dx /= 2.
    return w
