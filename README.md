# Interior Point Method Quadratic Programming Solver with CUDA Support

Quadratic programming solver. Solves problems with the following form:

$$ \min \quad c^T x + \frac{1}{2} x^T Q x $$ $$ \text{s.t.} \quad A x = b,   \quad x \geq 0$$

Where $c \in \mathbb{R}^{n}$, $b \in \mathbb{R}^{m}$, $Q \in \mathbf{S}^{n}_{+}$, and $A\in \mathbb{R}^{m \times n}$ and has full row rank. It can be used as the following:

```python
# Problem data
m = 30
n = 40

Q = np.random.randn(n, n)
Q = Q.T @ Q

c = np.random.randn(n)

# Two constraints. Variables sum up to 1, and x[0] is equal to 0.1.
a1 = np.ones(n)
a2 = np.zeros(n)
a2[0] = 1

A = np.vstack((a1, a2))

b = np.array([1, 0.1])

# Generate an instance of the solver
qp = QP(c, Q, A, b, verbose=True)

x, f = qp.solve()

print("Decision variables:", x)
print("Objective function value:", f)
```

## Progress:
    [DONE] Implement solver for CPU
    [DONE] Handle exceptions
    [DONE] Output formatting
    [DONE] README.md
    [x] Add CUDA support

## References:
Implementation is based on the work by Jacek Gondzio: [Interior Point Methods 25 Years Later](https://www.pure.ed.ac.uk/ws/portalfiles/portal/10662023/Interior_point_methods_25_years_later.pdf)