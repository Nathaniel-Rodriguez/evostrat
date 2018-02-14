def elli(x):
    """ellipsoid-like test cost function"""
    n = len(x)
    aratio = 1e3
    return sum(x[i] ** 2 * aratio ** (2. * i / (n - 1)) for i in range(n))


def sphere(x):
    """sphere-like, ``sum(x**2)``, test cost function"""
    return sum(x[i] ** 2 for i in range(len(x)))


def rosenbrock(x):
    """Rosenbrock-like test cost function"""
    n = len(x)
    if n < 2:
        raise ValueError('dimension must be greater one')
    return sum(100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
               for i in range(n - 1))