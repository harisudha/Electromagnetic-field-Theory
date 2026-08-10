"""
Cartesian Coordinate Systems
==============================
Utilities for working with Cartesian (rectangular) coordinate systems
as used in Electromagnetic Field Theory.

A point in 3-D Cartesian space is represented as (x, y, z).
Unit vectors: ax (x̂), ay (ŷ), az (ẑ).
"""

import math


# ---------------------------------------------------------------------------
# Basic vector operations
# ---------------------------------------------------------------------------

def dot_product(a, b):
    """Return the scalar dot product of two 3-D Cartesian vectors.

    Parameters
    ----------
    a, b : sequence of three floats (x, y, z)

    Returns
    -------
    float
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross_product(a, b):
    """Return the cross product a × b of two 3-D Cartesian vectors.

    Parameters
    ----------
    a, b : sequence of three floats (x, y, z)

    Returns
    -------
    tuple of three floats
    """
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def magnitude(v):
    """Return the magnitude (length) of a 3-D Cartesian vector.

    Parameters
    ----------
    v : sequence of three floats (x, y, z)

    Returns
    -------
    float
    """
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def unit_vector(v):
    """Return the unit vector in the direction of v.

    Parameters
    ----------
    v : sequence of three floats (x, y, z)

    Returns
    -------
    tuple of three floats

    Raises
    ------
    ValueError if v is the zero vector.
    """
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot form a unit vector from the zero vector.")
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def add_vectors(a, b):
    """Return the vector sum a + b.

    Parameters
    ----------
    a, b : sequence of three floats (x, y, z)

    Returns
    -------
    tuple of three floats
    """
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtract_vectors(a, b):
    """Return the vector difference a - b.

    Parameters
    ----------
    a, b : sequence of three floats (x, y, z)

    Returns
    -------
    tuple of three floats
    """
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale_vector(scalar, v):
    """Return the scalar multiple scalar * v.

    Parameters
    ----------
    scalar : float
    v : sequence of three floats (x, y, z)

    Returns
    -------
    tuple of three floats
    """
    return (scalar * v[0], scalar * v[1], scalar * v[2])


# ---------------------------------------------------------------------------
# Distance and position
# ---------------------------------------------------------------------------

def distance(p1, p2):
    """Return the Euclidean distance between two Cartesian points.

    Parameters
    ----------
    p1, p2 : sequence of three floats (x, y, z)

    Returns
    -------
    float
    """
    return magnitude(subtract_vectors(p2, p1))


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def cartesian_to_cylindrical(x, y, z):
    """Convert Cartesian (x, y, z) to cylindrical (ρ, φ, z).

    Parameters
    ----------
    x, y, z : float

    Returns
    -------
    tuple (rho, phi_rad, z)
        rho  – radial distance from the z-axis (≥ 0)
        phi  – azimuthal angle in radians [0, 2π)
        z    – unchanged
    """
    rho = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x) % (2 * math.pi)
    return (rho, phi, z)


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (r, θ, φ).

    Parameters
    ----------
    x, y, z : float

    Returns
    -------
    tuple (r, theta_rad, phi_rad)
        r     – radial distance from the origin (≥ 0)
        theta – polar (inclination) angle in radians [0, π]
        phi   – azimuthal angle in radians [0, 2π)
    """
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.acos(z / r) if r != 0 else 0.0
    phi = math.atan2(y, x) % (2 * math.pi)
    return (r, theta, phi)


# ---------------------------------------------------------------------------
# Differential operators (scalar and vector fields on a grid / at a point)
# ---------------------------------------------------------------------------

def gradient(f, point, h=1e-5):
    """Numerically approximate the gradient ∇f at *point* using central differences.

    Parameters
    ----------
    f     : callable  f(x, y, z) -> float
    point : sequence of three floats (x, y, z)
    h     : float, step size for finite differences

    Returns
    -------
    tuple of three floats (∂f/∂x, ∂f/∂y, ∂f/∂z) at *point*
    """
    x, y, z = point
    dfdx = (f(x + h, y, z) - f(x - h, y, z)) / (2 * h)
    dfdy = (f(x, y + h, z) - f(x, y - h, z)) / (2 * h)
    dfdz = (f(x, y, z + h) - f(x, y, z - h)) / (2 * h)
    return (dfdx, dfdy, dfdz)


def divergence(Fx, Fy, Fz, point, h=1e-5):
    """Numerically approximate the divergence ∇·F at *point*.

    Parameters
    ----------
    Fx, Fy, Fz : callable  each  f(x, y, z) -> float  (components of F)
    point       : sequence of three floats (x, y, z)
    h           : float, step size

    Returns
    -------
    float  ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z  at *point*
    """
    x, y, z = point
    dFx = (Fx(x + h, y, z) - Fx(x - h, y, z)) / (2 * h)
    dFy = (Fy(x, y + h, z) - Fy(x, y - h, z)) / (2 * h)
    dFz = (Fz(x, y, z + h) - Fz(x, y, z - h)) / (2 * h)
    return dFx + dFy + dFz


def curl(Fx, Fy, Fz, point, h=1e-5):
    """Numerically approximate the curl ∇×F at *point*.

    Parameters
    ----------
    Fx, Fy, Fz : callable  each  f(x, y, z) -> float  (components of F)
    point       : sequence of three floats (x, y, z)
    h           : float, step size

    Returns
    -------
    tuple of three floats  (curlx, curly, curlz)  at *point*
    """
    x, y, z = point

    dFz_dy = (Fz(x, y + h, z) - Fz(x, y - h, z)) / (2 * h)
    dFy_dz = (Fy(x, y, z + h) - Fy(x, y, z - h)) / (2 * h)

    dFx_dz = (Fx(x, y, z + h) - Fx(x, y, z - h)) / (2 * h)
    dFz_dx = (Fz(x + h, y, z) - Fz(x - h, y, z)) / (2 * h)

    dFy_dx = (Fy(x + h, y, z) - Fy(x - h, y, z)) / (2 * h)
    dFx_dy = (Fx(x, y + h, z) - Fx(x, y - h, z)) / (2 * h)

    curlx = dFz_dy - dFy_dz
    curly = dFx_dz - dFz_dx
    curlz = dFy_dx - dFx_dy
    return (curlx, curly, curlz)


def laplacian(f, point, h=1e-5):
    """Numerically approximate the scalar Laplacian ∇²f at *point*.

    Parameters
    ----------
    f     : callable  f(x, y, z) -> float
    point : sequence of three floats (x, y, z)
    h     : float, step size

    Returns
    -------
    float  ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²  at *point*
    """
    x, y, z = point
    f0 = f(x, y, z)
    d2x = (f(x + h, y, z) - 2 * f0 + f(x - h, y, z)) / h ** 2
    d2y = (f(x, y + h, z) - 2 * f0 + f(x, y - h, z)) / h ** 2
    d2z = (f(x, y, z + h) - 2 * f0 + f(x, y, z - h)) / h ** 2
    return d2x + d2y + d2z


# ---------------------------------------------------------------------------
# Quick demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    a = (1, 2, 3)
    b = (4, 5, 6)

    print("Vector A:", a)
    print("Vector B:", b)
    print("A + B   :", add_vectors(a, b))
    print("A - B   :", subtract_vectors(a, b))
    print("A · B   :", dot_product(a, b))
    print("A × B   :", cross_product(a, b))
    print("|A|     :", magnitude(a))
    print("â       :", unit_vector(a))
    print("Distance:", distance(a, b))

    print("\nCartesian (1, 1, 1) → Cylindrical:", cartesian_to_cylindrical(1, 1, 1))
    print("Cartesian (1, 1, 1) → Spherical  :", cartesian_to_spherical(1, 1, 1))

    # gradient of f = x² + y² + z² at (1, 1, 1) → should be ≈ (2, 2, 2)
    f = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
    print("\n∇f at (1,1,1):", gradient(f, (1, 1, 1)))
