import numpy as np

from .arithmetic import skew


# Cached for efficiency...
_eye3 = np.eye(3)


def exp(v):
    """
    Compute a rotation of ||v|| radians about the axis v.
    This equals the matrix exponential of the skew-symmetric matrix of v.
    This is also known as the mapping from the Lie algebra so(3) to the Lie group SO(3).
    """
    v = np.asarray(v)
    assert np.shape(v) == (3,), 'exp() received shape %s' % str(v.shape)

    vsq = np.dot(v, v)
    if vsq < 1e-8:
        # Taylor expansion of sin(sqrt(v))/sqrt(v):
        #   http://www.wolframalpha.com/input/?i=sin(sqrt(v))/sqrt(v)
        a = 1. - vsq/6. + vsq*vsq/120.

        # Taylor expansion of (1 - cos(sqrt(v))/v:
        #   http://www.wolframalpha.com/input/?i=(1-cos(sqrt(v)))/v
        b = .5 - vsq/24. + vsq*vsq/720.
    else:
        t = np.sqrt(vsq)
        a = np.sin(t)/t
        b = (1. - np.cos(t)) / vsq

    sk = skew(v)
    return _eye3 + a*sk + b*np.dot(sk, sk)


def exp_jacobian(v):
    """
    Compute the jacobian of exp(v) w.r.t. v. This function return a 3x3 matrix R such that
    for infinitesimal 3-vectors dx, exp(R * dx) * exp(v) = exp(v + dx)
    """
    vsq = np.dot(v, v)
    if vsq < 1e-8:
        # Taylor expansion:
        # http://www.wolframalpha.com/input/?i=2+*+sin%28x%2F2%29+*+sin%28x%2F2%29+%2F+%28x*v%29
        a = .5 - vsq/24 + vsq*vsq/720
        # Taylor expansion:
        # http://www.wolframalpha.com/input/?i=%28x-sin%28x%29%29%2Fx%5E3
        b = 1./6. - vsq/120 + vsq*vsq/5040
    else:
        t = np.sqrt(vsq)
        a = -2. * np.sin(t/2.) * np.sin(t/2.) / vsq
        b = (t - np.sin(t)) / (t*t*t)

    sk = skew(v)
    return np.transpose(_eye3 + a * sk + b * np.dot(sk, sk))


def log(q):
    """
    Compute the axis-angle representation of a rotation.
    If the return value of this function is v then the matrix logarithm of Q is skew(v).
    This is also known as the mapping from the Lie group SO(3) to the Lie algebra so(3).
    """
    q = np.asarray(q)
    assert np.shape(q) == (3, 3), 'log() received shape %s' % str(np.shape(q))

    # http://math.stackexchange.com/questions/83874/
    t = float(q.trace())
    v = np.array((q[2, 1] - q[1, 2],
                  q[0, 2] - q[2, 0],
                  q[1, 0] - q[0, 1]))
    if t >= 3. - 1e-8:
        return (.5 - (t-3.)/12.) * v
    elif t > -1. + 1e-8:
        th = np.arccos(t/2. - .5)
        return th / (2. * np.sin(th)) * v
    else:
        assert t <= -1. + 1e-8, 't=%f, R=%s' % (t, q)
        a = int(np.argmax(q[np.diag_indices_from(q)]))
        b = (a+1) % 3
        c = (a+2) % 3
        s = np.sqrt(q[a,a] - q[b, b] - q[c, c] + 1.)
        v = np.empty(3)
        v[a] = s/2.
        v[b] = (q[b, a] + q[a, b]) / (2.*s)
        v[c] = (q[c, a] + q[a, c]) / (2.*s)
        return v / np.linalg.norm(v)


def perturb_left(q, dq):
    """
    Compute exp(dq)*q where q is a 3x3 rotation and dq is an axis-angle vector.
    """
    return np.dot(exp(dq), q)


def displacement_left(r1, r2):
    """
    Find a vector v such that exp(v) * r1 = r2.
    """
    return log(np.dot(r2, r1.T))


def displacement_left_jacobian_wrt_lhs(r1, r2):
    # TODO: implement analytically
    from manifolds import numeric_jacobian
    return numeric_jacobian(lambda q: displacement_left(q, r2), r1, atlas=RotationAtlas)


def displacement_left_jacobian_wrt_rhs(r1, r2):
    # TODO: implement analytically
    from manifolds import numeric_jacobian
    return numeric_jacobian(lambda q: displacement_left(r1, q), r2, atlas=RotationAtlas)


def angular_velocity_from_axisangle_rates(x, dxdt):
    """
    Compute the derivative of exp(x(t)) w.r.t v, where dxdt is the derivative of x(.) w.r.t. t.
    """
    return -np.dot(exp_jacobian(x).T, dxdt)


class RotationAtlas(object):
    """
    Represents a mapping from 3D rotations to a local parameterization for rotations.
    """
    @classmethod
    def perturb(cls, q, delta):
        return perturb_left(q, delta)

    @classmethod
    def displacement(cls, r1, r2):
        return displacement_left(r1, r2)

    @classmethod
    def dof(cls, _):
        return 3


def unroll_axisangle(a, ref):
    """
    Out of all the axisangle vectors representing the same rotation as A, return the one closest to REF, where
    closest means L2 norm between axisangle vectors.

    Note that if A is a unit vector and t is in radians then the axisangle vector t*a represents the same rotation as
    a*(t + 2*pi), a*(t + 4*pi), etc. So if v is an axisangle vector then v represents the same rotation as
    v*(1. + 2*pi/norm(a)), v*(1. + 4*pi/norm(a)), etc.
    """
    num_full_circles = np.round(np.dot(ref - a, a) / (2. * np.pi * np.linalg.norm(a)))
    return a + num_full_circles * 2 * np.pi * a / np.linalg.norm(a)
