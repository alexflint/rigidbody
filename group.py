import numpy as np

from . import rotation


class SO3(object):
    """
    Represents a rotation in three dimensions.
    """
    DoF = 3

    class Atlas(object):
        """
        Represents an atlas for rotations.
        """
        @classmethod
        def dof(cls, _):
            return SO3.DoF

        @classmethod
        def perturb(cls, r, tangent):
            """
            Evaluate the chart for the given rotation.
            """
            assert len(tangent) == SO3.DoF
            return SO3(rotation.perturb_left(r.matrix, tangent))

        @classmethod
        def displacement(cls, r1, r2):
            """
            Get a vector v such that perturb(x1, v) = x2.
            """
            return rotation.log(np.dot(SO3.asarray(r2), SO3.asarray(r1).T))

    def __init__(self, r):
        """
        Initialize from a rotation matrix.
        """
        self._r = r

    @classmethod
    def identity(cls):
        """
        Construct the identity rotation.
        """
        return SO3(np.eye(3))

    @classmethod
    def from_tangent(cls, v):
        """
        Construct a rotation for the tangent space at the identity element.
        """
        assert len(v) == SO3.DoF
        return SO3(rotation.exp(v))

    @classmethod
    def asarray(cls, x):
        """
        Convert an SO3 to a matrix.
        """
        if isinstance(x, SO3):
            return x.matrix
        else:
            return np.asarray(x)

    @property
    def matrix(self):
        """
        Get the matrix representation of this rotation.
        """
        return self._r

    def __mul__(self, rhs):
        """
        Multiply this rotation with another.
        """
        return self.transform(rhs)

    def transform(self, rhs):
        """
        Multiply this rotation with another.
        """
        if isinstance(rhs, SO3):
            return SO3(np.dot(self._r, rhs._r))
        elif isinstance(rhs, np.ndarray):
            return SO3(np.dot(self._r, rhs))

    def inverse(self):
        """
        Get the inverse of this rotation
        """
        return SO3(self._r.T)

    def log(self):
        """
        Compute the axis angle representation of this rotation.
        """
        return rotation.log(self._r)

    def __str__(self):
        """
        Get a string representation of this rotation.
        """
        return 'SO3(%s)' % str(self._r).replace('\n', '\n    ')


class SE3(object):
    """
    Represents a rigid transform in three dimensions.
    """
    DoF = 6

    class Atlas(object):
        """
        Represents an atlas for rigid transforms.
        """
        @classmethod
        def dof(cls, _):
            return SE3.DoF

        @classmethod
        def perturb(cls, pose, tangent):
            """
            Evaluate the chart for the given pose at tangent.
            """
            assert len(tangent) == SE3.DoF
            return SE3(rotation.perturb_left(pose.orientation, tangent[:3]), pose.position + tangent[3:])

        @classmethod
        def displacement(cls, x1, x2):
            """
            Get a vector v such that perturb(x1, v) = x2.
            """
            return np.hstack((rotation.log(np.dot(x2.orientation, x1.orientation.T)), x2.position - x1.position))

    def __init__(self, orientation, position):
        """
        Initialize a rigid body transform from a rotation matrix and position vector.
        """
        self._orientation = np.asarray(orientation, float)
        self._position = np.asarray(position, float)

    @classmethod
    def identity(cls):
        """
        Get the identity transform.
        """
        return SE3(np.eye(3), np.zeros(3))

    @classmethod
    def from_tangent(cls, v):
        """
        Construct a rigid body transform from the tangent space at the identity element.
        """
        assert len(v) == SE3.DoF
        return SE3(rotation.exp(v[:3]), v[3:])

    @classmethod
    def from_matrix(cls, m):
        """
        Construct a rigid body transform from a 3x4 or 4x4 matrix
        """
        m = np.asarray(m)
        assert m.shape in ((3, 4), (4, 4)), 'shape was %s' % str(m.shape)
        r = m[:3, :3]
        t = m[:3, 3]
        return SE3(r, -np.dot(r.T, t))

    @property
    def orientation(self):
        """
        Get the orientation component of this transform.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, v):
        """
        Set the orientation component of this transform.
        """
        self._orientation = v

    @property
    def position(self):
        """
        Get the position component of this transform.
        """
        return self._position

    @position.setter
    def position(self, v):
        """
        Set the position component of this transform.
        """
        self._position = v

    @property
    def matrix(self):
        """
        Get the matrix representation of this transform.
        """
        return np.r_[np.c_[self._orientation, -np.dot(self._orientation, self._position)],
                     np.c_[0., 0., 0., 1.]]

    @property
    def rp(self):
        """
        Get the (rotation, position) pair for this transform.
        """
        return self._orientation, self._position

    @property
    def rt(self):
        """
        Get the (rotation, translation) pair for this transform.
        """
        return self._orientation, -np.dot(self._orientation, self._position)

    def __mul__(self, rhs):
        """
        Multiply this transform with another.
        """
        return self.transform(rhs)

    def transform(self, rhs):
        """
        Multiply this transform with another.
        """
        if isinstance(rhs, SE3):
            r1, r2 = self._orientation, rhs._orientation
            return SE3(np.dot(r1, r2), rhs.position + np.dot(r2.T, self.position))
        elif isinstance(rhs, np.ndarray):
            if rhs.shape[-1] == 3:
                return np.dot(self._orientation, rhs - self.position)
            elif rhs.shape[-1] == 4:
                return np.dot(self.matrix, rhs)

    def inverse(self):
        """
        Get the inverse of this transform.
        """
        return SE3(self._orientation.T, -np.dot(self.orientation, self.position))

    def log(self):
        """
        Map this transform to the Lie algebra se3.
        """
        return np.concatenate((rotation.log(self.orientation), self.position))

    def __str__(self):
        """
        Get a string representation of this transform.
        """
        return 'SE3(position=%s, log_rotation=%s)' % (self._position, rotation.log(self._orientation))
