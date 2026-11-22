"""Tests for unified broadcasting and shape semantics of matrix operations.

Single-matrix and single-quaternion functions accept exactly one fixed
shape family and reject stacks (batches) and malformed shapes with
ValueError. Batched operations are only defined for inverse_matrix and
concatenate_matrices, following numpy.matmul broadcasting rules, so the
output shape is uniquely determined by the input shapes.
"""

import math

import numpy
import pytest

from transformations import transformations as tf

RNG = numpy.random.default_rng(42)


def rotation():
    """Return a valid single rotation matrix."""
    return tf.rotation_matrix(0.3, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])


def transform():
    """Return a valid single homogeneous transformation matrix."""
    return tf.concatenate_matrices(
        tf.translation_matrix([1.0, 2.0, 3.0]),
        rotation(),
        tf.scale_matrix(1.1),
    )


# fixed shape families of invalid inputs for single-matrix functions
BAD_MATRIX_SHAPES = [
    (2, 4, 4),  # batch of matrices
    (4, 4, 4),  # batch of matrices
    (1, 4, 4),  # batch of one matrix
    (3, 4),  # non-square 2D
    (4, 3),  # non-square 2D
    (5, 5),  # wrong square size
    (4,),  # vector
]

# fixed shape families of invalid inputs for single-quaternion functions
BAD_QUATERNION_SHAPES = [
    (2, 4),  # batch of quaternions
    (4, 4),  # matrix
    (3,),  # too short
    (5,),  # too long
]


def bad_arrays(shapes):
    """Return one array per shape family."""
    return [numpy.zeros(shape) for shape in shapes]


# ---------------------------------------------------------------------------
# single-matrix functions: valid input family (4, 4)


def test_translation_from_matrix_shape():
    assert tf.translation_from_matrix(transform()).shape == (3,)


def test_reflection_from_matrix_shape():
    M = tf.reflection_matrix([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    point, normal = tf.reflection_from_matrix(M)
    assert point.shape == (4,)
    assert normal.shape == (3,)


def test_rotation_from_matrix_shape():
    angle, direction, point = tf.rotation_from_matrix(rotation())
    assert isinstance(angle, float)
    assert direction.shape == (3,)
    assert point.shape == (4,)


def test_scale_from_matrix_shape():
    M = tf.scale_matrix(1.2, [0.1, 0.2, 0.3], [1.0, 0.0, 1.0])
    factor, origin, direction = tf.scale_from_matrix(M)
    assert isinstance(factor, float)
    assert origin.shape == (4,)
    assert direction.shape == (3,)


def test_projection_from_matrix_shape():
    M = tf.projection_matrix([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    point, normal, direction, perspective, pseudo = (
        tf.projection_from_matrix(M)
    )
    assert point.shape == (4,)
    assert normal.shape == (3,)


def test_shear_from_matrix_shape():
    M = tf.shear_matrix(0.3, [1.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 1.0, 0.0])
    angle, direction, point, normal = tf.shear_from_matrix(M)
    assert isinstance(angle, float)
    assert direction.shape == (3,)
    assert point.shape == (4,)
    assert normal.shape == (3,)


def test_decompose_matrix_shapes():
    scale, shear, angles, translate, perspective = tf.decompose_matrix(
        transform()
    )
    assert scale.shape == (3,)
    assert numpy.asarray(shear).shape == (3,)
    assert numpy.asarray(angles).shape == (3,)
    assert translate.shape == (3,)
    assert perspective.shape == (4,)


def test_euler_from_matrix_shapes():
    angles = tf.euler_from_matrix(rotation())
    assert len(angles) == 3
    assert all(isinstance(a, float) for a in angles)
    # a plain (3, 3) rotation matrix remains valid input
    angles3 = tf.euler_from_matrix(rotation()[:3, :3])
    assert numpy.allclose(angles, angles3)


def test_quaternion_from_matrix_shape():
    assert tf.quaternion_from_matrix(rotation()).shape == (4,)
    assert tf.quaternion_from_matrix(rotation(), isprecise=True).shape == (4,)


def test_is_same_transform_shape():
    M = transform()
    assert tf.is_same_transform(M, M) is True
    assert tf.is_same_transform(M, rotation()) is False


# ---------------------------------------------------------------------------
# single-matrix functions: invalid shape families raise ValueError


SINGLE_MATRIX_FUNCS = [
    tf.translation_from_matrix,
    tf.rotation_from_matrix,
    tf.scale_from_matrix,
    tf.shear_from_matrix,
    tf.decompose_matrix,
    tf.quaternion_from_matrix,
]


@pytest.mark.parametrize('func', SINGLE_MATRIX_FUNCS)
@pytest.mark.parametrize('shape', BAD_MATRIX_SHAPES)
def test_single_matrix_funcs_reject_bad_shapes(func, shape):
    with pytest.raises(ValueError, match='shape'):
        func(numpy.zeros(shape))


@pytest.mark.parametrize('shape', BAD_MATRIX_SHAPES)
def test_euler_from_matrix_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        tf.euler_from_matrix(numpy.zeros(shape))


@pytest.mark.parametrize('shape', [(2, 4, 4), (4, 4, 4), (3, 4), (5, 5)])
def test_is_same_transform_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        tf.is_same_transform(numpy.zeros(shape), numpy.identity(4))
    with pytest.raises(ValueError, match='shape'):
        tf.is_same_transform(numpy.identity(4), numpy.zeros(shape))


def test_reflection_from_matrix_rejects_batch():
    M = tf.reflection_matrix([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='shape'):
        tf.reflection_from_matrix(numpy.stack([M, M]))


def test_projection_from_matrix_rejects_batch():
    M = tf.projection_matrix([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='shape'):
        tf.projection_from_matrix(numpy.stack([M, M]))


# ---------------------------------------------------------------------------
# single-quaternion functions: valid input family (4,)


def test_quaternion_funcs_shapes():
    q = tf.random_quaternion(RNG.random(3))
    assert q.shape == (4,)
    assert tf.quaternion_matrix(q).shape == (4, 4)
    assert tf.quaternion_conjugate(q).shape == (4,)
    assert tf.quaternion_inverse(q).shape == (4,)
    assert tf.quaternion_multiply(q, q).shape == (4,)
    assert tf.quaternion_slerp(q, q, 0.5).shape == (4,)
    assert tf.euler_from_quaternion(q)[0] == pytest.approx(
        tf.euler_from_matrix(tf.quaternion_matrix(q))[0]
    )
    assert tf.is_same_quaternion(q, q) is True


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_quaternion_matrix_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_matrix(numpy.zeros(shape))


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_quaternion_conjugate_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_conjugate(numpy.zeros(shape))


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_quaternion_inverse_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_inverse(numpy.zeros(shape))


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_quaternion_multiply_rejects_bad_shapes(shape):
    q = tf.random_quaternion(RNG.random(3))
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_multiply(numpy.zeros(shape), q)
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_multiply(q, numpy.zeros(shape))


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_quaternion_slerp_rejects_bad_shapes(shape):
    q = tf.random_quaternion(RNG.random(3))
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_slerp(numpy.zeros(shape), q, 0.5)
    with pytest.raises(ValueError, match='shape'):
        tf.quaternion_slerp(q, numpy.zeros(shape), 0.5)


@pytest.mark.parametrize('shape', BAD_QUATERNION_SHAPES)
def test_is_same_quaternion_rejects_bad_shapes(shape):
    q = tf.random_quaternion(RNG.random(3))
    with pytest.raises(ValueError, match='shape'):
        tf.is_same_quaternion(numpy.zeros(shape), q)
    with pytest.raises(ValueError, match='shape'):
        tf.is_same_quaternion(q, numpy.zeros(shape))


# ---------------------------------------------------------------------------
# batched operations: unified numpy.matmul broadcasting semantics


def test_concatenate_matrices_single_unchanged():
    M = RNG.random((4, 4))
    assert numpy.allclose(tf.concatenate_matrices(M), M)
    assert numpy.allclose(
        tf.concatenate_matrices(M, M.T, M), M @ M.T @ M
    )


def test_concatenate_matrices_batch_shapes():
    A = RNG.random((5, 4, 4))
    B = RNG.random((4, 4))
    C = RNG.random((5, 4, 4))
    # batch @ single -> batch
    R = tf.concatenate_matrices(A, B)
    assert R.shape == (5, 4, 4)
    assert numpy.allclose(R, A @ B)
    # single @ batch -> batch
    R = tf.concatenate_matrices(B, A)
    assert R.shape == (5, 4, 4)
    assert numpy.allclose(R, B @ A)
    # batch @ batch -> batch
    R = tf.concatenate_matrices(A, C)
    assert R.shape == (5, 4, 4)
    assert numpy.allclose(R, A @ C)
    # broadcasting of batch dimensions
    A2 = RNG.random((2, 1, 4, 4))
    C2 = RNG.random((3, 4, 4))
    R = tf.concatenate_matrices(A2, C2)
    assert R.shape == (2, 3, 4, 4)
    assert numpy.allclose(R, A2 @ C2)


def test_concatenate_matrices_incompatible_batches_raise():
    with pytest.raises(ValueError):
        tf.concatenate_matrices(
            RNG.random((2, 4, 4)), RNG.random((3, 4, 4))
        )


def test_inverse_matrix_batch_shape():
    M = RNG.random((3, 4, 4))
    Minv = tf.inverse_matrix(M)
    assert Minv.shape == (3, 4, 4)
    assert numpy.allclose(Minv @ M, numpy.identity(4))


# ---------------------------------------------------------------------------
# single-matrix results are unchanged by validation


def test_single_matrix_results_unchanged():
    M = transform()
    assert numpy.allclose(tf.translation_from_matrix(M), [1.0, 2.0, 3.0])
    R = rotation()
    q = tf.quaternion_from_matrix(R)
    assert numpy.allclose(
        tf.quaternion_matrix(q), tf.quaternion_matrix(q.copy())
    )
    assert tf.is_same_transform(
        tf.euler_matrix(*tf.euler_from_matrix(R)), R
    )
    angle, direction, point = tf.rotation_from_matrix(R)
    assert math.isclose(angle, 0.3)
