"""Tests for a unified matrix-composition convention.

Every interface that combines two or more transformation matrices must
follow the single convention documented by the module:

* "column vectors on the right": a homogeneous point ``p`` is transformed
  with ``M @ p`` (``numpy.dot(M, p)``).
* "row major storage" (C contiguous); the translation components live in
  the right column, ``M[:3, 3]``.
* Composition therefore left-multiplies: the matrix returned for the
  sequence ``(M0, M1, ...)`` equals ``M0 @ M1 @ ...`` and applies the
  rightmost transformation first.

All assertions use fixed transformation sequences (no random state) so
the expected matrices are pinned exactly. The single-matrix constructors
are treated as given; only composition results are checked here.
"""

import math

import numpy

from transformations import transformations as tf

# A fixed, non-symmetric transform sequence reused by most tests:
# translation, rotation (static xyz Euler angles), shear and scale.
ANGLES = (0.3, -0.7, 1.1)
TRANSLATION = (0.5, -1.0, 2.0)
SHEAR_ANGLE = 0.25
SCALE = 1.7

# A fixed homogeneous point transformed by the sequence.
POINT = numpy.array([1.5, -2.0, 0.7, 1.0])


def fixed_chain():
    """Return the fixed (T, R, Z, S) transformation sequence."""
    T = tf.translation_matrix(TRANSLATION)
    R = tf.euler_matrix(*ANGLES, axes='sxyz')
    Z = tf.shear_matrix(
        SHEAR_ANGLE, [1, 0, 0], [0, 0, 0], [0, 0, 1]
    )
    S = tf.scale_matrix(SCALE)
    return T, R, Z, S


def test_concatenate_matrices_is_left_multiplication():
    """concatenate_matrices(A, B) == A @ B (column-vector convention)."""
    T, R, Z, S = fixed_chain()
    expected = T @ R @ Z @ S
    result = tf.concatenate_matrices(T, R, Z, S)
    assert numpy.allclose(result, expected)
    # explicit numpy.dot chain, the reference spelling from the docstring
    assert numpy.allclose(
        result, numpy.dot(numpy.dot(numpy.dot(T, R), Z), S)
    )


def test_concatenate_matrices_pinned_matrix():
    """Pin the composed matrix of the fixed sequence exactly."""
    M = tf.concatenate_matrices(*fixed_chain())
    expected = numpy.array(
        [
            [0.5897801, -1.594189, 0.1237466, 0.5],
            [1.1587761, 0.4482381, -0.8644265, -1.0],
            [1.0951701, 0.3842447, 1.5218016, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert numpy.allclose(M, expected)


def test_concatenate_applies_rightmost_first():
    """The rightmost matrix of a chain acts on the point first."""
    T, R, Z, S = fixed_chain()
    M = tf.concatenate_matrices(T, R, Z, S)
    manual = T @ (R @ (Z @ (S @ POINT)))
    assert numpy.allclose(M @ POINT, manual)
    # reversing the arguments must give a different matrix in general
    assert not numpy.allclose(
        tf.concatenate_matrices(T, R), tf.concatenate_matrices(R, T)
    )


def test_concatenate_matrices_single_and_identity():
    """One-matrix concatenation is the identity composition."""
    (T,) = (tf.translation_matrix(TRANSLATION),)
    assert numpy.allclose(tf.concatenate_matrices(T), T)
    assert numpy.allclose(
        tf.concatenate_matrices(tf.identity_matrix(), T), T
    )
    assert numpy.allclose(
        tf.concatenate_matrices(T, tf.identity_matrix()), T
    )


def test_compose_matches_concatenate_convention():
    """compose_matrix builds P @ T @ R @ Z @ S, same as concatenate."""
    chain = tf.concatenate_matrices(*fixed_chain())
    scale, shear, angles, trans, persp = tf.decompose_matrix(chain)
    composed = tf.compose_matrix(scale, shear, angles, trans, persp)
    assert tf.is_same_transform(composed, chain)
    # and the composed matrix applies the point identically
    assert numpy.allclose(composed @ POINT, chain @ POINT)


def test_compose_matrix_factor_order():
    """The explicit factors compose in the documented P,T,R,Z,S order."""
    T, R, Z, S = fixed_chain()
    # decompose gives canonical factors; rebuild the same chain manually
    scale, shear, angles, trans, _ = tf.decompose_matrix(
        tf.concatenate_matrices(T, R, Z, S)
    )
    composed = tf.compose_matrix(
        scale=scale, shear=shear, angles=angles, translate=trans
    )
    reference = tf.concatenate_matrices(
        tf.translation_matrix(trans),
        tf.euler_matrix(angles[0], angles[1], angles[2], 'sxyz'),
    )
    # rotation/translation block must agree regardless of scale/shear
    assert numpy.allclose(composed[:3, 3], reference[:3, 3])
    assert composed.flags['C_CONTIGUOUS']


def test_decompose_compose_roundtrip_fixed():
    """decompose_matrix is the exact inverse of compose_matrix."""
    M0 = tf.compose_matrix(
        scale=(1.7, 1.7, 1.7),
        shear=(0.0, 0.0, math.tan(SHEAR_ANGLE)),
        angles=list(ANGLES),
        translate=list(TRANSLATION),
    )
    M1 = tf.compose_matrix(*tf.decompose_matrix(M0))
    assert tf.is_same_transform(M0, M1)


def test_translation_is_in_right_column():
    """Storage convention: translation components at M[:3, 3]."""
    T = tf.translation_matrix(TRANSLATION)
    M = tf.concatenate_matrices(T, tf.euler_matrix(*ANGLES))
    assert numpy.allclose(M[:3, 3], TRANSLATION)
    assert M.flags['C_CONTIGUOUS']
    # bottom row of an affine transform is [0, 0, 0, 1]
    assert numpy.allclose(M[3, :], [0.0, 0.0, 0.0, 1.0])


def test_euler_path_agrees_with_concatenate():
    """euler_matrix('rxyz') equals the concatenated axis rotations."""
    ai, aj, ak = 0.123, -1.234, 2.345
    Rx = tf.rotation_matrix(ai, [1, 0, 0])
    Ry = tf.rotation_matrix(aj, [0, 1, 0])
    Rz = tf.rotation_matrix(ak, [0, 0, 1])
    chained = tf.concatenate_matrices(Rx, Ry, Rz)
    euler = tf.euler_matrix(ai, aj, ak, 'rxyz')
    assert tf.is_same_transform(chained, euler)
    assert numpy.allclose(
        tf.euler_from_matrix(chained, 'rxyz'), [ai, aj, ak]
    )


def test_quaternion_path_agrees_with_concatenate():
    """Quaternion multiplication order matches matrix left-multiplication."""
    ai, aj, ak = ANGLES
    qx = tf.quaternion_about_axis(ai, [1, 0, 0])
    qy = tf.quaternion_about_axis(aj, [0, 1, 0])
    qz = tf.quaternion_about_axis(ak, [0, 0, 1])
    q = tf.quaternion_multiply(tf.quaternion_multiply(qx, qy), qz)
    from_quaternion = tf.quaternion_matrix(q)
    from_matrices = tf.concatenate_matrices(
        tf.rotation_matrix(ai, [1, 0, 0]),
        tf.rotation_matrix(aj, [0, 1, 0]),
        tf.rotation_matrix(ak, [0, 0, 1]),
    )
    assert tf.is_same_transform(from_quaternion, from_matrices)


def test_affine_registration_uses_same_convention():
    """affine_matrix_from_points returns M such that v1 == M @ v0."""
    rng = numpy.random.default_rng(7)
    M = tf.concatenate_matrices(
        tf.translation_matrix(TRANSLATION),
        tf.euler_matrix(*ANGLES),
        tf.scale_matrix(SCALE),
    )
    v0 = rng.random((4, 30))
    v0[3] = 1.0
    v1 = M @ v0
    recovered = tf.affine_matrix_from_points(v0[:3], v1[:3])
    assert numpy.allclose(v1, recovered @ v0, atol=1e-6)
    assert recovered.flags['C_CONTIGUOUS']


def test_superimposition_uses_same_convention():
    """superimposition_matrix (both algorithms) maps v0 onto v1 via M @ v0."""
    rng = numpy.random.default_rng(11)
    R = tf.euler_matrix(*ANGLES)
    v0 = rng.random((4, 20))
    v0[3] = 1.0
    v1 = R @ v0
    for usesvd in (True, False):
        M = tf.superimposition_matrix(v0, v1, usesvd=usesvd)
        assert numpy.allclose(v1, M @ v0, atol=1e-8)
        assert M.flags['C_CONTIGUOUS']
