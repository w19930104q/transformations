# transformations/test_euler_conventions.py

"""Tests for explicit, consistent Euler angle conversion conventions.

The tests pin down the axis order, sign and principal-value conventions
shared by ``euler_matrix``, ``euler_from_matrix``,
``quaternion_from_euler`` and ``euler_from_quaternion``, including
stability around the gimbal-lock singularities.

Both the pure-Python functions and, when available, the compiled C
implementations are exercised.
"""

from __future__ import annotations

import math

import numpy
import pytest

from transformations import transformations as tm

# every public Euler function and its pure-Python twin (if the C extension
# is installed, the public name is the C implementation)
EULER_FUNCTIONS = [
    'euler_matrix',
    'euler_from_matrix',
    'euler_from_quaternion',
    'quaternion_from_euler',
]

IMPLEMENTATIONS = [('', 'C extension when available, else Python')]
for name in EULER_FUNCTIONS:
    if hasattr(tm, name + '_py'):
        IMPLEMENTATIONS = [
            ('', 'compiled C implementation'),
            ('_py', 'pure Python implementation'),
        ]
        break

ALL_AXES = list(tm._AXES2TUPLE.keys())

# fixed angle triplets, including values spanning the full angle range
FIXED_ANGLES = [
    (0.123, -0.456, 0.789),
    (1.0, 2.0, 3.0),
    (-1.5, 0.25, -2.75),
    (math.pi, -math.pi / 3.0, math.pi / 6.0),
    (0.0, 0.0, 0.0),
]


def _impl(suffix):
    """Return the four Euler functions for the given implementation suffix."""
    return tuple(getattr(tm, name + suffix) for name in EULER_FUNCTIONS)


@pytest.mark.parametrize('suffix,label', IMPLEMENTATIONS)
class TestEulerConventions:
    """Convention tests run against every available implementation."""

    def test_default_axes_is_sxyz(self, suffix, label):
        """The default axis sequence is static x-y-z on every interface."""
        (
            euler_matrix,
            euler_from_matrix,
            euler_from_quaternion,
            quaternion_from_euler,
        ) = _impl(suffix)
        ai, aj, ak = 0.123, -0.456, 0.789
        assert numpy.allclose(
            euler_matrix(ai, aj, ak), euler_matrix(ai, aj, ak, 'sxyz')
        )
        R = euler_matrix(ai, aj, ak, 'sxyz')
        assert numpy.allclose(euler_from_matrix(R),
                              euler_from_matrix(R, 'sxyz'))
        q = quaternion_from_euler(ai, aj, ak, 'sxyz')
        assert numpy.allclose(quaternion_from_euler(ai, aj, ak), q)
        assert numpy.allclose(
            euler_from_quaternion(q), euler_from_quaternion(q, 'sxyz')
        )

    def test_sxyz_matrix_product_order(self, suffix, label):
        """'sxyz' equals Rz(ak) @ Ry(aj) @ Rx(ai) on column vectors."""
        euler_matrix, _, _, _ = _impl(suffix)
        ai, aj, ak = 0.34, -0.87, 1.56
        # concatenate_matrices(A, B, C) returns A @ B @ C
        expected = tm.concatenate_matrices(
            tm.rotation_matrix(ak, [0, 0, 1]),
            tm.rotation_matrix(aj, [0, 1, 0]),
            tm.rotation_matrix(ai, [1, 0, 0]),
        )
        assert numpy.allclose(euler_matrix(ai, aj, ak, 'sxyz'), expected)

    @pytest.mark.parametrize('angles', FIXED_ANGLES)
    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_matrix_round_trip_all_axes(self, suffix, label, angles, axes):
        """Fixed angles reconstruct the same matrix for all 24 sequences."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        R0 = euler_matrix(*angles, axes=axes)
        recovered = euler_from_matrix(R0, axes)
        R1 = euler_matrix(*recovered, axes=axes)
        assert numpy.allclose(R0, R1, atol=1e-12)

    @pytest.mark.parametrize('angles', FIXED_ANGLES)
    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_quaternion_path_matches_matrix_path(
        self, suffix, label, angles, axes
    ):
        """Euler->quaternion->matrix equals Euler->matrix for every axes."""
        (
            euler_matrix,
            _,
            euler_from_quaternion,
            quaternion_from_euler,
        ) = _impl(suffix)
        R = euler_matrix(*angles, axes=axes)
        Rq = tm.quaternion_matrix(
            quaternion_from_euler(*angles, axes=axes)
        )
        assert numpy.allclose(R, Rq, atol=1e-12)
        # and the inverse decomposition agrees
        from_matrix = tm.euler_from_matrix(R, axes)
        from_quaternion = euler_from_quaternion(
            quaternion_from_euler(*angles, axes=axes), axes
        )
        assert numpy.allclose(from_matrix, from_quaternion, atol=1e-12)

    @pytest.mark.parametrize('angles', FIXED_ANGLES)
    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_principal_value_ranges(self, suffix, label, angles, axes):
        """Inverse angles always lie in the documented canonical ranges."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        _, parity, repetition, _ = tm._AXES2TUPLE[axes]
        if repetition:
            mid_lo, mid_hi = (
                (0.0, math.pi) if parity == 0 else (-math.pi, 0.0)
            )
        else:
            mid_lo, mid_hi = -math.pi / 2.0, math.pi / 2.0
        ax, ay, az = euler_from_matrix(
            euler_matrix(*angles, axes=axes), axes
        )
        # outer angles: (-pi, pi]
        assert -math.pi < ax <= math.pi
        assert -math.pi < az <= math.pi
        # middle angle
        assert mid_lo - 1e-12 <= ay <= mid_hi + 1e-12
        # never a negative zero
        for value in (ax, ay, az):
            assert not (value == 0.0 and math.copysign(1.0, value) < 0)

    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_identity_decomposes_to_zero_angles(self, suffix, label, axes):
        """The identity matrix yields zero angles on every axis sequence."""
        _, euler_from_matrix, _, _ = _impl(suffix)
        angles = euler_from_matrix(numpy.identity(4), axes)
        assert numpy.allclose(angles, (0.0, 0.0, 0.0), atol=1e-12)

    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_exact_lock_is_deterministic_and_reconstructs(
        self, suffix, label, axes
    ):
        """Exact gimbal-lock matrices give stable, reconstructing angles."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        repetition = tm._AXES2TUPLE[axes][2]
        lock = math.pi if repetition else math.pi / 2.0
        for middle in (lock, -lock):
            R = euler_matrix(0.3, middle, 0.4, axes)
            first = euler_from_matrix(R, axes)
            # repeated calls return identical tuples (no randomness/flips)
            assert first == euler_from_matrix(R, axes)
            assert numpy.allclose(
                euler_matrix(*first, axes=axes), R, atol=1e-12
            )

    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_degenerate_zero_matrix_uses_zero_undetermined_angle(
        self, suffix, label, axes
    ):
        """A truly singular matrix (pivot term exactly zero) hits the
        degenerate branch, which reports the undetermined angle as zero."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        repetition = tm._AXES2TUPLE[axes][2]
        frame = tm._AXES2TUPLE[axes][3]
        undetermined = 0 if frame else 2
        lock = math.pi if repetition else math.pi / 2.0
        for middle in (lock, -lock):
            R = euler_matrix(0.3, middle, 0.4, axes).copy()
            # remove floating-point residuals so the pivot term is exactly 0
            R[numpy.abs(R) < 1e-14] = 0.0
            angles = euler_from_matrix(R, axes)
            assert angles[undetermined] == 0.0
            assert numpy.allclose(
                euler_matrix(*angles, axes=axes), R, atol=1e-12
            )

    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_no_2pi_jump_near_lock(self, suffix, label, axes):
        """Angles stay on principal branches and reconstruct near a lock.

        A genuine gimbal lock changes the branch of the first and third
        angles, but every returned angle must remain inside the
        principal-value range and both the near-lock and exact-lock
        decompositions must reconstruct the same rotation matrix.
        """
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        repetition = tm._AXES2TUPLE[axes][2]
        lock = math.pi if repetition else math.pi / 2.0
        for sign in (1.0, -1.0):
            middle = sign * lock
            R_near = euler_matrix(
                0.3, middle - sign * 1e-10, 0.4, axes
            )
            R_exact = euler_matrix(0.3, middle, 0.4, axes)
            near = euler_from_matrix(R_near, axes)
            exact = euler_from_matrix(R_exact, axes)
            for value in (*near, *exact):
                assert -math.pi <= value <= math.pi
            assert numpy.allclose(
                euler_matrix(*near, axes=axes), R_near, atol=1e-12
            )
            assert numpy.allclose(
                euler_matrix(*exact, axes=axes), R_exact, atol=1e-12
            )

    def test_tait_bryan_lock_continuous_default_sxyz(self, suffix, label):
        """For 'sxyz' the exact-lock angles equal the limiting branch."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        limit = euler_from_matrix(
            euler_matrix(0.3, math.pi / 2.0 - 1e-12, 0.4, 'sxyz'), 'sxyz'
        )
        exact = euler_from_matrix(
            euler_matrix(0.3, math.pi / 2.0, 0.4, 'sxyz'), 'sxyz'
        )
        assert numpy.allclose(limit, exact, atol=1e-9)

    @pytest.mark.parametrize('axes', ALL_AXES)
    def test_tuple_and_string_axes_agree(self, suffix, label, axes):
        """The encoded 4-tuple is an alias for the 4-character string."""
        euler_matrix, euler_from_matrix, _, _ = _impl(suffix)
        tuple_axes = tm._AXES2TUPLE[axes]
        angles = (0.321, -0.654, 0.987)
        assert numpy.allclose(
            euler_matrix(*angles, axes=axes),
            euler_matrix(*angles, axes=tuple_axes),
        )
        R = euler_matrix(*angles, axes=axes)
        assert numpy.allclose(
            euler_from_matrix(R, axes),
            euler_from_matrix(R, tuple_axes),
        )

    def test_fixed_reference_values_sxyz(self, suffix, label):
        """Pin exact numerical output for a fixed non-singular triplet."""
        euler_matrix, euler_from_matrix, _, quaternion_from_euler = (
            _impl(suffix)
        )
        angles = (0.123, -0.456, 0.789)
        R = euler_matrix(*angles, axes='sxyz')
        # away from any singularity the original angles are recovered
        assert numpy.allclose(euler_from_matrix(R, 'sxyz'), angles,
                              atol=1e-12)
        # fixed quaternion reference value
        q = quaternion_from_euler(*angles, axes='sxyz')
        expected_q = numpy.array(
            [0.89225758, 0.14198145, -0.18526256, 0.38651709]
        )
        assert numpy.allclose(q, expected_q, atol=1e-7)

    def test_invalid_axes_raise(self, suffix, label):
        euler_matrix, euler_from_matrix, _, quaternion_from_euler = (
            _impl(suffix)
        )
        with pytest.raises((KeyError, ValueError)):
            euler_matrix(1, 2, 3, 'xxxx')
        with pytest.raises((KeyError, ValueError)):
            quaternion_from_euler(1, 2, 3, 'xxxx')
        with pytest.raises((KeyError, ValueError)):
            euler_from_matrix(numpy.identity(4), 'xxxx')
