Revisions
---------

Unreleased

- Define and document consistent Euler angle conventions across
  ``euler_matrix``, ``euler_from_matrix``, ``quaternion_from_euler`` and
  ``euler_from_quaternion``: default 'sxyz' axis sequence, radian
  right-hand-rule sign convention and canonical principal-value ranges
  (first/third angles in (-pi, pi]; second angle in [-pi/2, pi/2] for
  Tait-Bryan sequences, [0, pi] or [-pi, 0] for proper Euler sequences).
- Return principal-value angles from the inverse Euler conversions,
  removing spurious 2*pi jumps and negative-zero outputs.
- Use the regular ``atan2`` formulas unless the gimbal-lock pivot term is
  exactly zero, avoiding an artificial threshold discontinuity near
  singularities while keeping the degenerate branch for truly singular
  matrices.
- Add fixed-angle tests covering all 24 axis sequences, both the Python
  and C implementations.

2026.8.8

- Fix code review issues.
- Make C extension ABI3 and free-threading compatible.
- Drop support for Python 3.11 and numpy 2.0 (SPEC0).
- Support Python 3.15.

2026.1.18

- Use multi-phase initialization.
- Improve code quality.

2025.8.1

- Drop support for Python 3.10, support Python 3.14.

2025.1.1

- Drop support for Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.4.24

- Support NumPy 2.

2024.1.6

- Remove support for Python 3.8 and numpy 1.22 (NEP 29).

2022.9.26

- Add dimension check on superimposition_matrix (#2).

2022.8.26

- Update metadata
- Remove support for Python 3.7 (NEP 29).

2021.6.6

- Remove support for Python 3.6 (NEP 29).

2020.1.1

- Remove support for Python 2.7 and 3.5.

2019.4.22

- Fix setup requirements.
