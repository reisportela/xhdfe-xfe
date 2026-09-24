"""Saturation must be proved by rank, not inferred from a tiny positive RSS."""
from fractions import Fraction

from exact_two_fe_cycle_oracle import certify

x = [[1., 0.], [0., 1.], [0., 0.]]
y = [7., 2., 5.]
fit = certify(y, x, [0] * 3, [0] * 3)
assert fit['saturated_point_identified'] and fit['residual_df'] == 0
assert fit['b_exact'] == [Fraction(2), Fraction(-3), Fraction(5)]

# A fourth row leaves one residual degree of freedom, with exact RSS=2^-81.
near = certify(y + [5. + 2.**-40], x + [[0., 0.]], [0] * 4, [0] * 4)
assert not near['saturated_point_identified'] and near['cycle_dimension'] == 3

singular = certify(y, [[1., 2.], [0., 0.], [0., 0.]], [0] * 3, [0] * 3)
assert not singular['saturated_point_identified']

disconnected = certify(y + [9.], x + [[0., 0.]], [0, 0, 0, 1], [0, 0, 0, 1])
assert disconnected['saturated_point_identified'] and disconnected['components'] == 2
assert disconnected['b_exact'] == [Fraction(2), Fraction(-3), Fraction(6)]

scaled = certify(y, [[row[0] * 2.**-20, row[1]] for row in x], [0] * 3, [0] * 3)
assert scaled['b_exact'] == [Fraction(2**21), Fraction(-3), Fraction(5)]
print('PASS: exact saturation, small positive residual variance, singular contrasts, disconnected FEs and units')
