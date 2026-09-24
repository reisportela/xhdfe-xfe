"""Exact saturation certificate for represented inputs with two ordinary FEs.

The bipartite incidence rank is vertices minus connected components. Its
fundamental cycles span the left nullspace. Two independent regressor contrasts
fill a two-dimensional cycle space, proving full row rank without a float solve.
This is deliberately not a waiver based on a small residual or saturated_ flag.
"""
from collections import deque
from fractions import Fraction


def certify(y, x, first, second):
    n = len(y)
    assert len(x) == len(first) == len(second) == n and n > 0
    assert all(len(row) == 2 for row in x)
    left = {v: i for i, v in enumerate(dict.fromkeys(first))}
    right = {v: i + len(left) for i, v in enumerate(dict.fromkeys(second))}
    edges = [(left[a], right[b]) for a, b in zip(first, second)]
    vertices = len(left) + len(right)
    parent, sizes = list(range(vertices)), [1] * vertices
    tree, chords = [[] for _ in parent], []

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for i, (u, v) in enumerate(edges):
        a, b = root(u), root(v)
        if a == b:
            chords.append(i)
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b] = a
        sizes[a] += sizes[b]
        tree[u].append((v, i))
        tree[v].append((u, i))
    components = len({root(i) for i in range(vertices)})
    rank_fe = vertices - components
    result = dict(observations=n, rank_fe=rank_fe, components=components,
                  cycle_dimension=n - rank_fe, saturated_point_identified=False)
    assert len(chords) == result['cycle_dimension']
    if len(chords) != 2:
        return result

    cycles = []
    for chord in chords:
        u, v = edges[chord]
        previous, queue = {v: (None, None)}, deque([v])
        while u not in previous:
            at = queue.popleft()
            for other, edge in tree[at]:
                if other not in previous:
                    previous[other] = (at, edge)
                    queue.append(other)
        cycle, at = {chord: 1}, u
        while at != v:
            before, edge = previous[at]
            cycle[edge] = 1 if edges[edge][0] == before else -1
            at = before
        balance = [0] * vertices
        for edge, sign in cycle.items():
            a, b = edges[edge]
            balance[a] += sign
            balance[b] -= sign
        assert not any(balance)
        cycles.append(cycle)

    rational = lambda v: Fraction.from_float(float(v))
    matrix = [[sum((sign * rational(x[i][j]) for i, sign in cycle.items()), Fraction(0))
               for j in range(2)] for cycle in cycles]
    rhs = [sum((sign * rational(y[i]) for i, sign in cycle.items()), Fraction(0))
           for cycle in cycles]
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if determinant == 0:
        return result
    beta = [(d * rhs[0] - b * rhs[1]) / determinant,
            (a * rhs[1] - c * rhs[0]) / determinant]
    mean_y = sum(map(rational, y), Fraction(0)) / n
    mean_x = [sum((rational(row[j]) for row in x), Fraction(0)) / n for j in range(2)]
    beta.append(mean_y - sum((mean_x[j] * beta[j] for j in range(2)), Fraction(0)))
    assert rank_fe + 2 == n
    result.update(saturated_point_identified=True, rank_full=n,
                  residual_df=0, b_exact=beta, cycle_determinant=determinant)
    return result
