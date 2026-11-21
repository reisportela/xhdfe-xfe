"""Reject unreviewed drift from the unchanged CPU recurrence and stopping rules.

The CUDA tier's run_group (src/reduced_schur_pcg_cuda_controller.hpp) must equal the CPU
run_group (src/reduced_schur_pcg.hpp) after exactly two substitutions (package 3-lite,
17 November 2026): the iterative product on pv and the residual replacement on x go to the
adapter, each exactly once and with these arguments,

    iterative_operator.template apply<W>(g, team, pv, q, T, work);
    iterative_operator.template apply_replacement<W>(g, team, x, q, T, work);

and after the signature adaptations. The CPU reference is authenticated by the SHA-256 of
its run_group text, so that editing both copies at once cannot erase the evidence of a
change. Negative controls (an omitted, duplicated or misdirected call; a change in the fresh
sums, in a mask or in the refresh condition) must each fail the same check.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_RUN_GROUP_SHA256 = '8ec5fb1e0cb2773b2250239b09bd3036e2ed68987adb536ca02523bd4d471340'
ITERATIVE = 'iterative_operator.template apply<W>(g, team, pv, q, T, work);'
REPLACEMENT = 'iterative_operator.template apply_replacement<W>(g, team, x, q, T, work);'


def cpu_run_group():
    old = (ROOT / 'src/reduced_schur_pcg.hpp').read_text()
    a = old.index('template <int W>\ninline Decline run_group(')
    b = old.index('\n}  // namespace internal', a)
    expected = old[a:b].rstrip()
    digest = hashlib.sha256(expected.encode()).hexdigest()
    assert digest == CPU_RUN_GROUP_SHA256, 'the CPU run_group changed: review before re-authenticating'
    return expected


def check(new, expected):
    a = new.index('template <int W, class IterativeOperator>\ninline Decline run_group(')
    b = new.index('\ntemplate<class IterativeOperator>\ninline Report solve(', a)
    actual = new[a:b].rstrip()
    actual = actual.replace('template <int W, class IterativeOperator>', 'template <int W>', 1)
    actual = actual.replace('Report& rep, IterativeOperator& iterative_operator)', 'Report& rep)', 1)
    assert actual.count(ITERATIVE) == 1, 'the iterative product must go to the adapter exactly once, on pv'
    assert actual.count(REPLACEMENT) == 1, 'the replacement must go to the adapter exactly once, on x'
    actual = actual.replace(ITERATIVE, 'apply_operator<W>(g, team, pv, q, T, work);', 1)
    actual = actual.replace(REPLACEMENT, 'apply_operator<W>(g, team, x, q, T, work);', 1)
    assert actual == expected, 'CPU recurrence/stopping drift: review before regenerating'


def negative_controls(new):
    """Each mutation of the CUDA controller's text that must fail the check."""
    def once(text, old, replacement):
        assert text.count(old) == 1, 'a negative control lost its anchor: ' + old
        return text.replace(old, replacement, 1)
    fresh = 'acc[3 * W + w] += xv * gv[base + w];'
    mask = '                    if (!act[w]) continue;\n                    const double rv = gv[base + w] - q[base + w];'
    refresh = 'const bool refresh = any_candidate || periodic || any_enter;'
    return {
        'replacement omitted': once(new, REPLACEMENT, ''),
        'replacement duplicated': once(new, REPLACEMENT, REPLACEMENT + '\n            ' + REPLACEMENT),
        'replacement on p': once(new, REPLACEMENT, REPLACEMENT.replace('(g, team, x,', '(g, team, pv,')),
        'replacement on r': once(new, REPLACEMENT, REPLACEMENT.replace('(g, team, x,', '(g, team, r,')),
        'replacement as the iterative entry': once(new, REPLACEMENT, ITERATIVE.replace('pv,', 'x,')),
        'replacement back on the CPU': once(new, REPLACEMENT, 'apply_operator<W>(g, team, x, q, T, work);'),
        'iterative product omitted': once(new, ITERATIVE, ''),
        'iterative product on x': once(new, ITERATIVE, ITERATIVE.replace('pv,', 'x,')),
        'fresh sums changed': once(new, fresh, 'acc[3 * W + w] += xv * xv;'),
        'mask dropped': once(new, mask, mask.replace('if (!act[w]) continue;\n', '')),
        'refresh condition changed': once(new, refresh, 'const bool refresh = any_candidate || periodic;'),
    }


def main():
    expected = cpu_run_group()
    new = (ROOT / 'src/reduced_schur_pcg_cuda_controller.hpp').read_text()
    check(new, expected)
    assert expected.count('apply_operator<W>(g, team, x, q, T, work);') == 1, 'the CPU replacement must remain in the reference'
    controls = negative_controls(new)
    for name, mutated in controls.items():
        try:
            check(mutated, expected)
        except AssertionError:
            continue
        raise AssertionError('negative control passed the check: ' + name)
    print('CUDA_CONTROLLER_SOURCE_PARITY_PASS two substitutions, CPU reference authenticated, '
          f'{len(controls)} negative controls rejected')


if __name__ == '__main__':
    main()
