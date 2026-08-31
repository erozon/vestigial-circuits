"""
Dihedral group D_n math.

Elements of D_n (order 2n):
- Rotations: r0, r1, ..., r{n-1}  (r_k = rotation by 2*pi*k/n)
- Reflections: s0, s1, ..., s{n-1} (s_k = reflection through axis k)

Multiplication rules:
  r_a * r_b = r_{(a+b) mod n}
  r_a * s_b = s_{(a+b) mod n}
  s_a * r_b = s_{(a-b) mod n}
  s_a * s_b = r_{(a-b) mod n}
"""


def parse_element(name):
    """Parse element string like 'r3' or 's12' into (type, index)."""
    return name[0], int(name[1:])


def element_name(typ, idx, n):
    """Create element string from type and index (mod n)."""
    return f"{typ}{idx % n}"


def multiply(g, h, n):
    """Compute g * h in D_n."""
    g_type, g_idx = parse_element(g)
    h_type, h_idx = parse_element(h)

    if g_type == 'r' and h_type == 'r':
        return element_name('r', (g_idx + h_idx) % n, n)
    elif g_type == 'r' and h_type == 's':
        return element_name('s', (g_idx + h_idx) % n, n)
    elif g_type == 's' and h_type == 'r':
        return element_name('s', (g_idx - h_idx) % n, n)
    elif g_type == 's' and h_type == 's':
        return element_name('r', (g_idx - h_idx) % n, n)


def elements(n):
    """Return list of all 2n elements of D_n."""
    return [f"r{i}" for i in range(n)] + [f"s{i}" for i in range(n)]


def identity(n):
    """Return the identity element r0."""
    return "r0"


def inverse(g, n):
    """Compute the inverse of element g in D_n."""
    g_type, g_idx = parse_element(g)
    if g_type == 'r':
        return element_name('r', (-g_idx) % n, n)
    else:
        # s_a * s_a = r_0, so reflections are self-inverse
        return g


def multiplication_table(n):
    """Compute the full 2n x 2n multiplication table.

    Returns:
        Dict mapping (g, h) -> g*h for all pairs.
    """
    elems = elements(n)
    table = {}
    for g in elems:
        for h in elems:
            table[(g, h)] = multiply(g, h, n)
    return table


def verify_group_axioms(n):
    """Verify that D_n satisfies group axioms. Returns True if all pass."""
    elems = elements(n)
    e = identity(n)

    # Identity: e * g = g * e = g
    for g in elems:
        assert multiply(e, g, n) == g, f"Identity failed: {e} * {g} != {g}"
        assert multiply(g, e, n) == g, f"Identity failed: {g} * {e} != {g}"

    # Inverse: g * g^-1 = g^-1 * g = e
    for g in elems:
        g_inv = inverse(g, n)
        assert multiply(g, g_inv, n) == e, f"Inverse failed: {g} * {g_inv} != {e}"
        assert multiply(g_inv, g, n) == e, f"Inverse failed: {g_inv} * {g} != {e}"

    # Associativity (spot check — full check is O(n^3))
    import random
    random.seed(42)
    for _ in range(min(1000, len(elems) ** 3)):
        a, b, c = random.choices(elems, k=3)
        ab_c = multiply(multiply(a, b, n), c, n)
        a_bc = multiply(a, multiply(b, c, n), n)
        assert ab_c == a_bc, f"Associativity failed: ({a}*{b})*{c} = {ab_c} != {a}*({b}*{c}) = {a_bc}"

    return True
