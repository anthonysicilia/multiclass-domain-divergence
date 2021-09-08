from math import ceil, log, log2, sqrt
import torch

def hoeffding_bound(mc_est, m, delta):
    """
    Hoeffding's Bound
    """
    return mc_est + sqrt(1 / 2 * log(2 / delta) / m)

def mauer_term(kl_div, m, delta):
    """
    A Note on the PAC-Bayesian Theorem
    The term in the upperbound of Theorem 5
    """
    return (kl_div + log((2 * sqrt(m)) / delta)) / m

def langford_caruana_term(m, delta):
    """
    (Not) Bounding the True Error
    The term in the upperbound of Theorem 2.5
    """
    return log(2 / delta) / m

def dziugaite_pinsker_bound(sample_loss, _mauer_term):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2

    notes: use torch.sqrt to preserve kl gradients
    """
    return sample_loss + torch.sqrt(_mauer_term / 2)

def dziugaite_moment_bound(sample_loss, _mauer_term):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2

    notes: use torch.sqrt to preserve gradients
    """
    third_term = torch.sqrt(_mauer_term * (_mauer_term + 2 * sample_loss))
    return sample_loss + _mauer_term + third_term

def dziugaite_variational_bound(sample_loss, kl_div, m, delta):
    """
    On the Role of Data in PAC Bayes Bounds
    Appendix Theorem D.2
    """
    _mauer_term = mauer_term(kl_div, m, delta)
    a = dziugaite_pinsker_bound(sample_loss, _mauer_term)
    b = dziugaite_moment_bound(sample_loss, _mauer_term)
    # min preserves gradients
    return min(a, b)

def pac_bayes_hoeffding_bound(sample_loss, m, n, delta, *args):
    """
    Hoeffding bound
    + application of Langford and Caruana's Result
    + Pinsker's Inequality

    Technically just two applications of Hoeffding's bound
    param notes:
    m is the number data samples
    n is the number of hypothesis samples
    """
    delta = delta / 2
    mc_sample_loss = hoeffding_bound(sample_loss, n, delta)
    return hoeffding_bound(mc_sample_loss, m, delta)

def mauer_bound(sample_loss, m, n, delta, kl_div):
    """
    A Note on the PAC-Bayesian Theorem
    Theorem 5
    +
    (Not) Bounding the True Error
    Theorem 2.5

    Uses inverted KL divergence

    param notes:
    m is the number data samples
    n is the number of hypothesis samples
    """
    delta = delta / 2
    mc_sample_loss = inv_kl(sample_loss, langford_caruana_term(n, delta))
    return inv_kl(mc_sample_loss, mauer_term(kl_div, m, delta))

def binary_kl(q, p):
    """
    Binary KL divergence KL(q || p)
    """
    assert p != 1 and p != 0
    return q * log(q / p) + (1 - q) * log((1 - q) / (1 - p))

def inv_kl(q, epsilon, tolerance=1e-7):
    """
    Compute the inverse KL divergence 
    by way of the bisection method:
    https://en.wikipedia.org/wiki/Bisection_method

    Specifcally, we find the root of the below
    f(p) = epsilon - KL(q || p)
    f(q) = epsilon
    f(1) = -infinity; i.e., should have KL(q || 1 - 1e-8) > epsilon
    """
    a = q
    b = 1 - tolerance

    f = lambda x: epsilon - binary_kl(q, x)

    sign = lambda x: x > 0

    if not a < b:
        raise AssertionError('Condition a < b violated for bisection method.')
    if not f(a) > 0:
        raise AssertionError('Condition f(a) > 0 violated for bisection method.')
    if not f(b) < 0:
        print('Condition f(b) < 0 violated for bisection method.')
        print('Implies root 1 >= p > 1 - tolerance. So, returning 1.')
        return 1.0

    max_its = ceil(log2(abs(b - a) / tolerance))
    i = 0

    while i <= max_its:

        p = (a + b) / 2

        fp = f(p)

        if fp == 0 or abs(b - a) / 2 < tolerance:
            return p

        if sign(f(p)) == sign(f(a)):
            a = p
        else:
            b = p

        i += 1

    return p
