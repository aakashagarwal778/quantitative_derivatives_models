"""qdlib: quantitative derivatives library.

Reusable pricing models, numerical methods, volatility models, calibration
workflows, and empirical preprocessing utilities for quantitative finance.
"""

from . import calibration, common, empirical, jump_models, lattice_methods, local_volatility, monte_carlo, pde_ode_methods, pricing_foundations, sabr, stochastic_volatility, transform_methods

__all__ = [
    'pricing_foundations',
    'lattice_methods',
    'monte_carlo',
    'pde_ode_methods',
    'stochastic_volatility',
    'jump_models',
    'local_volatility',
    'sabr',
    'calibration',
    'transform_methods',
    'empirical',
    'common',
]

__version__ = '0.2.0'
