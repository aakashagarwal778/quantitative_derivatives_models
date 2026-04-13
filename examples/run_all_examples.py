from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = [
    'pricing_foundations/compare_black_scholes_prices.py',
    'pricing_foundations/plot_black_scholes_greeks.py',
    'pricing_foundations/solve_implied_volatility.py',
    'lattice_methods/compare_binomial_and_black_scholes.py',
    'lattice_methods/price_american_put_with_crr.py',
    'lattice_methods/compare_binomial_and_trinomial.py',
    'monte_carlo/price_european_option_with_monte_carlo.py',
    'monte_carlo/demonstrate_antithetic_variates.py',
    'monte_carlo/demonstrate_control_variates.py',
    'monte_carlo/demonstrate_importance_sampling.py',
    'monte_carlo/price_american_option_with_longstaff_schwartz.py',
    'pde_ode_methods/solve_option_price_with_explicit_fd.py',
    'pde_ode_methods/solve_option_price_with_crank_nicolson.py',
    'pde_ode_methods/solve_perpetual_american_put.py',
    'stochastic_volatility/price_options_with_heston_fft.py',
    'stochastic_volatility/plot_heston_implied_vol_smile.py',
    'stochastic_volatility/calibrate_heston_to_market_data.py',
    'jump_models/compare_merton_and_black_scholes.py',
    'local_volatility/build_local_vol_surface.py',
    'local_volatility/compare_local_vol_and_black_scholes.py',
    'sabr/plot_sabr_smile.py',
    'sabr/calibrate_sabr.py',
    'calibration/run_heston_calibration.py',
    'transform_methods/price_with_laplace_black_scholes.py',
    'empirical/estimate_volatility_from_returns.py',
]

for rel in EXAMPLE_FILES:
    path = ROOT / 'examples' / rel
    print(f'Running {rel} ...')
    subprocess.run([sys.executable, str(path)], check=True)
print('All example scripts executed successfully.')
