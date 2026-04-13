# qdlib

![PyPI - Version](https://img.shields.io/badge/pypi-v0.1.2-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

---

### Installation
Since this is currently on TestPyPI, install it using:

```bash
pip install -i [https://test.pypi.org/simple/](https://test.pypi.org/simple/) qdlib
```

A structured Python library for foundational and intermediate quantitative derivatives models, numerical pricing methods, volatility models, calibration workflows, and explanatory notebooks.

## What this project is

`qdlib` is designed as a clean, educational, and reusable quant library.

The goal is not just to store scripts, but to organize core derivatives models in a way that is:

- mathematically sound
- easy to navigate
- reusable as a Python package
- supported by examples, tests, and notebooks

This repository focuses on **core pricing models and numerical methods** rather than exotic contract design. Exotic options are intended to live in a separate dedicated project.

---

## Scope

The library currently covers the following areas:

- pricing foundations
- lattice methods
- Monte Carlo methods
- PDE and ODE methods
- stochastic volatility
- jump models
- local volatility
- SABR
- calibration workflows
- transform methods
- empirical preprocessing

The emphasis throughout is on clarity, correctness, and structure.

---

## Repository structure

```text
quant-derivatives-library/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── qdlib/
├── examples/
├── tests/
├── notebooks/
├── data/
└── docs/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.