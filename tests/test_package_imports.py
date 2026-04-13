import qdlib


def test_top_level_version_present():
    assert hasattr(qdlib, '__version__')


def test_key_subpackages_imported():
    assert hasattr(qdlib, 'pricing_foundations')
    assert hasattr(qdlib, 'stochastic_volatility')
