"""Dataset helpers exposed as ``mjxsim.datasets``."""

from mjxsim._alias import alias_package

_module = alias_package(__name__, "datasets")
globals().update(_module.__dict__)
