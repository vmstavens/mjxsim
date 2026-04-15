"""Model-building helpers exposed as ``mjxsim.models``."""

from mjxsim._alias import alias_module

_module = alias_module(__name__, "models")
globals().update(_module.__dict__)
