"""Environment implementations exposed as ``mjxsim.envs``."""

from mjxsim._alias import alias_package

_module = alias_package(__name__, "envs")
globals().update(_module.__dict__)
