"""Agent implementations exposed as ``mjxsim.agents``."""

from mjxsim._alias import alias_package

_module = alias_package(__name__, "agents")
globals().update(_module.__dict__)
