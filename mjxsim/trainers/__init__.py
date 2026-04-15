"""Trainer implementations exposed as ``mjxsim.trainers``."""

from mjxsim._alias import alias_package

_module = alias_package(__name__, "trainers")
globals().update(_module.__dict__)
