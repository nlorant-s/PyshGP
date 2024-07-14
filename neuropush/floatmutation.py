import random
from pyshgp.gp.variation import LiteralMutation
from pyshgp.push.types import PushFloat
from pyshgp.push.atoms import Literal

class FloatMutation(LiteralMutation):
    def __init__(self, rate: float = 0.1, std_dev: float = 0.1):
        super().__init__(PushFloat, rate)
        self.std_dev = std_dev

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value + random.gauss(0, self.std_dev)
        return Literal(value=new_value, push_type=PushFloat)