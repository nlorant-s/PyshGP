import random
from pyshgp.gp.variation import LiteralMutation, AdditionMutation, DeletionMutation, VariationOperator
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.push.atoms import Literal

class NullMutation(LiteralMutation):
    def __init__(self):
        super().__init__(PushFloat, 0.0)

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value
        return Literal(value=new_value, push_type=PushFloat)

class SlightMutation(LiteralMutation):
    def __init__(self):
        super().__init__(PushInt)

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = 5
        return Literal(value=new_value, push_type=PushInt)