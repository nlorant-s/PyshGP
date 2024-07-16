import random
from pyshgp.gp.variation import LiteralMutation, AdditionMutation, DeletionMutation, VariationOperator
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.push.atoms import Literal

class FloatMutation(LiteralMutation):
    def __init__(self, rate: float = 0.1, std_dev: float = 0.1):
        super().__init__(PushFloat, rate)
        self.std_dev = std_dev

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value + random.gauss(0, self.std_dev)
        return Literal(value=new_value, push_type=PushFloat)
    
class IntMutation(LiteralMutation):
    def __init__(self, rate: float = 0.1, std_dev: float = 0.1):
        super().__init__(PushInt, rate)
        self.std_dev = std_dev

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value + random.gauss(0, self.std_dev)
        return Literal(value=new_value, push_type=PushInt)
    
class NullMutation(LiteralMutation):
    def __init__(self):
        super().__init__(PushFloat, 0.0)

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value
        return Literal(value=new_value, push_type=PushFloat)
    
class AdditionDeletionMutation(VariationOperator):
    def __init__(self, addition_rate: float = 0.1, deletion_rate: float = 0.1):
        super().__init__(1)  # This operator requires 1 parent
        self.addition_mutation = AdditionMutation(addition_rate)
        self.deletion_mutation = DeletionMutation(deletion_rate)

    def produce(self, parents, spawner):
        # First apply addition mutation
        intermediate = self.addition_mutation.produce(parents, spawner)
        # Then apply deletion mutation to the result
        return self.deletion_mutation.produce([intermediate], spawner)