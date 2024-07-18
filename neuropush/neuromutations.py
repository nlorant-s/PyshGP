import random
from pyshgp.gp.variation import LiteralMutation, VariationOperator
from pyshgp.push.types import PushFloat, PushInt
from pyshgp.push.atoms import Literal
from pyshgp.push.types import PushType
from pyshgp.gp.genome import Genome, GeneSpawner
from pyshgp.tap import tap
from pyshgp.utils import DiscreteProbDistrib, instantiate_using

from abc import ABC, abstractmethod
from typing import Sequence, Union
import math
from numpy.random import choice
import random

class NullMutation(LiteralMutation):
    def __init__(self):
        super().__init__(PushFloat, 0.0)

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value
        return Literal(value=new_value, push_type=PushFloat)

class IntReplacement(VariationOperator):
    """
    Mutation operator that has a chance of mutating a PushInt literal.
    """
    
    def __init__(self, rate=0.2):
        super().__init__(1)
        self.rate = rate
    
    def produce(self, parents, spawner):
        parent = parents[0]
        child = []
        
        for gene in parent:
            if isinstance(gene, Literal) and gene.push_type == PushInt:
                new_value = int(max(1, min(16, random.gauss(gene.value, 3))))
                if random.random() < self.rate:
                    child.append(Literal(value=new_value, push_type=PushInt))
                else:
                    child.append(gene)
            else:
                child.append(gene)
        
        return child

class FloatReplacement(VariationOperator):
    """
    Mutation operator that has a chance of mutating a PushFloat literal.
    """
    
    def __init__(self, rate=0.5):
        super().__init__(1)
        self.rate = rate
    
    def produce(self, parents, spawner):
        parent = parents[0]
        child = []
        
        for gene in parent:
            if isinstance(gene, Literal) and gene.push_type == PushFloat:
                new_value = max(-0.99, min(0.99, random.gauss(gene.value, 0.5)))
                if random.random() < self.rate:
                    child.append(Literal(value=new_value, push_type=PushFloat))
                else:
                    child.append(gene)
            else:
                child.append(gene)
        
        return child