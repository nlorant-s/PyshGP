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
from numpy.random import random, choice

class NullMutation(LiteralMutation):
    def __init__(self):
        super().__init__(PushFloat, 0.0)

    def _mutate_literal(self, literal: Literal) -> Literal:
        new_value = literal.value
        return Literal(value=new_value, push_type=PushFloat)

class IntMutation(VariationOperator):
    """Mutates the value of one PushInt literal in the genome.

    Parameters
    ----------
    rate : float
        The probability of applying the mutation to a given PushInt literal.
        Default is 0.1.

    Attributes
    ----------
    rate : float
        The probability of applying the mutation to a given PushInt literal.
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.
    """

    def __init__(self, rate: float = 0.1):
        super().__init__(1)
        self.rate = rate

    @tap
    def produce(self, parents: Sequence[Genome], spawner: GeneSpawner) -> Genome:
        """Produce a child Genome by mutating one PushInt literal.

        Parameters
        ----------
        parents : Sequence[Genome]
            A list containing a single parent Genome.
        spawner : GeneSpawner
            A GeneSpawner that can be used to produce new genes (aka Atoms).

        Returns
        -------
        Genome
            A new Genome with potentially one mutated PushInt literal.
        """
        super().produce(parents, spawner)
        self.checknum_parents(parents)
        new_genome = Genome()
        mutated = False
        
        for atom in parents[0]:
            if isinstance(atom, Literal) and atom.push_type == PushType.INT and random() < self.rate and not mutated:
                new_value = spawner.random_int()
                new_atom = Literal(new_value, PushType.INT)
                new_genome = new_genome.append(new_atom)
                mutated = True
            else:
                new_genome = new_genome.append(atom)
        
        return new_genome