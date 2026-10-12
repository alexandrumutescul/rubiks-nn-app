"""rubiks-nn: a numpy model of the 3x3x3 Rubik's Cube and a seeded dataset generator.

Modules:
    rubiks.cube      facelet-level model; move permutations derived from geometry
    rubiks.cubie     cubie-level model (Kociemba's representation), the independent second model
    rubiks.sampling  random scrambles from SOLVED and uniformly random states
    rubiks.generate  dataset CLI (python -m rubiks.generate)
"""

__version__ = "0.1.0"
