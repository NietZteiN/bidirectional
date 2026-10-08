"""Post-diagnostic output-contract variant; inherits units semantics and splits."""
from bidir.domains.units import *
from bidir.domains import units as original
NAME = CELL = 'units_explicit'

def build_pairs(cfg):
    splits=original.build_pairs(cfg)
    for rows in splits.values():
        for p in rows:
            p.domain=NAME; p.pair_id=p.pair_id.replace('units::',NAME+'::',1)
    return splits

def instruction(direction,inst):
    target=inst['meta']['unit_b' if direction=='forward' else 'unit_a']
    return (f'Your entire answer must be one number, one space, and the unit {target}. '
            'Do not repeat the input, write an equation, or append other units. '
            'Use m=meters, cm=centimeters, kg=kilograms, g=grams, h=hours, '
            'min=minutes, C=Celsius, F=Fahrenheit. '
            +original.instruction(direction,inst))
