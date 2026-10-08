"""Exact unit conversions, grouped by physical quantity across splits."""
from fractions import Fraction
import random
import re
from bidir.schema import PairInstance
from bidir.domains._common import base_row

NAME = CELL = 'units'
CELL_OPTS = {}
FORWARD_LABEL = 'convert to target unit'
REVERSE_LABEL = 'convert to source unit'
# value_b = scale * value_a + offset; canonical value is in unit_a.
CONVERSIONS = (
    ('length', 'm', 'cm', Fraction(100), Fraction(0)),
    ('mass', 'kg', 'g', Fraction(1000), Fraction(0)),
    ('time', 'h', 'min', Fraction(60), Fraction(0)),
    ('temperature', 'C', 'F', Fraction(9, 5), Fraction(32)),
)
_NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?'
_ANSWER = re.compile(r'^\s*(' + _NUMBER + r')\s+(m|cm|kg|g|h|min|C|F)\s*$')

def render(value):
    if value.denominator == 1:
        return str(value.numerator)
    # Exact finite decimals for our generator; fractions remain allowed for answers.
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 40
        return format(Decimal(value.numerator) / Decimal(value.denominator), 'f').rstrip('0').rstrip('.')

def parse(text):
    match = _ANSWER.fullmatch(text or '')
    if not match:
        return None
    number, unit = match.groups()
    try:
        if '/' in number:
            n, d = number.split('/')
            value = Fraction(n) / Fraction(d)
        else:
            value = Fraction(number)
        return value, unit
    except (ValueError, ZeroDivisionError):
        return None

def content_key(pair):
    meta = pair.meta if hasattr(pair, 'meta') else pair['meta']
    return meta['quantity_key']

def build_pairs(cfg):
    rng = random.Random(int(cfg.get('seed', 17)))
    seen = set(); out = {}
    for split, size_key in [('train','n_train'), ('val','n_val'), ('test','n_test')]:
        rows = []
        for i in range(int(cfg[size_key])):
            dimension, ua, ub, scale, offset = CONVERSIONS[i % len(CONVERSIONS)]
            for _ in range(10000):
                # Quantities are unique across every split; negative values, decimals and zero
                # are valid mathematical conversion inputs, not claims of physical feasibility.
                value = Fraction(rng.randint(-1000000, 1000000), 100)
                key = f'{dimension}:{value}'
                if key not in seen:
                    seen.add(key); break
            else:
                raise ValueError('unique quantity space exhausted')
            rows.append(PairInstance(pair_id=f'units::{split}::{i}', domain=NAME,
                subtask=dimension, side_a=f'{render(value)} {ua}',
                side_b=f'{render(scale * value + offset)} {ub}', split=split,
                meta={'quantity_key':key,'unit_a':ua,'unit_b':ub}))
        out[split] = rows
    return out

def instruction(direction, inst):
    forward = direction == 'forward'
    source = inst['side_a' if forward else 'side_b']
    target_unit = inst['meta']['unit_b' if forward else 'unit_a']
    return (f'Convert {source} to {target_unit}. C and F mean Celsius and Fahrenheit. '
            'Return only the exact numeric value followed by the target unit. '
            'An exact fraction such as 1/3 is allowed. Do not round.')

def augmented_hint(direction):
    return 'Use the exact conversion. Return one number and the requested unit, with no explanation.'

def score_batch(direction, outputs, insts, cfg):
    rows = []
    for output, inst in zip(outputs, insts):
        source = inst['side_a' if direction == 'forward' else 'side_b']
        target = inst['side_b' if direction == 'forward' else 'side_a']
        row = base_row(output, source, target)
        got, want = parse(output), parse(target)
        if want is None:
            raise ValueError('invalid generated gold unit quantity')
        row['off_target'] = int(got is None or got[1] != want[1])
        row['exact_quantity'] = int(got == want)
        row['strict'] = int(got == want and not row['echo'] and not row['empty_output'])
        row['criterion'] = 'exact rational value and requested unit; no echo'
        rows.append(row)
    return rows
