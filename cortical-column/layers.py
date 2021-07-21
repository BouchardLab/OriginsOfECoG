import numpy as np

# Middle of each layer
layer_depths = {
    1: -82.5,
    2: -239.5,
    3: -490.5,
    4: -762,
    5: -1119.5,
    6: -1732,
}

layer_thicknesses = {
    1: 165.0,
    2: 149.0,
    3: 353.0,
    4: 190.0,
    5: 525.0,
    6: 700.0,
}

layer_divs = np.cumsum([-x for x in layer_thicknesses.values()])
layer_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI']
