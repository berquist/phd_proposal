#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import pandas as pd

# import openbabel as ob
# import pybel as pb

labels = pd.read_table('data.txt', sep=' ', header=None)
labels.columns = ['filename', 'rotation']

for idx, ir in enumerate(labels['filename']):
    filename = ir
    # pb.readfile('cml', filename)
