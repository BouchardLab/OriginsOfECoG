"""
Class for performing command line arg parsing, tokenizing, etc.
"""

__author__ = 'Vyassa Baratham <vbaratham@lbl.gov>'

import argparse
import os

from mars import NSE_DATAROOT
from mars.io.tokenizer import Tokenizer
from mars.configs.block_directory import bl
from mars.io import NSENWB

class MarsBaseArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(MarsBaseArgParser, self).__init__(*args, **kwargs)

        self.add_argument('--block', '--blockname', type=str, required=True,
                          help="Block whose configuration to use " + \
                          "(see block_directory.py)")
        self.add_argument('--nwb', type=str, required=False, default=None,
                          help="use this .nwb file instead of looking for one " + \
                          "within the block directory. Required if not passing" + \
                          "--droot")
        self.add_argument('--droot', type=str, required=False, default=NSE_DATAROOT,
                          help="root data directory. Required if not passing --nwb")

        self._args = None
        
    @property
    def args(self):
        if not self._args:
            self.parse_args()
        return self._args

    def parse_args(self):
        self._args = super(MarsBaseArgParser, self).parse_args()
        return self._args

    def nwb_filename(self):
        if self.args.nwb:
            return self.args.nwb
        
        return os.path.join(
            self.args.droot,
            self.args.block.split('_')[0],
            self.args.block,
            '{}.nwb'.format(self.args.block)
        )

    def block_info(self):
        return bl[self.args.block]

    def reader(self):
        # return NWBReader(self.nwb_filename(), block_name=self.args.block)
        return NSENWB.from_existing_nwb(self.args.block, self.nwb_filename())

    def tokenizer(self):
        # TODO: Load the right one based on block directory (when we put that info there)
        return Tokenizer(self.reader())

    
class MarsArgParser(MarsBaseArgParser):
    def __init__(self, *args, **kwargs):
        super(MarsArgParser, self).__init__(*args, **kwargs)

        self.add_argument('--device', type=str, required=True,
                          help="eg 'Poly' or 'ECoG'")

        
class MarsProcessedArgParser(MarsArgParser):
    def __init__(self, *args, **kwargs):
        super(MarsProcessedArgParser, self).__init__(*args, **kwargs)

        self.add_argument('--processed', type=str, required=False, default='Hilb_54bands',
                          help="which preprocessed data to use, " + \
                          "eg. 'Wvlt_4to1200_54band_CAR1' (must be a key " + \
                          "in processing/preprocessed/ in the .nwb file)")
    
# class MarsRawArgParser(MarsArgParser):
#     def __init__(self, *args, **kwargs):
#         super(MarsArgParser, self).__init__(*args, **kwargs)

#         self.add_argument('--raw', type=str, required=True,
#                           help="which raw data to use, " + \
#                           "eg. 'Wvlt_4to1200_54band_CAR1' (must be a key " + \
#                           "in acquisition/Raw/ in the .nwb file)")
    
