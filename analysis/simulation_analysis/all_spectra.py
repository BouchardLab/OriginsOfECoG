import os
import argparse

def run_power_spectrum(file, ch):
    cmd = 'python power_spectrum.py --nwb ' + file + ' --channel ' + str(ch) #channel doesn't do anything
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and save all spectra')
    parser.add_argument('file', metavar='f', type=str, 
                        help='nwbfile to compute and save spectra to')
    args = parser.parse_args()
    file = args.file
    
    for ch in range(2):
        run_power_spectrum(file, ch)