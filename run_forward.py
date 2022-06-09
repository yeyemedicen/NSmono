''' Run a forward simulation.
'''

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

def get_forward_solver():
    ''' Get forward solver from input file '''

    from NSmono import solver

    return solver


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run forward simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputfile', type=str, help='path to YAML input file')
    return parser


if __name__ == '__main__':
    
    inputfile = get_parser().parse_args().inputfile

    solver = get_forward_solver()
    sol = solver.init(inputfile)
    sol.solve()
