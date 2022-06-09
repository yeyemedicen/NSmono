''' Solver utility wrapper '''

def init(inputfile):
    ''' Select solver according to inputfile and return solver instance.

    Args:
        inputfile (str):   path/to/inputfile.yaml

    Returns:
        NSmono Solver object
    '''

    from .codes.problem import problem
    from .codes.solver import solver


    problem = problem(inputfile)
    problem.init()
    solver = solver(problem)

    return solver
