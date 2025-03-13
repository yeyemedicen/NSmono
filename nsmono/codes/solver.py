'''
MONOLITHIC NAVIER STOKES SOLVER
 
Written by Jeremias Garay L: j.e.garay.labra@rug.nl

'''

from dolfin import *
from common import inout, utils
from petsc4py import PETSc
from ..logger.logger import LoggerBase
from pathlib import Path
import numpy as np
import sys
import pickle
import shutil
import os
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.sparse.linalg import eigsh
#from matspy import spy

def rank0(func):
    ''' Rank 0 decorator: decorated function "does nothing" if rank > 0 '''
    def inner(*args, **kwargs):
        if MPI.rank(MPI.comm_world) == 0:
            func(*args, **kwargs)
    return inner


class Solver(LoggerBase):
    def __init__(self, problem, dump_parameters=True):
        super().__init__()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Initializing')
        self.options = problem.options
        self.inputfile = problem.inputfile
        self.t = 0.
        self._t_write = 0.
        self._t_checkpt = 0.
        self.ndim = problem.ndim
        self.bc_dict = problem.bc_dict
        self.w = problem.w
        self.u0 = problem.u0
        self._using_wk = problem._using_wk
        self._using_mapdd = problem._using_mapdd
        self.forms = problem.forms
        self._diverged = False
        self._optimizing = False
        self._initialized = False
        self.bnds = problem.bnds
        self._is_eigenproblem = problem._is_eigenproblem
        self._is_eigen_cube = problem._is_eigen_cube
        self._is_laplace = problem._is_laplace

        if self._is_eigenproblem:
            self.W = problem.W
            self.Ve = problem.Ve
            self.eigen_matrices = problem.eigen_matrices

        if 'nl_forms' in vars(problem):
            self.nl_forms = problem.nl_forms

        mesh = self.w.function_space().mesh()
        
        V = VectorFunctionSpace(mesh, 'P', 1)
        Q = FunctionSpace(mesh, 'P', 1)
        
        #self.uwrite = Function(self.w.function_space().sub(0).collapse())
        #self.pwrite = Function(self.w.function_space().sub(1).collapse())
        # for use solution as ref interpolation
        self.uwrite = Function(V)
        self.pwrite = Function(Q)
        
        self.uwrite.rename('u', 'velocity')
        self.pwrite.rename('p', 'pressure')

        if self._using_wk:
            self.pi_functions = {}
            for bid, prm in self.bc_dict['windkessel'].items():
                self.pi_functions[bid] = []

        self.mat = {}
        self.vec = {}
        self.mat.update({
                'mass': None, 
                'conv': None,
                'press': None,
                'rhs': None})

        self.vec.update({
                'rhs_const': None,
                'inflow_rhs': None,
                'mapdd_rhs': None,
                'windkessel': None,
                'fnv_rhs': None})

        if dump_parameters:
            self.dump_parameters()

        problem.close_logs()

    @rank0
    def dump_parameters(self):
        ''' Write parameters and git rev hash to files to
        timemarching>write_path if timemarching>write is set.
        '''
        self.logger.info('Copying inputfile to results directory')
        path = self.options['io']['write_path']
        self.logger.info('inputfile: ' + self.inputfile)
        self.logger.info('results dir: ' + path)
        if (os.path.abspath(self.inputfile) ==
                os.path.abspath(path + '/input.yaml')):
            self.logger.info('Same input.yaml, not copying.')
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy2(self.inputfile, path + '/input.yaml')

    def init(self):
        ''' Initialize matrices and solvers '''
        self.init_assembly()
        self.init_solvers()
        self._initialized = True
        if not self._is_eigenproblem:
            self.write_initial_condition()

    def init_assembly(self):
        ''' Initialize, assemble static matrices. '''


        if not self._is_eigenproblem:
            self.mat['diff'] = assemble(self.forms['diff'])
            self.mat['press'] = assemble(self.forms['press'])
            self.mat['mass'] = assemble(self.forms['mass'])
            self.mat['rhs'] = self.mat['mass'].copy()
            # init convection matrix (sparsity pattern) ?
            #self.mat['conv'] = assemble(self.forms['conv'])
            self.mat['conv'] = self.mat['mass'].copy()
        else:
            pass



        if self.forms['neumann']:
            self.vec['rhs_const'] = assemble(self.forms['neumann'])
        
        if 'inflow_rhs' in self.forms:
            self.vec['inflow_rhs'] = assemble(self.forms['inflow_rhs'])
        
        if 'mapdd_rhs' in self.forms:
            self.vec['mapdd_rhs'] = assemble(self.forms['mapdd_rhs'])
        
        if 'windkessel' in self.forms:
            self.vec['windkessel'] = assemble(self.forms['windkessel'])

        if 'fnv_rhs' in self.forms:
            self.vec['fnv_rhs'] = assemble(self.forms['fnv_rhs'])
        
        # done
        self._init_assembly_done = True

    def init_solvers(self):
        self.iterations_ksp = {}
        self.residuals_ksp = {}

        if self._is_eigenproblem:
            self.solver_ns  = PETScSolver(self.options, self._logging_filehandler, is_eigen=True)
        else:
            self.solver_ns  = PETScSolver(self.options, self._logging_filehandler)

        self.iterations_ksp.update({'monolithic': []})
        self.residuals_ksp.update({'monolithic': []})

    def write_initial_condition(self):
        ''' Write initial condition XDMF and HDF5 checkpoints '''
        if (self.options['io']['write_xdmf'] or
                self.options['io']['write_checkpoints']):
            self.logger.info('Writing initial condition')

        self.write_xdmf(t=0.)
        self.write_checkpoint(0)

    def solve(self):
        ''' Solve fractional step scheme '''

        self.init()

        if self._is_eigenproblem:
            self.solve_eigenproblem()
        else:
            dt = self.options['timemarching']['dt']
            T = self.options['timemarching']['T']
            times = np.arange(self.t + dt, T + dt, dt)

            for i, t in enumerate(times):
                it = i + 1
                self.timestep(it, t)
                self.monitor()

                if self._diverged:
                    break

        self.cleanup()

    def restart_timestep(self, state, parameters):
        ''' Restart time step.

        Assign previous state and parameters.
        Compute internal variables from the given state, as required for
        computing the next state.

        Args:
            state (list of Function): list of state functions
            parameters (numpy.ndarray):    array of parameters
        '''
        t0 = self.t
        self.t -= self.options['timemarching']['dt']

        if not state:
            raise Exception('State required for restarting algorithm')

        self.assign_state(state)
        self.assign_parameters(parameters)

        self.t = t0

    def solve_eigenproblem(self):

        self.logger.info('Solving the eigenvalue problem...')

        nmode = 3
        nmode_tot = 25

        
        velbcs = []

        if self._is_eigen_cube:
            velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 1))
            velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 2))
            velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 3))
            velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 4))
            #velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 5))
            #velbcs.append(DirichletBC(self.W.sub(0), Constant((0.0,0.0,0.0)), self.bnds, 6))
        else:
            if self._is_laplace:
                velbcs.append(DirichletBC(self.Ve, Constant(0.0), self.bnds, 1))
            else:
                for bc in self.bc_dict['dirichlet']:
                    velbcs.append(bc)
        
        
        if len(velbcs) == 0:
            modifying_matrices = False
            self.logger.info('Not changing Stokes matrices')
        else:
            modifying_matrices = True
            self.logger.info('Reducing the matrices according to Dirichlet DoFs...')


        # according to highlando
        # Assemble system
        A_tot = assemble(self.eigen_matrices['aa_tot'])
        
        if self.eigen_matrices['ma']:
            M = assemble(self.eigen_matrices['ma'])
            #parameters.linear_algebra_backend = "uBLAS"
            # convert DOLFIN representation to numpy arrays
            mat = as_backend_type(M).mat()
            Ma = sps.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)

        mat = as_backend_type(A_tot).mat()
        Aa_tot = sps.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
        num_v = Aa_tot.shape[0]
        frecuencies = []
        self.logger.info('Size of the system: {}x{}'.format(num_v,num_v))


        #if self._is_laplace:
        if False:
                A_p = PETSc.Mat().createAIJ(size=Aa_tot.shape, csr=(Aa_tot.indptr, Aa_tot.indices, Aa_tot.data))
                AA = PETScMatrix(A_p)

                # solving
                #self.eigensolver = SLEPcEigenSolver(as_backend_type(AA))
                self.eigensolver = SLEPcEigenSolver(as_backend_type(A_tot))
                
                # random stuff
                #self.eigensolver.parameters['spectrum'] = 'smallest magnitude' # div
                self.eigensolver.parameters['spectrum'] = 'smallest real'
                self.eigensolver.parameters['tolerance'] = 1.e-14
                self.eigensolver.parameters['problem_type'] = 'gen_hermitian'
                # settings for neumann
                #self.eigensolver.parameters['problem_type'] = 'pos_gen_non_hermitian'
                #self.eigensolver.parameters['spectrum'] = 'target magnitude'
                #self.eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
                #self.eigensolver.parameters['spectral_shift'] = sys.float_info.epsilon
                self.logger.info('Computing eigenvalues. This could take some time ...')
                self.eigensolver.solve(nmode_tot)
                #k = self.solver_ns.eigensolver.get_number_converged()
                k = self.eigensolver.get_number_converged()
                print('number of converged solutions: {}'.format(k))
                uaux = Function(self.Ve)
                uaux.rename('u', 'velocity')

                for k in range(nmode_tot):
                    if k == 0:
                        self._xdmf_u = XDMFFile(self.options['io']['write_path'] + '/u.xdmf')
                    w, c, wx, cx = self.eigensolver.get_eigenpair(k)
                    frecuencies.append(w)
                    uaux.vector()[:] = wx
                    # normalizing vector
                    u_norm = norm(uaux)
                    uaux.vector()[:] = uaux.vector()[:]/u_norm
                    self._xdmf_u.write(uaux, float(k))
                    # saving checkpoints
                    path = (self.options['io']['write_path'] + '/checkpoint/{i}/'.format(i=k))
                    comm = self.w.function_space().mesh().mpi_comm()
                    inout.write_HDF5_data(comm, path + '/u.h5', uaux, '/u', t=k)

                # saving the frecuencies
                np.savetxt(self.options['io']['write_path'] + '/eigenvalues.txt', frecuencies)
                self.logger.info('Done')
        else:
            if modifying_matrices:
                auxu = np.zeros((num_v,1))
                bcinds = []

                for bc in velbcs:
                    bcdict = bc.get_boundary_values()
                    bcdict_lst = list(bcdict.keys())
                    bcdictval_lst = list(bcdict.values())
                    auxu[bcdict_lst,0] = bcdictval_lst
                    bcinds.extend(bcdict.keys())

                bcinds = set(bcinds)
                bcinds = [c for c in bcinds]
                invinds = np.setdiff1d(range(num_v),bcinds).astype(np.int32)
                # extract the inner nodes equation coefficients
                Mc = Ma[invinds,:][:,invinds]
                Ac_tot = Aa_tot[invinds,:][:,invinds]
                row_maskA = np.arange(Ac_tot.shape[0] - 1)
                col_maskA = np.arange(Ac_tot.shape[1] - 1)
                row_maskM = np.arange(Mc.shape[0] - 1)
                col_maskM = np.arange(Mc.shape[1] - 1)
                Ac_tot = Ac_tot[row_maskA][:, col_maskA]
                Mc = Mc[row_maskM][:, col_maskM]
                ValEig = nmode_tot
                sigmashift = 0 #Garding shift
                E1, V1=eigsh(Ac_tot, ValEig, Mc, sigmashift,'LM')
                ind=np.argsort(E1)[::-1] #Sort descending
                ind = ind=np.argsort(E1)
                E1 = E1[ind]
                V1 = V1[:,ind] #orden de las matrices con los indices de E1 ordenadado
                sol=np.zeros((num_v,ValEig), dtype=np.float64)
                np.savetxt(self.options['io']['write_path'] + '/eigenvalues.txt', E1)
                
                if self._is_laplace:
                    uaux = Function(self.Ve)
                    uaux.rename('u', 'velocity')


                for k in range(ValEig):
                    sol[invinds[:len(invinds)-1],k] = V1[:,k]
                    if k == 0:
                        self._xdmf_u = XDMFFile(self.options['io']['write_path'] + '/u.xdmf')
                        self._xdmf_p = XDMFFile(self.options['io']['write_path'] + '/p.xdmf')
                        

                    # saving checkpoints
                    path = (self.options['io']['write_path']
                            + '/checkpoint/{i}/'.format(i=k))
                    comm = self.w.function_space().mesh().mpi_comm()


                    if self._is_laplace:
                        uaux.vector()[invinds[:len(invinds)-1]] = V1[:,k]
                        self._xdmf_u.write(uaux, float(k))
                        inout.write_HDF5_data(comm, path + '/u.h5', uaux, '/u', t=k)
                        
                    else:
                        self.w.vector()[invinds[:len(invinds)-1]] = V1[:,k]   
                        (u, p) = self.w.split()
                        u.rename('u', 'velocity')
                        p.rename('p', 'pressure')
                        LagrangeInterpolator.interpolate(self.uwrite,u)
                        LagrangeInterpolator.interpolate(self.pwrite,p)
                        
                        self._xdmf_u.write(u, float(k))
                        self._xdmf_p.write(p, float(k))
                        
                        inout.write_HDF5_data(comm, path + '/u.h5', self.uwrite, '/u', t=k)
                        inout.write_HDF5_data(comm, path + '/p.h5', self.pwrite, '/p', t=k)
                        


                self.logger.info('Done')

            else:

                A_p = PETSc.Mat().createAIJ(size=Aa_tot.shape, csr=(Aa_tot.indptr, Aa_tot.indices, Aa_tot.data))
                B_p = PETSc.Mat().createAIJ(size=Ma.shape, csr=(Ma.indptr, Ma.indices, Ma.data))
                AA = PETScMatrix(A_p)
                BB = PETScMatrix(B_p)

                # solving
                self.eigensolver = SLEPcEigenSolver(as_backend_type(AA), as_backend_type(BB))
                #self.eigensolver.parameters["solver"] = "krylov-schur"
                #self.eigensolver.parameters['spectrum'] = 'smallest magnitude'
                #self.eigensolver.parameters['spectrum'] = 'smallest real'
                #self.eigensolver.parameters['tolerance'] = 1.e-14
                #self.eigensolver.parameters['problem_type'] = 'gen_hermitian'
                self.eigensolver.parameters['problem_type'] = 'pos_gen_non_hermitian'
                self.eigensolver.parameters['spectrum'] = 'target magnitude'
                self.eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
                self.eigensolver.parameters['spectral_shift'] = sys.float_info.epsilon
                self.logger.info('Computing eigenvalues. This could take some time ...')
                self.eigensolver.solve(nmode_tot)


                #k = self.solver_ns.eigensolver.get_number_converged()
                k = self.eigensolver.get_number_converged()
                print('number of converged solutions: {}'.format(k))

                for i in range(k):
                    if i == nmode:
                        #w, c, wx, cx = self.solver_ns.eigensolver.get_eigenpair(i)
                        w, c, wx, cx = self.eigensolver.get_eigenpair(i)
                        
                        self.logger.info('{}th eigenvalue: {}'.format(i+1,w))
                        self.write_xdmf_and_check(uvec = wx, pvec=None , t=i)
                        #self.write_xdmf_and_check(uvec = None, pvec = wx , t=i)

    def timestep(self, i=0, t=None, state=None, parameters=None,
                restart=False, observations=None):
        ''' Timestep interface for ROUKF algorithm.

        Args (optional):
            i (int):        iteration count
            t (float)       current (new) time
            state (list):   state variables (list of fenics functions)
            parameters (list):   parameters (list of numbers)
            restart (bool): precomputes internal variables from state
            observations (list):   list of Functions to save observations
        '''
        
        if restart and i == 0:
            self.init_state(state)
            
        if not self._initialized:
            self.init()

        if t:
            self.t = t

        if restart:
            self.restart_timestep(state, parameters)

        self.solve_NS()

        if state:
            self.update_state(state)

        if observations:
            self.observation(observations)

        self.write_timestep(i)

    def init_state(self,state):
        ''' initialize the state variables according inputfile values'''

        if self._using_wk:
            enum_wk = enumerate(self.bc_dict['windkessel'].items())
            for k, (bid, prm) in enum_wk:
                if abs(float(prm['C'])) > 1e-14:
                    value = float(prm['pi'])
                    state[k+1].vector()[:] = value

    def init_observations(self):
        ''' Initialize observations for ROUKF.

        Reads mesh or meshes, creates function space(s) and initializes and
        returns function(s).

        Returns:
            fun_lst     list of measurement/observation functions, for each
                        given mesh
        '''
        measurement_lst = self.options['estimation']['measurements']
        if not isinstance(measurement_lst, list):
            measurement_lst = [measurement_lst]

        mesh_lst = [meas['mesh'] for meas in measurement_lst]

        if 'fe_degree' in self.options['estimation']['measurements'][0]:
            degree = self.options['estimation']['measurements'][0]['fe_degree']
        else:
            degree = 1

        for measurement in measurement_lst[1:]:
            if not degree == measurement['fe_degree']:
                raise Exception('fe_degree must the the same for all '
                                'measurements!')


        veldir_given = []
        for meas in measurement_lst:
            if 'velocity_direction' in meas:
                direction = meas['velocity_direction']
                if isinstance(direction, list):
                    if not all(isinstance(d, (int, float)) for d in direction):
                        raise Exception('velocity_direction should be list of '
                                        'numbers or False/None omitted. Is: {}'
                                        .format(direction))

                    veldir_given.append(True)
                else:
                    assert not direction
                    veldir_given.append(False)
            else:
                veldir_given.append(False)

        all_scalar = all([flag is True for flag in veldir_given])
        all_vector = all([flag is False for flag in veldir_given])

        if not (all_scalar or all_vector):
            raise Exception('All measurements need to be given as 3D (no '
                            'velocity_direction) or 1D, mixing is not '
                            'supported')

        if degree == 1:
            element_family = 'P'
        elif degree == 0:
            element_family = 'DG'
        else:
            raise Exception('Unsupported measurement FE degree: {}'
                            .format(degree))

        if all_scalar:
            self._observation_aux_assigner = []

        fun_lst = []
        fun_aux_lst = []
        V_aux = None
        for meshfile in mesh_lst:
            mesh, _, _ = inout.read_mesh(meshfile)
            if all_scalar:
                V = FunctionSpace(mesh, element_family, degree)
                # 1. interpolate velocity vector onto measurement grid (V_aux)
                # 2. perform component projection in the measurement space
                # ---> need to store both scalar and vector spaces
                V_aux = VectorFunctionSpace(mesh, element_family, degree)
                fun_aux_lst.append(Function(V_aux))
                # fun_aux_lst.append([Function(V) for i in range(self.ndim)])
                self._observation_aux_assigner.append(
                    FunctionAssigner([V]*self.ndim, V_aux))
            else:
                V = VectorFunctionSpace(mesh, element_family, degree)
            fun_lst.append(Function(V))


        self._observation_fun_aux_lst = fun_aux_lst
        return fun_lst

    def init_parameters(self):
        ''' ROUKF interface: Initialize parameters.

        Returns:
            tuple: tuple containing

                * theta_arr (numpy.ndarray):  numpy array of initial conditions
                  of parameters, in correct order
                * theta_sd_arr (numpy.ndarray):  numpy array with corresp.
                  standard deviations
        '''
        bc_param_lst = self.options['estimation']['boundary_conditions']

        self.theta_internal = []
        theta_sd_lst = []
        theta_arr = []

        for bc in bc_param_lst:
            if bc['type'] == 'dirichlet':
                # find DBCs on the corresponding boundary with expression
                i_lst = []
                th_index = []
                for expr_dict in self.bc_dict['dbc_expressions'].values():

                    if expr_dict['id'] == bc['id']:
                        if not bc['id'] in i_lst:
                            # first time visiting this boundary:
                            for prm in bc['parameters']:
                                assert prm in \
                                    expr_dict['expression'].user_parameters

                                self.theta_internal.append({
                                    'expression_lst': [
                                        expr_dict['expression']],
                                    'parameter': prm
                                })
                                theta_arr.append(
                                    expr_dict['expression'].user_parameters[
                                        prm]
                                )
                                i_lst.append(bc['id'])
                                th_index.append(len(self.theta_internal) - 1)

                        else:
                            i = i_lst.index(bc['id'])
                            if (not self.theta_internal[th_index[i]]
                                    ['parameter'] in bc['parameters']):
                                raise Exception('Parameters dont match!')

                            self.theta_internal[th_index[i]] \
                                ['expression_lst'].append(
                                    expr_dict['expression'])


                if isinstance(bc['initial_stddev'], list):
                    assert (isinstance(bc['parameters'], list) and
                            len(bc['parameters']) == len(bc['initial_stddev']))
                    theta_sd_lst.extend(bc['initial_stddev'])
                else:
                    assert isinstance(bc['parameters'], str), (
                        'Conflichting options: Dimension of expression '
                        'parameters and initial_stddev must match!')
                    theta_sd_lst.append(bc['initial_stddev'])

            elif bc['type'] == 'inflow':
                # find DBCs on the corresponding boundary with expression
                i_lst = []
                th_index = []
                for bid,expr_dict in self.bc_dict['inflow'].items():
                    if bid == bc['id']:
                        if not bc['id'] in i_lst:
                            # first time visiting this boundary:
                            for prm in bc['parameters']:
                                assert prm in expr_dict['waveform'].user_parameters

                                self.theta_internal.append({
                                    'expression_lst': [
                                        expr_dict['waveform']],
                                    'parameter': prm
                                })
                                theta_arr.append(
                                    expr_dict['waveform'].user_parameters[
                                        prm]
                                )
                                i_lst.append(bc['id'])
                                th_index.append(len(self.theta_internal) - 1)

                        else:
                            i = i_lst.index(bc['id'])
                            if (not self.theta_internal[th_index[i]]
                                    ['parameter'] in bc['parameters']):
                                raise Exception('Parameters dont match!')

                            self.theta_internal[th_index[i]] \
                                ['expression_lst'].append(
                                    expr_dict['waveform'])


                if isinstance(bc['initial_stddev'], list):
                    assert (isinstance(bc['parameters'], list) and
                            len(bc['parameters']) == len(bc['initial_stddev']))
                    theta_sd_lst.extend(bc['initial_stddev'])
                else:
                    assert isinstance(bc['parameters'], str), (
                        'Conflichting options: Dimension of expression '
                        'parameters and initial_stddev must match!')
                    theta_sd_lst.append(bc['initial_stddev'])

            elif bc['type'] == 'windkessel':
                if isinstance(bc['parameters'], str):
                    opt_lst = [bc['parameters']]
                elif isinstance(bc['parameters'], list):
                    opt_lst = bc['parameters']
                else:
                    raise Exception('{} not supported'.format(
                                    type(bc['parameters'])))
                
                bid = bc['id']
                wk_dict = self.bc_dict['windkessel']
                assert bid in wk_dict.keys(), 'BC id not in windkessel'

                for prm in opt_lst:
                    assert prm in ('R_d', 'R_p', 'C','L'), f'prm {prm}'
                    self.theta_internal.append(wk_dict[bid][prm])
                    theta_arr.append(float(self.theta_internal[-1]))

                if not isinstance(bc['initial_stddev'], list):
                    opt_std = [bc['initial_stddev']]
                else:
                    opt_std = bc['initial_stddev']

                if len(opt_std) == len(opt_lst):
                    theta_sd_lst.extend(opt_std)
                else:
                    raise Exception('Required more stddevs')

            elif bc['type'] == 'mapdd':

                bid = bc['id']
                mdict = self.bc_dict['mapdd'][bid]
                self.theta_internal.append(mdict['l'])
                theta_arr.append(float(self.theta_internal[-1]))
                opt_std = [bc['initial_stddev']]
                theta_sd_lst.extend(opt_std)

            else:
                raise NotImplementedError('BC type "{}" not yet supported for '
                                          'optimization'.format(bc['type']))
        
        theta_arr = np.array(theta_arr)
        theta_sd_arr = np.array(theta_sd_lst)

        return theta_arr, theta_sd_arr

    def observation(self, Xobs_lst):
        ''' Compute observation by applying the observation operator to the
        state, H(X).

        Args:
            Xobs_lst    list of receiving measurement functions
        '''
        if not self._observation_fun_aux_lst:
            Xobs_aux_lst = [None]*len(Xobs_lst)
        else:
            Xobs_aux_lst = self._observation_fun_aux_lst

        LI = LagrangeInterpolator
        for i, (Xobs, Xobs_aux) in enumerate(zip(Xobs_lst, Xobs_aux_lst)):

            if Xobs_aux:
                # Xobs is scalar, Xobs_aux vector

                direction = (self.options['estimation']['measurements'][i]
                             ['velocity_direction'])

                # handle cartesian component selection manually for performance
                if direction.count(0) == 2 and direction.count(1) == 1:
                    LI.interpolate(Xobs, self.w.sub(0).sub(direction.index(1)))

                else:
                    assert not Xobs.value_shape(), 'Xobs is not a scalar'
                    # normalize projection direction
                    direction = np.array(direction, dtype=float)
                    direction /= np.sqrt(np.dot(direction, direction))
    
                    LI.interpolate(Xobs_aux, self.w.sub(0))

                    # This is faster than simply Xobs_aux.split(True) !
                    Xobs_aux_i = [Xobs] + [Xobs.copy(True) for i in
                                           range(self.ndim - 1)]
                    self._observation_aux_assigner[i].assign(Xobs_aux_i,
                                                             Xobs_aux)
                    Xobs.vector()[:] *= direction[0]
                    for Xi, d in zip(Xobs_aux_i[1:], direction[1:]):
                        if d:
                            Xobs.vector().axpy(d, Xi.vector())
            else:
                LI.interpolate(Xobs, self.w.sub(0))

    def assign_state(self, state):
        ''' ROUKF interface: Update instance solution functions from state
        variable (inverse to update_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return


        assign(self.w.sub(0),state[0])
        
        if self._using_wk:
            enum_wk = enumerate(self.bc_dict['windkessel'].items())
            for k, (bid, prm) in enum_wk:
                # adding wk pressure if C != 0
                if abs(float(prm['C'])) > 1e-14:
                    value = state[k+1].vector().get_local()
                    mpi_comm = self.w.function_space().mesh().mpi_comm()

                    if mpi_comm.Get_size() > 1:
                        # parallel -- maybe not the best solution, but works
                        # we don't know which proc owns the single dof of the
                        # windkessel state 'real' function space, so gather from
                        # all procs on root=0 and filter
                        value_gathered = mpi_comm.gather(value, root=0)
                        if mpi_comm.Get_rank() == 0:
                            value = list(
                                filter(lambda x: x.size > 0, value_gathered)
                            )
                            assert len(value) == 1
                            value = value[0]
                        else:
                            value = np.empty(1)
                        mpi_comm.Bcast(value, root=0)

                    prm['pi'].assign(Constant(float(value)))

    def update_state(self, state):
        ''' ROUKF interface: update state variables from solution functions
        (inverse to assign_state).

        Args:
            state       list of state variables
        '''
        if not state:
            return

        assign(state[0],self.w.sub(0))

        if self._using_wk:
            enum_wk = enumerate(self.bc_dict['windkessel'].items())
            # update wk part of the state
            for k, (bid, prm) in enum_wk:
                if abs(float(prm['C'])) > 1e-14:
                    value = float(prm['pi'])
                    state[k+1].vector()[:] = value

    def assign_parameters(self, parameters):
        ''' ROUKF interface: Update PDE parameters from ROUKF.

        Args:
            parameters   list of parameters or None
        '''
        if parameters is None:
            return

        if not hasattr(self, 'theta_internal'):
            raise Exception('Need to call init_parameters() before '
                            'assign_parameters()!')

        assert len(self.theta_internal) == len(parameters)
        for th_old, th_new in zip(self.theta_internal, parameters):
           
            if isinstance(th_old, Constant):
                th_old.assign(th_new)

            elif isinstance(th_old, dict):
                # parameters are expression parameters
                expr_lst = th_old['expression_lst']
                prms = th_old['parameter']

                for expr in expr_lst:
                    expr.user_parameters[prms] = th_new

                    for dbc, dict_ in (self.bc_dict['dbc_expressions']
                                       .items()):

                        if dict_['expression'] == expr:
                            self.project_enriched_dbc(dbc, expr)
                            break

            else:
                raise Exception('Parameter type not recognized')

    def assemble_NS(self, is_eigenproblem = False):
        ''' Assemble changing matrices '''
        # Note: Matrices are stored into mat['conv'] and mat['rhs']
        # A: system matrix. Assemble convection and add mass/diffusion


        if is_eigenproblem:
            pass
        else:
            A = self.mat['conv']
            assemble(self.forms['conv'], tensor=A)

            A.axpy(1, self.mat['mass'], True)
            A.axpy(1., self.mat['diff'], True)
            A.axpy(1., self.mat['press'], True)
            
            if 'supg_time' in self.forms and self.forms['supg_time']:
                assemble(self.forms['supg_time'], tensor=self.mat['rhs'])
                A.axpy(1., self.mat['rhs'], True)
                self.mat['rhs'].axpy(1., self.mat['mass'], True)

            [bc.apply(A) for bc in self.bc_dict['dirichlet']]


        return A

    def build_rhs(self):
        ''' Build RHS vector'''

        x = self.w.vector()
        b = self.mat['rhs']*x

        if self.vec['rhs_const']:
            b.axpy(1.0, self.vec['rhs_const'])

        # adding inflow right hand side
        if self.vec['inflow_rhs']:
            self.vec['inflow_rhs'] = assemble(self.forms['inflow_rhs'])
            b.axpy(1.0, self.vec['inflow_rhs'])

        # adding mapdd right hand side
        if self.vec['mapdd_rhs']:
            self.vec['mapdd_rhs'] = assemble(self.forms['mapdd_rhs'])
            b.axpy(1.0, self.vec['mapdd_rhs'])

        # adding windkessel right hand side
        if self.vec['windkessel']:
            self.vec['windkessel'] = assemble(self.forms['windkessel'])
            b.axpy(-1.0, self.vec['windkessel'])

        # adding fnv right hand side
        #if self.vec['fnv_rhs']:
        #    self.vec['fnv_rhs'] = assemble(self.forms['fnv_rhs'])
        #    b.axpy(1.0, self.vec['fnv_rhs'])

        [bc.apply(b) for bc in self.bc_dict['dirichlet']]

        return b

    def solve_NS(self):
        ''' Solve tentative velocity PDE '''

        self.logger.info('Solve NS monolithic')
        self.update_bcs()

        assign(self.u0, self.w.sub(0))

        if self._using_wk:
            self.solve_windkessel()

        A = self.assemble_NS()
        #self.solver_u_ten.set_operator(A)

        b = self.build_rhs()

        self.solver_ns.solve(A, self.w.vector(), b)
        #self.solver_ns.solve(self.w.vector(), b)
        

        if self.solver_ns.conv_reason < 0:
            self.logger.error('Solver DIVERGED ({})'.
                              format(self.solver_ns.conv_reason))
            if len(self.iterations_ksp['monolithic']) > 0:
                self._diverged = True

        self.iterations_ksp['monolithic'].append(self.solver_ns.iterations)
        self.residuals_ksp['monolithic'].append(self.solver_ns.residuals)

    def update_bcs(self):
        ''' Update time dependent boundary conditions. '''
        for bc, dict_ in self.bc_dict['dbc_expressions'].items():
            expr = dict_['expression']
            if 't' in expr.user_parameters:
                expr.t = float(self.t)
                self.project_enriched_dbc(bc, expr)
        
        for key, dict_ in self.bc_dict['inflow'].items():
            if 'flow_func' in dict_:
                flow_func = dict_['flow_func']
                flow_upd = flow_func(self.t)
                dict_['waveform'].assign(Constant(flow_upd))
            else:
                waveform = self.bc_dict['inflow'][key]['waveform']
                waveform.t = float(self.t)

    def solve_windkessel(self):
        ''' Solve windkessel
        '''

        ds = Measure('ds', domain=self.w.function_space().mesh(),
                        subdomain_data=self.bnds)
        n = FacetNormal(self.w.function_space().mesh())
        

        for bid, prm in self.bc_dict['windkessel'].items():

            dt = float(self.options['timemarching']['dt'])
            R_d = float(prm['R_d'])
            R_p = float(prm['R_p'])
            C = float(prm['C'])
            L = float(prm['L'])

            alpha = R_d*C/(R_d*C + dt)
            beta = R_d*(1 - alpha)
            gamma = R_p + beta
            delta = gamma + L/dt
            eta = -L/dt

            Q = assemble(dot(self.u0,n)*ds(bid))

            pi0 = float(prm['pi0'])
            pi = float(prm['pi'])
            Q0 = float(prm['Q0'])
            pi_upd = alpha*pi0 + beta*Q
            Pl_upd = alpha*pi0 + delta*Q + eta*Q0
            
            self.pi_functions[bid].append(pi)
            prm['pi0'].assign(Constant(pi))
            prm['Q0'].assign(Constant(Q))
            prm['pi'].assign(Constant(pi_upd))
            prm['Pl'].assign(Constant(Pl_upd))

    def project_enriched_dbc(self, bc, expr):
        if utils.is_enriched(bc.function_space()):
            V = bc.function_space()
            bc.set_value(project(expr, V))

    def monitor(self , it=0, t=0):
        ''' Solution monitor, output interface.

        Args:
            it (int):   iteration count
            t (float): current time (required for compatibility with ROUKF
        '''
        # self._writeout:
        # 0: not written, 1: wrote xdmf, +2: wrote checkpoint
        # => *: xdmf, **: checkpoint, ***: both

        if self.options['timemarching']['report']:
            self.logger.info('t = {t:.{width}f} \t({T})'
                             .format(t=self.t,
                                     T=self.options['timemarching']['T'],
                                     width=6))

    def cleanup(self):
        ''' Cleanup '''

        if self._using_wk:
            with open(self.options['io']['write_path'] + '/pi_functions.pickle', 'wb') as handle:
                pickle.dump(self.pi_functions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.close_xdmf()
        self.close_logs()

    def write_timestep(self, i):
        ''' Combined checkpoint and XDMF write out.

        Args:
            i       (int)  iteration count for checkpoint index
            update  (bool) specify which velocity to write (ALE)

        '''
        tol = 1e-8
        writeout = 0

        T = self.options['timemarching']['T']

        if self.options['io']['write_xdmf']:

            write_dt = self.options['timemarching']['write_dt']

            if ((self.t > self._t_write + write_dt - tol)
                    or (self.t >= T - tol)):
                # if (time for write) or (first) or (last)
                self._t_write = self.t
                writeout = 1
                self.write_xdmf()

        if self.options['io']['write_checkpoints']:
            checkpt_dt = self.options['timemarching']['checkpoint_dt']

            if ((self.t > self._t_checkpt + checkpt_dt - tol)
                    or (self.t >= T - tol)):
                # if (time for checkpoint) or (last)

                self._t_checkpt = self.t
                writeout += 2
                self.write_checkpoint(i)

        self._writeout = writeout

    def write_xdmf_and_check(self, uvec = None, pvec= None, t=None):
        if not self.options['io']['write_xdmf']:
            pass
        else:
            if not t:
                t = self.t

            if (not (hasattr(self, '_xdmf_u') or hasattr(self, '_xdmf_p'))
                    or not (self._xdmf_u or self._xdmf_p)):
                self._xdmf_u = XDMFFile(self.options['io']['write_path']
                                        + '/u.xdmf')
                self._xdmf_p = XDMFFile(self.options['io']['write_path']
                                        + '/p.xdmf')
                self._xdmf_u.parameters['rewrite_function_mesh'] = False
                self._xdmf_u.parameters['functions_share_mesh'] = True
                self._xdmf_p.parameters['rewrite_function_mesh'] = False

            #u = Function(self.w.function_space().sub(0).collapse())
            #p = Function(self.w.function_space().sub(1).collapse())
            #assign(u, self.w.sub(0))
            #assign(p, self.w.sub(1))
            
            (u, p) = self.w.split()
            u.rename('u', 'velocity')
            p.rename('p', 'pressure')

            #u = Function(self.V)
            #u.rename('u', 'velocity')
            
            if uvec is not None:
                u.vector()[:] = uvec
                self._xdmf_u.write(u, float(t))

            if pvec:
                pe = Function(self.Ve)
                pe.rename('lap', 'lap')
                pe.vector()[:] = pvec
                self._xdmf_p.write(pe, float(t))

            
        #if not self.options['io']['write_checkpoints']:
        if True:
            pass
        else:
            path = (self.options['io']['write_path']
                    + '/checkpoint/{i}/'.format(i=t))

            comm = self.w.function_space().mesh().mpi_comm()
            
            #u = Function(self.w.function_space().sub(0).collapse())
            #p = Function(self.w.function_space().sub(1).collapse())
            #assign(u, self.w.sub(0))
            #assign(p, self.w.sub(1))

            (u, p) = self.w.split()

            u.vector()[:] = uvec
            if pvec:
                p.vector()[:] = pvec

            LagrangeInterpolator.interpolate(self.uwrite,u)
            LagrangeInterpolator.interpolate(self.pwrite,p)
            
            #u.rename('u', 'velocity')
            #p.rename('p', 'pressure')

            inout.write_HDF5_data(comm, path + '/u.h5', self.uwrite, '/u',
                                    t=self.t)
            inout.write_HDF5_data(comm, path + '/p.h5', self.pwrite, '/p', t=self.t)

    def write_xdmf(self, t=None):
        ''' Write solution to XDMF files. If file objects have not been
        created, initialize. This works for steady and unsteady solvers with
        timestepping output.

        Args:
            t       (optional) time of solution
        '''
        if not self.options['io']['write_xdmf']:
            return

        if not t:
            t = self.t

        if (not (hasattr(self, '_xdmf_u') or hasattr(self, '_xdmf_p'))
                or not (self._xdmf_u or self._xdmf_p)):
            self._xdmf_u = XDMFFile(self.options['io']['write_path']
                                    + '/u.xdmf')
            self._xdmf_p = XDMFFile(self.options['io']['write_path']
                                    + '/p.xdmf')
            self._xdmf_u.parameters['rewrite_function_mesh'] = False
            self._xdmf_u.parameters['functions_share_mesh'] = True
            self._xdmf_p.parameters['rewrite_function_mesh'] = False

        #u = Function(self.w.function_space().sub(0).collapse())
        #p = Function(self.w.function_space().sub(1).collapse())
        #assign(u, self.w.sub(0))
        #assign(p, self.w.sub(1))
        
        (u, p) = self.w.split()
        u.rename('u', 'velocity')
        p.rename('p', 'pressure')

        self._xdmf_u.write(u, float(t))
        self._xdmf_p.write(p, float(t))

    def write_checkpoint(self, i):
        ''' Write HDF5 checkpoint of u, p to <write_path>/checkpoints folder.

        Args:
            i       (int)  iteration count
            update  (bool) specifies which velocity to store (in ALE)
        '''
        if not self.options['io']['write_checkpoints']:
            return

        path = (self.options['io']['write_path']
                + '/checkpoint/{i}/'.format(i=i))

        comm = self.w.function_space().mesh().mpi_comm()
        

        #uwrite = Function(self.w.function_space().sub(0).collapse())
        #pwrite = Function(self.w.function_space().sub(1).collapse())
        #assign(u, self.w.sub(0))
        #assign(p, self.w.sub(1))
        #assign(self.uwrite, self.w.sub(0))
        #assign(self.pwrite, self.w.sub(1))


        (u, p) = self.w.split()
        #u, p = self.w.split()
        
        #assign(self.uwrite, u)
        #assign(self.pwrite, p)
        LagrangeInterpolator.interpolate(self.uwrite,u)
        LagrangeInterpolator.interpolate(self.pwrite,p)

        inout.write_HDF5_data(comm, path + '/u.h5', self.uwrite, '/u',t=self.t)
        inout.write_HDF5_data(comm, path + '/p.h5', self.pwrite, '/p', t=self.t)


    def close_xdmf(self) -> None:
        ''' close XDMF Files '''
        if hasattr(self, '_dont_close_xdmf') and self._dont_close_xdmf:
            # a little hacky, I admit
            return

        if hasattr(self, '_xdmf_u'):
            del self._xdmf_u
        if hasattr(self, '_xdmf_p'):
            del self._xdmf_p

    def get_state_functionspaces(self) -> None:
        ''' Return function spaces which are related to state variables of the problem. These are:
            - monolithic (for reference): X = u_n

        If windkessel boundaries are included, they belong to the state.

        Returns:
            list of function spaces
        '''
        

        W_lst = [self.w.function_space().sub(0).collapse()]

        for bc in self.options['boundary_conditions']:
            type_ = bc.get('type', None)
            prms = bc.get('parameters', dict())
            C_ = prms['C'] if 'C' in prms else None
            if type_ == 'windkessel' and C_:
                R = FunctionSpace(self.w.function_space().mesh(), 'R', 0)
                W_lst.append(R)

        return W_lst



class PETScSolver(LoggerBase):
    ''' Solver class that handles preconditioned Krylov solvers
    '''

    def __init__(self, options, logging_fh=None, is_eigen = False):
        super().__init__()

        self._is_eigenproblem = is_eigen
        if self._is_eigenproblem:
            self.eigensolver = None

        self._logging_filehandler = logging_fh

        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.options_global = options

        self.residuals = []
        self.iterations = None
        self.conv_reason = 0
        self.timing = {'ksp': 0, 'pc': 0}

        if self._use_petsc():
            self.logger.info('Initializing PETSc Krylov solver')
            inputfile = options['linear_solver']['inputfile']
            self.param = inout.read_parameters(inputfile)
            self._reuse_ksp = False
            # this is not necessary !
            self._reuse_Pmat = False    # test this
            self._set_options()
            self.create_ksp()
            self.create_pc()
        else:
            if not self._is_eigenproblem:
                if not options['linear_solver']['method']:
                    raise Exception('Specify inbuilt solver (lu, mumps) or input '
                                    'file for PETSc!')
                self.logger.info('using inbuilt linear solver')
                self.logger.warn('prefer MUMPS via PETSc interface!')
            else:
                self.logger.info('using SLEPc library')

    def __del__(self):
        ''' Clean up PETScOptions '''
        self.close_logs()
        PETScOptions.clear()
        type(self)._initialized = False

    def _set_options(self):
        self.pc_name = self.param['config_name']
        self.petsc_options = [s.split() for s in self.param['petsc_options']]
        self.logger.info('  PC: {0}'.format(self.pc_name))
        self.logger.info('  Setting PETScOptions:')
        for popt in self.petsc_options:
            self.logger.info('  -' + ' '.join(popt))
            PETScOptions.set(*popt)

    def create_ksp(self):
        ''' Create KSP instance and set options '''
        self.logger.info('  Creating KSP from options')
        self.ksp = PETSc.KSP().create()
        if not hasattr(self, 'petsc_options') or self.petsc_options is None:
            self._set_options()
        self.ksp.setFromOptions()

    def create_pc(self):
        self.fstype = self._get_fieldsplit_type()
        # assert self.fstype, 'fieldsplit type not recognized'
        t0 = Timer('PETScSolver Build PC')
        self.logger.info('  Creating PC of type: {}'.format(self.fstype))
        if self.fstype is None:
            self.logger.info('No FieldSplit. Creating solver from options.')
        elif self.fstype in ('SCHUR_USER_DIAG', 'SCHUR_USER_FULL',
                             'SCHUR_USER_TRI'):
            self._create_user_schurPC()
        elif self.fstype in ('SCHUR_LSC', 'SCHUR'):
            self._create_alg_schurPC()
        elif self.fstype in ('ADD', 'MULT'):
            self._create_Pmat()
        elif self.fstype == 'MG_FIELDSPLIT':
            pass
            # self.logger.info('Setting FieldSplit IS for coupled MG')
            # self._set_fieldsplit_IS()
        elif self.fstype == 'LU':
            self._create_direct_solver()
        else:
            raise Exception('Fieldsplit type not supported!  {}'
                            .format(self.fstype))
        self.timing['pc'] = t0.stop()
        self.ksp.getPC().setReusePreconditioner(self._reuse_Pmat)

    def solve(self, A, x, b):
        ''' Wrap PETSc and "inbuilt" FEniCS solvers.

        Args:
            A   PETScMatrix
            x   PETScVector
            b   PETScVector
        '''
        if self._use_petsc():
            self.solve_ksp(A, x, b)
        else:
            if self._is_eigenproblem:
                pass
            
                #self.eigensolver = SLEPcEigenSolver(A)
                #self.eigensolver.parameters["solver"] = "krylov-schur"
                #self.eigensolver.parameters["spectrum"] = "smallest magnitude"
                #self.eigensolver.parameters["spectrum"] = "largest real"
                #self.logger.info('Computing eigenvalues. This could take some time ...')
                #self.eigensolver.solve(5)
            else:
                solve(A, x, b, self.options_global['linear_solver']['method'])

    def solve_ksp(self, A, x, b):
        ''' Solve the system Ax = b.
            1. Decide which operators have to be set (A or (A, P))
            2. setUp
            3. solve
            4. clear settings
        '''
        self.logger.debug('KSP: Setting up')
        if self.fstype in ('ADD', 'MULT'):
            # or just keep it constant?
            self._assemble_Pmat()
            self.ksp.setOperators(A.mat(), self.P.mat())
        else:
            self.ksp.setOperators(as_backend_type(A).mat())

        self.ksp.setConvergenceHistory()
        self.ksp.setUp()

        # self._mg_levels_setup()

        self.logger.debug('KSP: Solving Ax=b')
        t0 = Timer('PETScSolver SOLVE ')
        self.ksp.solve(as_backend_type(b).vec(), as_backend_type(x).vec())
        as_backend_type(x).update_ghost_values()
        self.timing['ksp'] = t0.stop()
        self.iterations = self.ksp.getIterationNumber()
        self.residuals = self.ksp.getConvergenceHistory()
        self.conv_reason = self.ksp.getConvergedReason()
        
        if len(self.residuals) > 0:
            convstr = 'CONVERGED' if self.conv_reason > 0 else 'DIVERGED'
            self.logger.info('{c} ({cr}) after {it} iterations. Residual: '
                             '{res}'.format(c=convstr, cr=self.conv_reason,
                                            it=self.iterations,
                                            res=self.residuals[-1]))

    def _set_fieldsplit_IS(self, ksp=None):
        ''' Sets and returns index fields for field split. For unstabilized
        saddle point problems, PETSc can detect the blocks by itself (zero
        diagonal in C=0).
            [  A  B^T ]
            [ -B  C   ]
        For stabilized problems with C != 0, the index sets need to be given
        according to the dofs of the u,p subspaces of the mixed function space,
        W. In the case of a user-defined preconditioner matrix for the Schur
        complement, the indices need to be known, too (are returned by this
        function.

        Args:
            ksp (optional)  PETSc KSP object, uses self.ksp if not given
        Returns:
            (is0, is1)  Tuple of PETSc index fields
        '''
        if not ksp:
            ksp = self.ksp
        W = self.generalproblem.w.function_space()
        pc = ksp.getPC()
        is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
        is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
        fields = [('0', is0), ('1', is1)]
        pc.setFieldSplitIS(*fields)
        return is0, is1

    def _create_user_schurPC(self):
        assert 'SCHUR_USER' in self.fstype, (
            'create_SchurPC called unexpectedly')
        W = self.generalproblem.w.function_space()
        if 'reuse_SchurPC' in self.param and self.param['reuse_SchurPC']:
            if hasattr(self, 'Sp') and self.Sp:
                return
            else:
                raise Exception('Cannot use stored Sp -- does not exist!')

        if 'SchurPC' in self.param and not self.param['SchurPC'] == 'PMM':
            raise Exception('Specified SchurPC matrix not recognized!')
        else:
            # build pressure mass matrix
            self.logger.info('  Building Sp = pressure mass matrix')
            if 'fluid' in self.options_global:
                mu = self.options_global['fluid']['dynamic_viscosity']
            else:
                mu = self.options_global['dynamic_viscosity']
            mu_inv = Constant(1./mu)
            (_, p) = TrialFunctions(W)
            (_, q) = TestFunctions(W)
            aschur = mu_inv*p*q*dx
        if self.fstype == 'SCHUR_USER_DIAG':
            aschur *= -1

        (is0, is1) = self._set_fieldsplit_IS()
        Sp = PETScMatrix()
        assemble(aschur, tensor=Sp)
        pc = self.ksp.getPC()
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER,
                                     Sp.mat().createSubMatrix(is1, is1))
        # DEBUG: save Sp matrix in CSR format
        # (ai, aj, av) = Sp.mat().createSubMatrix(is1, is1).getValuesCSR()
        # np.savez('Sp_mat', ai=ai, aj=aj, av=av)

    def _create_Pmat(self):
        ''' Create preconditioner matrix P for complete system matrix A '''
        assert self.fstype in ('ADD', 'MULT')
        self.logger.info('  Building Pmat = pressure mass matrix')
        W = self.generalproblem.w.function_space()
        if 'fluid' in self.options_global:
            mu = self.options_global['fluid']['dynamic_viscosity']
        else:
            mu = self.options_global['dynamic_viscosity']
        mu_inv = Constant(1./mu)
        (_, p) = TrialFunctions(W)
        (_, q) = TestFunctions(W)
        self.apform = self.generalproblem.jacobian + mu_inv*p*q*dx
        self.P = PETScMatrix()

    def _assemble_Pmat(self):
        ''' Assemble the preconditioner matrix P from form defined by
        _create_Pmat.
        '''
        # if 'stokes' in self.options_global and self.options_global['stokes']:
        #     if hasattr(self, '_P_assembled') and self._P_assembled is True:
        #         return
        if not (hasattr(self, '_reuse_Pmat') and self._reuse_Pmat):
            assemble(self.apform, tensor=self.P)
            [bc.apply(self.P) for bc in self.generalproblem.bcs]

    def _create_direct_solver(self):
        assert self.fstype == 'LU'
        self.logger.info('  Using direct solver')

    def _create_alg_schurPC(self):
        ''' Schur preconditioner is built automatically by PETSc. Set
        fieldsplit indes sets for the case of C != 0. '''
        self.logger.info('  {} PC built automatically from A'
                         .format(self.fstype))
        self._set_fieldsplit_IS()
        pass

    def _get_fieldsplit_type(self):
        ''' Detects fieldsplit types:
                Schur complement PCs with user defined preconditioner Sp:
                    SCHUR_USER_FULL
                    SCHUR_USER_DIAG
                    SCHUR_USER_TRI (=UPPER/LOWER)
                    SCHUR_USER_LSC *** FIXME
                Schur PCs with automatic (algebraic) preconditioner:
                    SCHUR_LSC
                    SCHUR
                Multiplicative:
                    MULT
                Additive:
                    ADD
                Direct solver:
                    LU

            returns: FieldSplit_Type (str)
        '''
        petsc_opt = self.param['petsc_options']
        if 'pc_type lu' in petsc_opt:
            return 'LU'
        elif 'pc_fieldsplit_type schur' in petsc_opt:
            if 'pc_fieldsplit_schur_precondition user' in petsc_opt:
                if 'fieldsplit_1_pc_type lsc' in petsc_opt:
                    return 'SCHUR_USER_LSC'
                elif 'pc_fieldsplit_schur_fact_type diag' in petsc_opt:
                    return 'SCHUR_USER_DIAG'
                elif 'pc_fieldsplit_schur_fact_type full' in petsc_opt:
                    return 'SCHUR_USER_FULL'
                elif ('pc_fieldsplit_schur_fact_type upper' in petsc_opt or
                      'pc_fieldsplit_schur_fact_type lower' in petsc_opt):
                    return 'SCHUR_USER_TRI'
            elif 'fieldsplit_1_pc_type lsc' in petsc_opt:
                return 'SCHUR_LSC'
            else:
                return 'SCHUR'
        elif 'pc_fieldsplit_type additive' in petsc_opt:
            return 'ADD'
        elif 'pc_fieldsplit_type multiplicative' in petsc_opt:
            return 'MULT'
        elif 'mg_levels_pc_type fieldsplit' in petsc_opt:
            return 'MG_FIELDSPLIT'
        else:
            return None

    def _use_petsc(self):
        if self._is_eigenproblem:
            return False
        else:
            return ('inputfile' in self.options_global['linear_solver'] and
                    self.options_global['linear_solver']['inputfile'])



def solver(problem):
    return Solver(problem)




