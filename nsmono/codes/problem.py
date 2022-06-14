'''
MONOLITHIC NAVIER STOKES PROBLEM
 
Written by Jeremias Garay L: j.e.garay.labra@rug.nl

'''

from dolfin import *
from common import inout
from dolfin import MPI
from pathlib import Path
from numpy import poly1d, polyfit, array
from ..logger.logger import LoggerBase
from .streamline_diffusion import SDParameter
from common import inout, utils
parameters["std_out_all_processes"] = False


class Problem(LoggerBase):
    def __init__(self, inputfile=None):

        super().__init__()
        self.options = None
        self.inputfile = inputfile
        if inputfile:
            self.get_parameters(inputfile)

        self._logging_filehandler = None
        self.setup_logger()

        #module = self.__module__.split('.')[0]
        #self.logger.info('{} {}'.format(module, utils.get_git_rev_hash(__file__)))

        self.logger.info('Initializing')
        self.logger.info('Number of parallel tasks: {}'.format(
            MPI.size(MPI.comm_world)))
        self.logger.info('Write out path: {}'.format(
            self.options['io']['write_path']))

        # mesh, boundaries, subdomains
        self.mesh = None
        self.bnds = None
        self.mu = None
        self.rho = None
        self.k = None
        self.w0 = None
        self.u0 = None
        self.w = None

    def init(self):
        ''' Initialize problem, performing the actions:
        '''

        self.set_constants()
        self.init_mesh()
        self.create_functionspaces()
        self.boundary_conditions()
        self.variational_form()

    def get_parameters(self, inputfile):
        ''' Read parameters from YAML input file into options dictionary

        Args:
            inputfile (str):     path to YAML file
        '''
        self.options = inout.read_parameters(inputfile)

    def setup_logger(self):
        ''' Create logging File Handler '''
        MPI.barrier(MPI.comm_world)
        path = Path(self.options['io']['write_path']).joinpath('run.log')
        if MPI.rank(MPI.comm_world) == 0:
            utils.trymkdir(str(path.parent))
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        MPI.barrier(MPI.comm_world)
        self.set_log_filehandler(str(path))

    def set_constants(self):
        ''' Set Constants from options file as instance attributes '''

        self.mu = Constant(self.options['fluid']['dynamic_viscosity'])
        self.rho = Constant(self.options['fluid']['density'])
        dt = self.options['timemarching']['dt']
        self.k = Constant(1./dt)
        self._is_stokes = self.options['fluid']['stokes']

    def init_mesh(self):
        ''' Read in mesh, subdomains and boundary information. '''
        self.logger.info('Reading mesh {}'.format(self.options['mesh']))
        self.mesh, self.subdomains, self.bnds = \
            inout.read_mesh(self.options['mesh'])
        self.ndim = self.mesh.topology().dim()

    def create_functionspaces(self):
        r''' Create function spaces for velocity and pressure, namely:

        * U:   VectorFunctionSpace for the velocity
        * P:   FunctionSpace for the pressure
        * W:   Mixed FunctionSpace
        

        * velocity:     p1, p2, p1b/p1+ (bubble enriched)
        * pressure:     p1, p0/dg0, p1-/dg1
        '''

        if 'elements' in self.options['fem']:
            raise Exception('Testing new interface: \'velocity_space\', '
                            '\'pressure_space\'')
        

        u_space = self.options['fem']['velocity_space'].lower()
        p_space = self.options['fem']['pressure_space'].lower()


        self.logger.info('Creating velocity space: {}'.format(
            u_space.capitalize()))

        if u_space in ('p1', 'p2'):
            deg = int(u_space[1])
            U = VectorElement('P', self.mesh.ufl_cell(), deg)
        elif u_space in ('p1b', 'p1+'):
            deg = int(u_space[1])
            P = FiniteElement('P', self.mesh.ufl_cell(), deg)
            B = FiniteElement('Bubble', self.mesh.ufl_cell(), 1 + self.ndim)
            U = VectorElement(P + B)
        else:
            raise Exception('Velocity space "{}" not supported!'
                            .format(u_space))

        self.logger.info('Creating pressure space: {}'.format(
            p_space.upper()))

        if p_space == 'p1':
            P = FiniteElement('P', self.mesh.ufl_cell(), 1)
        elif p_space in ('p0', 'dg0'):
            P = FiniteElement('DG', self.mesh.ufl_cell(), 0)
        elif p_space in ('p1-', 'dg1'):
            P = FiniteElement('DG', self.mesh.ufl_cell(), 1)
        else:
            raise Exception('Pressure space "{}" not supported!'
                            .format(p_space))

        W = FunctionSpace(self.mesh, MixedElement([U, P]))

        w = Function(W)
        w.vector().zero()

        self.w = w
        self.w0 = Function(W)
        self.W = W
        self.u0, _ = self.w0.split()
        
    def variational_form(self):
        ''' Set up variational forms of the problem and save in dictionary for
        later use (reassembly of parts).
        '''
        self.forms = {}
        
        rho = self.rho
        mu = self.mu
        k = self.k

        

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        #(u0, _) = split(self.w0)
        #self.u0 = u0

        self.u_conv_assigned = self.w.sub(0)

        is_temam = self.options['fem']['stabilization']['temam']

        def diff(u):
            return inner(mu*grad(u), grad(v))*dx

        def conv(u, u_conv):
            if self._is_stokes:
                a_conv = Constant(0)*dot(u,v)*dx
                return a_conv
            else:
                a_conv = rho*dot(grad(u)*u_conv, v)*dx
                if is_temam:
                    a_conv += 0.5*rho*div(u_conv)*dot(u, v)*dx

            return a_conv


        a_mass = rho*k*dot(u, v)*dx
        a_press = - p*div(v)*dx + q*div(u)*dx
        a_diff = diff(u)
        u_conv = self.u0
        a_conv = conv(u,u_conv)

        if self.options['fem']['stabilization']['pspg']['enabled']:
            eps = Constant(self.options['fem']['stabilization']['pspg']['eps'])
            h = CellDiameter(self.mesh)
            a_press += eps/mu*h**2*dot(grad(p), grad(q))*dx

        self.forms.update({'mass': a_mass,
                        'press': a_press,
                        'diff': a_diff,
                        'conv': a_conv})


        # adding neumann boundary conditions
        self.forms.update({'neumann': sum(self.bc_dict['neumann'])})
        # adding inflow boundary conditions
        if 'inflow' in self.bc_dict and self.bc_dict['inflow']:
            self.forms.update({'inflow_rhs': 0})
            for key, elem in self.bc_dict['inflow'].items():
                if key != 'waveform':
                    self.forms['conv'] += elem['lhs']
                    self.forms['inflow_rhs'] += elem['rhs']
                
        if self._using_mapdd:
            self.forms.update({'mapdd_rhs': 0})
            for _, elem in self.bc_dict['mapdd'].items():
                self.forms['conv'] += elem['lhs']
                self.forms['mapdd_rhs'] += elem['rhs']
        
        if self._using_wk:
            ds = self.ds
            n = FacetNormal(self.mesh)
            t_p = grad(p) - dot(grad(p),n)*n
            t_q = grad(q) - dot(grad(q),n)*n
            self.forms.update({'windkessel': 0})
            for bid, elem in self.bc_dict['windkessel'].items():
                const = rho*k*elem['eps']
                self.forms['conv'] += const*inner(t_p, t_q)*ds(bid)
                self.forms['windkessel'] += elem['Pl']*dot(v,n)*ds(bid)
        

        forms_stab = self.stabilization(u_conv)

        if 'bfs' in forms_stab:
            self.forms['conv'] += forms_stab['bfs']
        if 'fnv' in forms_stab:
            self.forms['conv'] += forms_stab['fnv']
            #self.forms.update({'fnv_rhs': forms_stab['fnv']['rhs']})
        if 'supg_convdiff' in forms_stab:
            self.forms['conv'] += forms_stab['supg_convdiff']
        if 'supg_time' in forms_stab:
            self.forms.update({'supg_time': forms_stab['supg_time']})

    def stabilization(self, u_conv=None):
        ''' Call stabilization methods as specified in options dict.

        Args:
            u_conv (Function): convecting velocity

        Returns:
            forms_stab (dict):  dictionary with keys 'bfs', 'supg_*',
              containing the weak forms of the corresponding stabilization
              terms
        '''
        forms_stab = {}
        opt = self.options['fem']['stabilization']
        if opt['backflow_boundaries']:
            forms_stab.update(self.stab_backflow(u_conv))
        if opt['streamline_diffusion']['enabled']:
            forms_stab.update(self.streamline_diffusion(u_conv))
        if 'forced_normal' in opt and opt['forced_normal']['enabled']:
            forms_stab.update(self.forced_normal())

        return forms_stab

    def streamline_diffusion(self, u_conv):
        ''' Streamline Diffusion stabilization.
        There are different definitions for the stabilization parameter tau if
        'length_scale' is 'metric' and otherwise (average or max).

        See Shakib and Hughes (1991), "A new finite element formulation for
        computational fluid dynamics" X and IX

        Args:
            u_conv:      convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2
        '''
        opt = self.options['fem']['stabilization']['streamline_diffusion']

        mu = self.mu
        rho = self.rho
        k = self.k

        sd = SDParameter(self.options, self.mesh, mu, rho, k,
                         self._logging_filehandler)
        tau = sd.stabilization_parameter(u_conv, self.u_conv_assigned)

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)

        residual_convdiff = rho*grad(u)*u_conv
        res_str = 'minimal'
        a_supg_time = None

        self.logger.info('SD/SUPG residual: {r}'.format(r=res_str))

        a_supg_convdiff = tau*rho*dot(grad(v)*u_conv, residual_convdiff)*dx

        return {'supg_convdiff': a_supg_convdiff, 'supg_time': a_supg_time}

    def stab_backflow(self, u_conv=None):
        ''' Backflow stabilization.

        Args:
            u_conv:      convecting velocity, for example Adams-Bashforth
                        interpolated for BDF2
        '''
        def abs_n(x):
            return 0.5*(x - abs(x))

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)

        ind = self.options['fem']['stabilization']['backflow_boundaries']
        self.logger.info('adding backflow stabilization on boundaries {}'.
                         format(ind))
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        rho = self.rho

        a_bfs = sum([-0.5*rho*abs_n(dot(u_conv, n))*dot(u, v)*ds(i) for i in ind])

        return {'bfs': a_bfs}

    def forced_normal(self):
        ''' Forcing normal velocity
        Args:
            u_conv:
        '''

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.mesh)

        ind = self.options['fem']['stabilization']['forced_normal']['boundaries']
        self.logger.info('forcing normal velocity on boundaries {}'.
                         format(ind))
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.bnds)
        gamma = self.options['fem']['stabilization']['forced_normal']['gamma']

        ut = u - dot(u,n)*n
        vt = v - dot(v,n)*n

        a_fnv = sum([gamma*dot(ut,vt)*ds(i) for i in ind])

        return {'fnv': a_fnv}

    def boundary_conditions(self):
        ''' Create boundary conditions '''

        BC = BoundaryConditions(self)
        BC.process_bcs()
        self.bc_dict = BC.bc_dict
        self.ds = BC.ds
        self._using_wk = BC._using_wk
        self._using_mapdd = BC._using_mapdd


class BoundaryConditions(LoggerBase):
    ''' Boundary Conditions class '''

    def __init__(self, problem):
        ''' 
        Args:
            options     options dictionary
            V           velocity FunctionSpace (vector if coupled components,
                        scalar if split components are used)
            Q           pressure FunctionSpace
            logging_filehandler (optional)      filehandler of log file
                                                (run.log)
        '''
        super().__init__()
        self._logging_filehandler = problem._logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.logger.info('Processing boundary conditions')
        self.bcs = problem.options['boundary_conditions']
        self.options = problem.options
        self.W = problem.W
        self.k = problem.k
        self.rho = problem.rho
        self.mu = problem.mu
        self.bnds = problem.bnds
        self.u0 = problem.u0
        self.w = problem.w
        self.ndim = self.W.mesh().topology().dim()
        self.ds = Measure('ds', domain=self.W.mesh(), subdomain_data=self.bnds)

        self._using_wk = False
        self._using_mapdd = False


        self.bc_dict = {
            'dirichlet': [],
            'neumann': [],
            'dbc_expressions': {},
            'windkessel': {},
            'mapdd': {},
            'inflow': {},
        }

    def process_bcs(self):
        ''' Call functions to process boundary conditions corresponding to
        their type.
        '''
        for bc in self.bcs:
            if bc['type'] == 'dirichlet':
                self._dirichlet_bc(bc)
            elif bc['type'] == 'neumann':
                self._neumann_bc(bc)
            elif bc['type'] == 'windkessel':
                self._using_wk = True
                self._windkessel(bc)
            elif bc['type'] == 'inflow':
                self._inflow_profile(bc)
            elif bc['type'] == 'mapdd':
                self._using_mapdd = True
                self._mapdd(bc)
            else:
                raise Exception('Unknown velocity BC at boundary {}'.
                                format(bc['id']))

    def _dirichlet_bc(self, bc):
        ''' Create velocity Dirichlet boundary condition

        Args:
            bc (dict):  dict describing one boundary condition
        '''

        expr = None
        val = bc['value']

        if not 'parameters' in bc:
            val = Constant(val)
        else:
            deg = bc['degree'] if 'degree' in bc else 3
            params = bc['parameters']
            expr = Expression(val, degree=deg, **params)
            val = expr

        dbc = DirichletBC(self.W.sub(0), val, self.bnds, bc['id'])
        self.bc_dict['dirichlet'].append(dbc)

        if expr:
            self.bc_dict['dbc_expressions'][dbc] = {
                'expression': expr, 'id': bc['id']}

    def _neumann_bc(self, bc):
        ''' Create weak form of Neumann boundary condition '''
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.W.mesh())
        val = Constant(bc['value'])
        a_bc = val*dot(n, v)*self.ds(bc['id'])
        self.bc_dict['neumann'].append(a_bc)

    def _inflow_profile(self,bc):

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)

        ds = self.ds
        nid = bc['id']
        n = FacetNormal(self.W.mesh())
        gamma = bc['gamma']
        uprofile = Function(self.W.sub(0).collapse())
        inout.read_HDF5_data(self.W.mesh().mpi_comm(), bc['profile'], uprofile, 'u')
        
        if '.csv' in bc['waveform']:
            self.logger.info('taking flow form csv file')
            flow_init = assemble(dot(uprofile,n)*ds(nid))
            flip = -1 if flow_init >0 else 1
            Norm_fact = flow_init*flip            
            time_data = []
            flow_data = []
            import csv
            with open(bc['waveform']) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    time_data.append(float(row[0]))
                    flow_data.append(float(row[1]))
            
            flow_func = poly1d(polyfit(time_data,flow_data,18))
            waveform = Constant(flow_func(0.0))
        else:
            ones = interpolate(Constant(1), self.W.sub(1).collapse()) 
            area = assemble(ones*ds(nid))
            Norm_fact = abs(assemble(dot(uprofile,n)*ds(nid))/area)
            params = bc['parameters']
            waveform = Expression(bc['waveform'], degree=3, **params)

        a_lhs = gamma*dot(u, v)*ds(nid)
        a_rhs = gamma*dot(uprofile*waveform/Norm_fact, v)*ds(nid)

        self.bc_dict['inflow'][bc['id']] = {
            'lhs': a_lhs,
            'rhs': a_rhs,
            'waveform': waveform,
        }

        if '.csv' in bc['waveform']:
            self.bc_dict['inflow'][bc['id']].update({'flow_func': flow_func})

    def _mapdd(self, bc):
        ''' Create weak form of MAPDD boundary condition '''

        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        n = FacetNormal(self.W.mesh())

        rho = self.rho
        mu = self.mu
        k = self.k
        l = bc['parameters']['l']
        ds = self.ds

        self.bc_dict['mapdd'][bc['id']] = {
            'lhs': None,
            'rhs': None,
            'l': Constant(l),
        }

        un = dot(u,n)
        vn = dot(v,n)
        gradtan_un = grad(un) - dot(grad(un),n)*n
        gradtan_vn = grad(vn) - dot(grad(vn),n)*n
          
        a_lhs = rho*k*dot(u,v)*ds(bc['id'])
        #a_lhs += mu*inner(grad(u), grad(v))*ds(bc['id'])
        a_lhs += mu*inner(gradtan_un, gradtan_vn)*ds(bc['id'])
        a_rhs = rho*k*dot(self.u0,v)*ds(bc['id'])

        mdict = self.bc_dict['mapdd'][bc['id']]

        mdict['lhs'] = mdict['l']*a_lhs
        mdict['rhs'] = mdict['l']*a_rhs
        
    def _windkessel(self, bc):
        ''' Create weak form of WK boundary condition '''

        dt = self.options['timemarching']['dt']
        bid = bc['id']
        R_p = bc['parameters']['R_p']
        R_d = bc['parameters']['R_d']
        C = bc['parameters']['C']
        L = bc['parameters']['L'] if 'L' in bc['parameters'] else 0
        l = bc['parameters']['l'] if 'l' in bc['parameters'] else 0
        eps = bc['parameters']['eps'] if 'eps' in bc['parameters'] else 0

        if l and not L:
            print('computing L from l in boundary',bid)
            rho = float(self.rho)
            ds = self.ds
            ones = interpolate(Constant(1), self.W.sub(1).collapse()) 
            area = assemble(ones*ds(bid))
            L = rho/area*l
        

        alpha = R_d*C/(R_d*C + dt)
        beta = R_d*(1 - alpha)
        gamma = R_p + beta
        
        pi0 = bc['parameters']['p0']
        P0 = alpha*pi0
        pi = P0
        
        self.bc_dict['windkessel'][bid] = {
            'eps': Constant(eps),
            'R_p': Constant(R_p),
            'R_d': Constant(R_d),
            'C': Constant(C),
            'L': Constant(L),
            'pi': Constant(pi),
            'pi0': Constant(pi0),
            'Q0': Constant(0),
            'Pl': Constant(P0)
        }



def problem(inputfile):
    return Problem(inputfile)