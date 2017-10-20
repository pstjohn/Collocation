from warnings import warn

import numpy as np
from scipy.interpolate import lagrange
import pandas as pd

import casadi as cs

from collocation.VariableHandler import VariableHandler
from collocation.BaseCollocation import BaseCollocation

class Collocation(BaseCollocation):

    def __init__(self, model, x_labels=None, tf=100, nk=20, d=2):
        """ A class to handle the optimization of kinetic uptake parameters to
        match a dynamic model to a given set of experimental data.

        model: dict
            a model that describes substrate uptake and biomass formation
            kinetics. Inputs should be {t, x, p, ode}

        x_labels: list
            List of string ID's for each of the states in the model. The first
            state should represent the current biomass concentration.

        """
        # Assign sizing variables
        t = model['t']
        x = model['x']
        p = model['p']
        ode = model['ode']

        self.nx = x.shape[0]
        self.np = p.shape[0]

        assert ode.shape[0] == self.nx, \
            "Output length mismatch"

        # Attach model
        self.dxdt = cs.Function('dxdt', [t, x, p], [ode])
        self._model_dict = model

        # Attach state names
        if x_labels is None:
            x_labels = [str(i) for i in range(self.nx)]
        assert len(x_labels) == self.nx, "Name length mismatch"
        self.x_labels = np.asarray(x_labels)

        super(Collocation, self).__init__(nk=nk, d=d)

        # setup defaults
        self.tf = tf

        # Initialize constraints
        self._initialize_polynomial_coefs()
        self._initialize_variables()
        self._initialize_polynomial_constraints()
        
    def initialize(self, **kwargs):
        """ Call after setting up boundary kinetics, finalizes the
        initialization and sets up the NLP problem. Keyword arguments are
        passed directly as options to the NLP solver """

        self._initialize_mav_objective()
        self._initialize_solver(**kwargs)

    def _initialize_variables(self):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'p'  : (self.np),
        }

        self.var = VariableHandler(core_variables)
        
        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 200.
        self.var.x_in[:] = 1.

        self.var.p_lb[:] = 0.
        self.var.p_ub[:] = 100.
        self.var.p_in[:] = 0.

        # Initialize optimization parameters
        parameters = {
            'alpha' : (1,),
        }

        self.pvar = VariableHandler(parameters)
        self.pvar.alpha_in[:] = 0.
        
    def _initialize_polynomial_constraints(self):
        """ Add constraints to the model to account for system dynamics and
        continuity constraints """

        h = self.tf / self.nk

        # All collocation time points
        T = np.zeros((self.nk, self.d+1), dtype=object)
        for k in range(self.nk):
            for j in range(self.d+1):
                T[k,j] = h*(k + self.col_vars['tau_root'][j])

        # For all finite elements
        for k in range(self.nk):

            # For all collocation points
            for j in range(1, self.d+1):

                # Get an expression for the state derivative at the collocation
                # point
                xp_jk = cs.mtimes(cs.DM(self.col_vars['C'][:,j]).T, self.var.x_sx[k]).T

                # Add collocation equations to the NLP.
                # Boundary fluxes are calculated by multiplying the EFM
                # coefficients in V by the efm matrix
                [fk] = self.dxdt.call(
                    [T[k,j], self.var.x_sx[k,j], self.var.p_sx[:]])

                self.add_constraint(h * fk - xp_jk)

            # Add continuity equation to NLP
            if k+1 != self.nk:
                
                # Get an expression for the state at the end of the finite
                # element for each state
                self.add_constraint(self.var.x_sx[k+1,0] - 
                                    self._get_endpoint_expr(self.var.x_sx[k]))

        # Get an expression for the endpoint for objective purposes
        xf = self._get_endpoint_expr(self.var.x_sx[-1])
        self.xf = {met : x_sx for met, x_sx in zip(self.x_labels, xf)}

        # Similarly, get an expression for the beginning point
        x0 = self.var.x_sx[0,0,:]
        self.x0 = {met : x_sx for met, x_sx in zip(self.x_labels, x0)}    

    def _initialize_mav_objective(self):
        """ Initialize the objective function to minimize the absolute value of
        the parameter vector """

        self.objective_sx += (self.pvar.alpha_sx[:] *
                              cs.sum1(cs.fabs(self.var.p_sx[:])))


    def _plot_setup(self):

        # Create vectors from optimized time and states
        h = self.tf / self.nk

        self.fs = h * np.arange(self.nk)
        self.ts = np.array(
            [point + h*np.array(self.col_vars['tau_root']) for point in 
             np.linspace(0, self.tf, self.nk,
                         endpoint=False)]).flatten()

        self.sol = self.var.x_op.reshape((self.nk*(self.d+1)), self.nx)


    def _get_interp(self, t, states=None, x_rep='sx'):
        """ Return a polynomial representation of the state vector
        evaluated at time t.

        states: list
            indicies of which states to return

        x_rep: 'sx' or 'op', most likely.
            whether or not to interpolate symbolic or optimal values of the
            state variable

        """

        assert t < self.tf, "Requested time is outside of the simulation range"

        h = self.tf / self.nk

        if states is None: states = range(1, self.nx)

        finite_element = int(t / h)
        tau = (t % h) / h
        basis = np.asarray(self.col_vars['lfcn'](tau)).flatten()
        if x_rep != 'sx':
            x = getattr(self.var, 'x_' + x_rep)
            x_roots = x[finite_element, :, states]
            return np.inner(basis, x_roots)

        else:
            x_roots = self.var.x_sx[finite_element, :, states]
            return cs.dot(basis, x_roots)


    def set_data(self, data, weights=None):
        """ Attach experimental measurement data.

        data : a pd.DataFrame object
            Data should have columns corresponding to the state labels in
            self.x_labels, with an index corresponding to the measurement
            times.

        TODO: add weights -- should correlate with stddev

        """

        # Should raise an error if no state name is present
        df = data.loc[:, self.x_labels]

        # Rename columns with state indicies
        df.columns = np.arange(self.nx)

        # Remove empty (nonmeasured) states
        self.data = df.loc[:, ~pd.isnull(df).all(0)]

        self._set_objective_from_data(self.data)


    def _set_objective_from_data(self, data):

        obj_list = []
        for ((ti, state), xi) in data.stack().items():
            obj_list += [(self._get_interp(ti, [state]) - xi)]

        obj_resid = cs.sum_square(cs.vertcat(*obj_list))
        self.objective_sx += obj_resid


    def solve(self, ode=True, **kwargs):

        out = super(Collocation, self).solve(**kwargs)
        self._plot_setup()

        if ode:
            self.solve_ode()

        return out

    def solve_ode(self):
        """ Solve the ODE using casadi's CVODES wrapper to ensure that the
        collocated dynamics match the error-controlled dynamics of the ODE """


        self.ts.sort() # Assert ts is increasing
                                     
        integrator = cs.integrator(
            'int', 'cvodes', self._model_dict,
            {
                'grid': self.ts,
                'output_t0': True,
            })


        x_sim = self.sol_sim = np.array(integrator(
            x0=self.sol[0], p=self.var.p_op)['xf']).T

        err = ((self.sol - x_sim).mean(0) /
               (self.sol.mean(0))).mean()

        if err > 1E-3: warn(
                'Collocation does not match ODE Solution: '
                '{:.2%} Error'.format(err))

    def reset_objective(self):
        self.objective_sx = 0

    @property
    def rss(self):
        """ Residual sum of squares """
        
        x_reg = pd.DataFrame([
            self._get_interp(t, states=self.data.columns, x_rep='op')
            for t in self.data.index], index=self.data.index,
                             columns=self.data.columns)

        return ((self.data - x_reg)**2).sum().sum()

    @property
    def aic(self):
        """ Akaike information criterion """

        n = np.multiply(*self.data.shape)
        k = self.np

        return 2*k + n*np.log(self.rss)

    def _interpolate_solution(self, ts):

        h = self.tf / self.nk
        stage_starts = pd.Series(h * np.arange(self.nk))
        stages = stage_starts.searchsorted(ts, side='right') - 1

        out = np.empty((len(ts), self.nx))
    
        for ki in range(self.nk):
            for ni in range(self.nx):
                interp = lagrange(self.col_vars['tau_root'], 
                                  self.var.x_op[ki, :, ni])

                out[stages == ki, ni] = interp(
                    (ts[stages == ki] - stage_starts[ki])/h)

        return out
