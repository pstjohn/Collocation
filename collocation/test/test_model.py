import casadi as cs
import numpy as np
import pandas as pd


def create_test_model():

    # Time
    t = cs.SX.sym('t')

    # Parameter
    u = cs.SX.sym('u')
    alpha = cs.SX.sym('alpha')
    beta = cs.SX.sym('beta')
    p = cs.vertcat(u, alpha, beta)

    # Differential states
    s = cs.SX.sym('s') 
    v = cs.SX.sym('v')
    m = cs.SX.sym('m')
    x = cs.vertcat(s,v,m)

    # Differential equation
    ode = cs.vertcat(v, (u-alpha*v*v)/m, -beta*u*u)

    model = {'t': t, 'x': x, 'p': p, 'ode': ode}
    x_labels = ['s', 'v', 'm']

    return model, x_labels


def create_test_data(model, x_labels):

    ts = np.linspace(0, 100, 100)
    opts = {
        'grid': ts,
        'output_t0': True
    }

    integrator = cs.integrator('integrator', 'cvodes', model, opts)
    xf = np.asarray(integrator(x0=[0.,0.,1.], p=[0.4, 0.05, 0.1])['xf']).T
    xf_noise = xf * (1 + .1 * np.random.randn(*xf.shape))

    return pd.DataFrame(xf_noise, index=ts, columns=x_labels)
