from brian2 import *

# ---------------- global settings ----------------
defaultclock.dt = 0.1*ms
sim_dur        = 2*second          # play with this
np.random.seed(48)

# external stimulus:  baseline 2.0  ± Gaussian noise
stim_rate = 10000                 # time points
stim = 2.0 + 0.8*np.random.randn(stim_rate)
u_ext = TimedArray(stim, dt=defaultclock.dt)

# time‑constants
tau_phi = 70*ms
tau_eps = 80*ms
tau_e   = 10*ms
phi2_prior = 1.0  # tune


# -------------- Brian2 node containing *all* variables --------------
eqs = '''


# --- equations ---

# --------- generative function and its derivative ----------
g_phi1 = theta * phi1**2 : 1
g_phi2 = theta * phi2**2 : 1
hprime1 = 2*phi1         : 1
hprime2 = 2*phi2         : 1

# ------------ feature units (phi) --------------------------
dphi1/dt = (theta*eps0*hprime1 - eps1)/tau_phi : 1
dphi2/dt = (theta*eps1*hprime2 - eps2)/tau_phi : 1

# ------------ prediction‑error units (eps) ----------------
deps0/dt = (u_ext(t) - g_phi1 - e0)/tau_eps : 1
deps1/dt = (phi1      - g_phi2 - e1)/tau_eps : 1
deps2/dt = (phi2 - phi2_prior - e2) / tau_eps : 1

# ------------ inhibitory interneurons (e) -----------------
de0/dt   = (sigma0*eps0 - e0)/tau_e : 1
de1/dt   = (sigma1*eps1 - e1)/tau_e : 1
de2/dt = (sigma2 * eps2 - e2) / tau_e : 1

# ------------ plasticity: variances & gain ---------------
dsigma0/dt = alpha_sigma * (eps0*e0 - 1)                : 1
dsigma1/dt = alpha_sigma * (eps1*e1 - 1)                : 1
dsigma2/dt = alpha_sigma * (eps2*e2 - 1)                : 1

# ----------------- plasticity variables -----------------
dtheta_raw/dt = alpha_theta * (eps0*phi1**2 + eps1*phi2**2 + eps2*phi2**2) : 1
theta : 1 (shared)      # read‑only copy used elsewhere


# ------------ parameters ----------------------------------
alpha_sigma : Hz (constant, shared)
alpha_theta : Hz (constant, shared)
'''

net = NeuronGroup(1, eqs, method='euler', name='pc')


# initial conditions
net.phi1     = 0.2
net.phi2     = 0.1
net.eps0     = 0.8
net.eps1     = 0.2
net.eps2=0.1
net.e0       = 0.3
net.e1       = 0.5
net.e2=0
net.sigma0   = 1.0
net.sigma1   = 1.0
net.sigma2=1.0
net.alpha_sigma = 1e-3 / second   # ⟵ tune these
net.alpha_theta = 2 /second
net.theta_raw = 1 # learnable gain


@network_operation(dt=defaultclock.dt)
def update_theta():
    net.theta = net.theta_raw
    
@network_operation(dt=10*ms)
def print_theta_derivative():
    dtheta = net.alpha_theta * (net.eps0*net.phi1**2 + net.eps1*net.phi2**2)
    print(dtheta)



# ------------------- monitors -------------------
mon = StateMonitor(net, ['phi1', 'phi2', 'eps0', 'eps1', 'theta','theta_raw',
                         'sigma0', 'sigma1'], record=0)

run(sim_dur)

import matplotlib.pyplot as plt

plot(mon.t/ms, mon.phi1[0], label='phi1')
plot(mon.t/ms, mon.theta_raw[0], label='theta')
plot(mon.t/ms, mon.eps1[0], label='eps1')

xlabel('Time (ms)')
ylabel('Value')
legend()
title('Learning dynamics of phi1 and theta')
plt.show()



