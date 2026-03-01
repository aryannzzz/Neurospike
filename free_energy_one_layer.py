from brian2 import *

# global settings 
defaultclock.dt = 0.1*ms
sim_dur = 2*second
np.random.seed(42)

# noisy sensory stream  u(t)  
stim = np.random.normal(loc=5, scale=2,  size=20000)
u_ext = TimedArray(stim, dt=defaultclock.dt)

# time constants
tau_phi = 100*ms
tau_eps = 70*ms
tau_e   = 15*ms

phi_prior = 2.0                # <-- prior belief for phi

# single–layer predictive‑coding node 
eqs = '''
# generative mapping  u ≈ θ · φ²
g_phi = theta * phi**2 : 1
hprime = 2*phi          : 1    # dg/dphi

# feature (latent) unit
dphi/dt = (theta*eps0*hprime - eps1) / tau_phi : 1

# prediction errors
deps0/dt = (u_ext(t) - g_phi - e0)       / tau_eps : 1  # sensory error
deps1/dt = (phi       - phi_prior - e1)  / tau_eps : 1  # prior error

# interneurons
de0/dt = (sigma0*eps0 - e0) / tau_e : 1
de1/dt = (sigma1*eps1 - e1) / tau_e : 1

# precision (optional: learnable or fixed)
dsigma0/dt = alpha_sigma * (eps0*e0 - 1) : 1
dsigma1/dt = alpha_sigma * (eps1*e1 - 1) : 1

# gain parameter θ  (learned)
dtheta_raw/dt = alpha_theta * eps0 * phi**2 : 1
theta : 1 (shared)

# learning‑rates
alpha_sigma : Hz (constant, shared)
alpha_theta : Hz (constant, shared)
'''

net = NeuronGroup(1, eqs, method='euler', name='pc')

# initial state
net.phi        = 0.5
net.eps0       = 0.6
net.eps1       = -0.9
net.e0         = 0.8
net.e1         = 0.1
net.sigma0     = 1
net.sigma1     = 1
net.alpha_sigma = 0.05/second
net.alpha_theta = 1/second          # learning speed
net.theta_raw  = 1.0                # starting guess
net.theta      = net.theta_raw      # sync copy

# keep θ shared‑safe
@network_operation(dt=defaultclock.dt)
def sync_theta():
    net.theta = net.theta_raw

# monitor dynamics
mon = StateMonitor(net, ['phi', 'eps0', 'eps1', 'theta','e1','e0','sigma0','sigma1'], record=0)

run(sim_dur)

# plotting
import matplotlib.pyplot as plt
plt.plot(mon.t/ms, mon.phi[0],       label='phi')
plt.plot(mon.t/ms, mon.theta[0],     label='theta')
plt.plot(mon.t/ms, mon.eps0[0], '--',label='eps0')
plt.plot(mon.t/ms, mon.e0[0], '--',label='e0')
plt.plot(mon.t/ms, mon.eps1[0], '--',label='eps1')
plt.plot(mon.t/ms, mon.e1[0], '--',label='e1')
plt.plot(mon.t/ms, mon.sigma0[0], '--',label='variance of u wrt g(phi)')
plt.plot(mon.t/ms, mon.sigma1[0], '--',label='variance of phi wrt phi_prior')

plt.xlabel('Time (ms)'); plt.legend(); plt.title('Single‑layer PC with prior')
plt.show()
