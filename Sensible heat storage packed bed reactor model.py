import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
## Reactor parameters
L = 1.8  # Reactor length (m)
D = 0.4  # Reactor bed diameter (m)
r = D / 2
w_t = 0.2 # Wall thickness (m)
D_ext = D + w_t # Reactor external diameter (m)
r_ext = D_ext / 2
e = 0.41  # Bed void fraction
Ds = 0.014 # Particle diameter (m)
Q_f = 0.0191 # Fluid flow rate (kg/s)

## Operating conditions
T_in = 433  # Inlet temperature (K)
T_f0 = 483  # Initial fluid temperature (K)
T_s0 = 483  # Initial solid temperature (K)
T_w0 = T_f0  # Initial wall temperature (K)
T_a = 293  # Ambient temperature (K)

a_air = 33.5e-6 # Air thermal diffusivity at 20°C (m²/s)
nu_air = 15.1e-6 # Air kinematic viscosity at 20°C (m²/s)
k_air = 25.9e-3 # Air thermal conductivity at 20°C (W/K)

## Calculated parameters
A = np.pi * D**2 / 4  # Cross-sectional area (m^2)
a_fs = 6 * (1 - e) / Ds  # Specific surface area for fluid-solid heat transfer (m^2/m^3)
a_w_int = 2 * r / (2*w_t * r_ext - w_t**2) # Specific internal surface area for fluid-wall heat transfer (m^2/m^3)
a_w_ext = 2 * r_ext / (2*w_t * r_ext - w_t**2)  # Specific external surface area for ambient-wall heat transfer (m^2/m^3)

## Discretization
Nz = 500 # Number of axial grid points
dz = L / (Nz - 1)  # Grid spacing
z = np.linspace(0, L, Nz)  # Axial coordinate


## Temperature-dependent property functions
# rapeseed oil
def rho_f(T):
    return 928-0.669*T

def Cp_f(T):
    return 0.0026*(T-273)+1915

def k_f(T):
    return 2e-7*T**2 + 1.714e-4*T + 0.1698

def mu_f(T):
    return 39.498*T**-1.764
# Quartzite
def rho_s(T):
    return 2500 + 0*T

def Cp_s(T):
    return 830 + 0*T

def k_s(T):
    return 5.69 + 0*T

def rho_w(T):
    return 180+0*T

def Cp_w(T):
    return 1e3+0*T

def k_w(T):
    return 0.07+0*T

def reactor_model(t, y): 
    T_f = y[:Nz]
    T_s = y[Nz:2*Nz]
    T_w = y[2*Nz:]
    
    dTf_dt = np.zeros(Nz)
    dTs_dt = np.zeros(Nz)
    dTw_dt = np.zeros(Nz)
    
    # Temperature-dependent properties
    rho_f_values = rho_f(T_f)
    Cp_f_values = Cp_f(T_f)
    k_f_values = k_f(T_f)
    mu_f_values = mu_f(T_f)
    rho_s_values = rho_s(T_s)
    Cp_s_values = Cp_s(T_s)
    k_s_values = k_s(T_s)
    rho_w_values = rho_w(T_w)
    Cp_w_values = Cp_w(T_w)
    k_w_values = k_w(T_w)
    u_values = Q_f / (rho_f_values * e * A)
    u_sup_values = e * u_values
    Re_values = rho_f_values * u_sup_values * Ds / mu_f_values
    Pr_values = mu_f_values * Cp_f_values / k_f_values
    Ra_a_values = 9.81 / (T_a*nu_air*a_air) * (T_w-T_a) * L**3
    Nu_a_values = 0.56*Ra_a_values**0.25 # Nusselt values for air around the cylinder
    h_fs_values = k_f_values / Ds * (2 + 1.1 * Re_values**0.6 * Pr_values**(1/3)) # Correlation by Ranz (1952)
    h_fw_values = k_f_values / L * (0.6 * Pr_values**(1/3) * Re_values**0.5) # Correlation by Yagi and Wakao (1959) with 1 < Re < 40
    h_wa_values = k_air * Nu_a_values / L # Nusselt number correlation with 10^4 < Ra < 10^9 ; Rayleigh number calculation with Pr = 0.71 for air at 25°C
    beta = (k_s_values - k_f_values) / (k_s_values + 2 * k_f_values) # Parameter for fluid-solid thermal conductivity calculation
    k_eff_0 = k_f_values * (1 + 2 * beta * (1 - e) + (2 * beta**3 - 0.1 * beta) * (1 - e)**2 + (1 - e)**3 * 0.05 * np.exp(4.5 * beta)) / (1 - beta * (1 - e)) # Effective fluid-solid thermal conductivity of the bed due to conduction (Gonzo, 2002)
    f = (k_eff_0 - e * k_f_values - (1 - e) * k_s_values) / (k_f_values - k_s_values) # Tortuosity of the bed
    k_eff_f = (e + f + 0.5*Re_values*Pr_values) * k_f_values # Effective fluid thermal conductivity
    k_eff_s = (1 - e - f) * k_s_values # Effective solid thermal conductivity (W/m/K)
    
    Bi = h_fs_values * Ds / (3 * k_s_values) # Biot number, must be < 0.1 for solids thermal gradient to be negligible
    
    ## Boundaries energy balances
    # Fluid phase
    dTf_dt[0] = -u_values[0] * (T_f[1] - T_in) / (2*dz) + (k_eff_f[0] * (T_f[1] - 2*T_f[0] + T_in) / dz**2 - h_fs_values[0] * a_fs * (T_f[0] - T_s[0]) - h_fw_values[0] * a_w_int * (T_f[0] - T_w[0])) / (e * rho_f_values[0] * Cp_f_values[0])
    dTf_dt[-1] = (k_eff_f[-1] * 2*(T_f[-1] - T_f[-2]) / dz**2 - h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1]) - h_fw_values[-1] * a_w_int * (T_f[-1] - T_w[-1])) / (e * rho_f_values[-1] * Cp_f_values[-1])
    
    # Solid phase
    dTs_dt[0] = (k_eff_s[0] * 2*(T_s[1] - T_s[0]) / dz**2 + h_fs_values[0] * a_fs * (T_f[0] - T_s[0])) / ((1 - e) * rho_s_values[0] * Cp_s_values[0])
    dTs_dt[-1] = (k_eff_s[-1] * 2*(T_s[-1] - T_s[-2]) / dz**2 + h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1])) / ((1 - e) * rho_s_values[-1] * Cp_s_values[-1])
    
    # Wall
    dTw_dt[0] = (k_w_values[0] * 2*(T_w[1] - T_w[0]) / dz**2 + h_fw_values[0] * a_w_int * (T_f[0] - T_w[0]) - h_wa_values[0] * a_w_ext * (T_w[0] - T_a)) / (rho_w_values[0] * Cp_w_values[0])
    dTw_dt[-1] = (k_w_values[-1] * 2*(T_w[-1] - T_w[-2]) / dz**2 + h_fw_values[-1] * a_w_int * (T_f[-1] - T_w[-1]) - h_wa_values[-1] * a_w_ext * (T_w[-1] - T_a)) / (rho_w_values[-1] * Cp_w_values[-1])
    
    ## Energy balances through the bed
    dTf_dt[1:-1] = -u_values[1:-1] * (T_f[2:] - T_f[:-2]) / (2*dz) + (k_eff_f[1:-1] * (T_f[2:] - 2*T_f[1:-1] + T_f[:-2]) / dz**2 - h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) - h_fw_values[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1])) / (e * rho_f_values[1:-1] * Cp_f_values[1:-1])
    dTs_dt[1:-1] = (k_eff_s[1:-1] * (T_s[2:] - 2*T_s[1:-1] + T_s[:-2]) / dz**2 + h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1])) / ((1 - e) * rho_s_values[1:-1] * Cp_s_values[1:-1])
    dTw_dt[1:-1] = (k_w_values[1:-1] * (T_w[2:] - 2*T_w[1:-1] + T_w[:-2]) / dz**2 + h_fw_values[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1]) - h_wa_values[1:-1] * a_w_ext * (T_w[1:-1] - T_a)) / (rho_w_values[1:-1] * Cp_w_values[1:-1])
    return np.concatenate((dTf_dt, dTs_dt, dTw_dt))

## Initial conditions
y0 = np.concatenate((np.ones(Nz) * T_f0, np.ones(Nz) * T_s0, np.ones(Nz) * T_w0))

## Time span (start, stop, number of points)
t_span = (0, 3.5*3600)
t_eval = np.linspace(0,3.5*3600, 3500)

## Solve ODE system
solution = solve_ivp(reactor_model, t_span, y0, method='Radau', t_eval=t_eval)

## Extract results
T_f = solution.y[:Nz]-273
T_s = solution.y[Nz:2*Nz]-273
T_w = solution.y[2*Nz:]-273

## Plot results
# =============================================================================
## Plot temperature function of axial position at different time steps
plt.figure(figsize=(12, 8))
plt.plot(z, T_f[:, 499],'b', label='0.5h')
plt.plot(z, T_f[:, 999],'g', label='1h')
plt.plot(z, T_f[:, 1499],'r', label='1.5h')
plt.plot(z, T_f[:, 1999],'c',label='2h')
plt.plot(z, T_f[:, 2499],'m', label='2.5h')
plt.plot(z, T_f[:, 2999],'y', label='3h')
plt.plot(z, T_f[:, 3499],'k', label='3.5h')

## Experimental data from Hoffman et al., 2016
x_05 = [0.020757825370675442, 0.1808896210873147, 0.341021416803954, 0.5011532125205932, 0.8154859967051072, 0.9785831960461285, 1.138714991762768, 1.2988467874794072, 1.4589785831960462, 1.6191103789126853, 1.7792421746293248]
T_05 = [161.18421052631578, 169.21052631578948, 177.76315789473685, 203.68421052631578, 206.97368421052633, 208.28947368421052, 208.0263157894737, 208.15789473684214, 208.42105263157896, 209.21052631578948, 209.60526315789474]
x_1 = [0.0236842105263158, 0.18059210526315791, 0.5032894736842106, 0.8141447368421054, 0.9799342105263159, 1.1338815789473686, 1.2996710526315791, 1.456578947368421, 1.6164473684210525, 1.7792763157894738, 0.34342105263157907]
T_1 = [160.82372322899505, 162.14168039538714, 170.70840197693576, 200.49423393739704, 205.8978583196046, 205.8978583196046, 206.29324546952225, 206.8204283360791, 207.87479406919275, 207.4794069192751, 161.4827018121911]
x_15 = [0.026644736842105297, 0.1894736842105264, 0.34638157894736854, 0.5032894736842106, 0.8200657894736845, 0.9769736842105263, 1.1427631578947373, 1.2996710526315791, 1.4595394736842107, 1.619407894736842, 1.7792763157894738]
T_15 = [160.42833607907744, 161.2191103789127, 160.82372322899505, 162.00988467874794, 175.5848434925865, 190.74135090609556, 198.3855024711697, 202.86655683690282, 204.57990115321252, 205.8978583196046, 205.8978583196046]
x_2 = [0.0236842105263158, 0.1865131578947369, 0.34342105263157907, 0.5032894736842106, 0.8200657894736845, 0.9799342105263159, 1.1427631578947373, 1.2996710526315791, 1.4625000000000001, 1.619407894736842, 1.7733552631578948]
T_2 = [160.0329489291598, 161.08731466227349, 160.42833607907744, 160.95551894563428, 164.77759472817132, 170.44481054365733, 176.37561779242174, 189.68698517298188, 196.54036243822077, 201.54859967051073, 203.65733113673807]
x_25 = [0.0236842105263158, 0.1835526315789474, 0.34046052631578955, 0.5032894736842106, 0.8230263157894739, 0.9799342105263159, 1.136842105263158, 1.2996710526315791, 1.4536184210526315, 1.6223684210526315, 1.7792763157894738]
T_25 = [159.63756177924216, 160.82372322899505, 160.0329489291598, 160.69192751235585, 162.00988467874794, 164.11861614497528, 165.4365733113674, 172.55354200988467, 178.8797364085667, 187.31466227347613, 195.35420098846788]
x_3 = [0.0236842105263158, 0.1835526315789474, 0.34046052631578955, 0.5032894736842106, 0.8200657894736845, 0.9799342105263159, 1.1398026315789476, 1.2967105263157896, 1.4625000000000001, 1.619407894736842, 1.7733552631578948]
T_3 = [159.63756177924216, 160.82372322899505, 160.0329489291598, 160.69192751235585, 161.08731466227349, 162.00988467874794, 162.27347611202634, 164.64579901153212, 167.54530477759474, 172.02635914332785, 178.8797364085667]
x_35 = [0.0236842105263158, 0.1835526315789474, 0.34046052631578955, 0.5032894736842106, 0.8230263157894739, 0.9769736842105263, 1.136842105263158, 1.2967105263157896, 1.4536184210526315, 1.6164473684210525, 1.7733552631578948]
T_35 = [159.63756177924216, 160.82372322899505, 160.0329489291598, 160.69192751235585, 160.56013179571664, 161.08731466227349, 160.95551894563428, 161.74629324546953, 162.93245469522242, 165.04118616144973, 167.80889621087314]
plt.plot(x_05,T_05,'bo',label='exp 0.5h')
plt.plot(x_1,T_1,'go',label='exp 1h')
plt.plot(x_15,T_15,'ro',label='exp 1.5h')
plt.plot(x_2,T_2,'co',label='exp 2h')
plt.plot(x_25,T_25,'mo',label='exp 2.5h')
plt.plot(x_3,T_3,'yo',label='exp 3h')
plt.plot(x_35,T_35,'ko',label='exp 3.5h')


plt.xlabel('Axial position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Packed Bed Reactor model')
plt.legend()
plt.grid(True)
plt.show()