import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Example with rapeseed oil as fluid and quartzite rocks as solids
## Reactor parameters
L = 1.8  # Reactor length (m)
D = 0.4  # Reactor diameter (m)
e = 0.41  # Bed void fraction, usually between 0.2 and 0.4
Ds = 0.04 # Particle diameter (m)
t_w = 0.05  # Wall thickness (m)
Q_f_in = 0.02 # Fluid flow rate (kg/s)

## Operating conditions
T_in = 500  # Inlet temperature (K)
T_f0 = 300  # Initial fluid temperature (K)
T_s0 = 300  # Initial solid temperature (K)
T_w0 = T_f0  # Initial wall temperature (K)
T_a = 298  # Ambient temperature (K)

nu_air = 1.8e-5 # Air dynamic viscosity at 25°C (Pa.s)
k_air = 0.025

## Calculated parameters
A = np.pi * D**2 / 4  # Cross-sectional area (m^2)
a_fs = 6 * (1 - e) / Ds  # Specific surface area for fluid-solid heat transfer (m^2/m^3)
a_w = 4 / D  # Specific surface area for wall heat transfer (m^2/m^3)

# Bi = h_fs(T) * Ds / (6 * k_s(T)) # Biot number, must be < 0.1 for solids thermal gradient to be negligible

## Discretization
Nz = 100 # Number of axial grid points
dz = L / (Nz - 1)  # Grid spacing
z = np.linspace(0, L, Nz)  # Axial coordinate

## Temperature-dependent property functions
def rho_f(T):
    return 871.1 - 0.713*T

def Cp_f(T):
    return 1836.8 + 3.456*T

def k_f(T):
    return 0.125 + 0.00014*T

def mu_f(T):
    return 72.159*T**(-2.096)
# rho_f = 850  # Fluid density (kg/m^3)
# Cp_f = 2400  # Fluid heat capacity (J/kg/K)
# mu_f = 0.002 # Fluid dynamic viscosity (kg/m/s)
# rho_s = 2500  # Solid density (kg/m^3)
# Cp_s = 830  # Solid heat capacity (J/kg/K)
# rho_w = 7800  # Wall density (kg/m^3)
# Cp_w = 466  # Wall heat capacity (J/kg/K)
# k_f = 0.15 # Fluid thermal conductivity (W/m/K)
# k_s = 5.69 # Solid thermal conductivity (W/m/K)
# k_w = 45 # Wall thermal conductivity (W/m/K)
def rho_s(T):
    return 2500+0*T

def Cp_s(T):
    return 830+0*T

def k_s(T):
    return 5.7+0*T

def rho_w(T):
    return 7800+0*T

def Cp_w(T):
    return 466+0*T

def k_w(T):
    return 45+0*T

def schumann_model(t, y): 
    T_f = y[:Nz]
    T_s = y[Nz:2*Nz]
    T_w = y[2*Nz:]
    
    dTf_dt = np.zeros(Nz)
    dTs_dt = np.zeros(Nz)
    dTw_dt = np.zeros(Nz)
    
    if t <= 5000:
       Q_f = Q_f_in  # Constant flow for first 1000 seconds
    else:
       Q_f = 0  # No flow after 1000 seconds
    
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
    h_fs_values = k_f_values / Ds * (2 + 1.8 * Re_values**0.5 * Pr_values**(1/3)) # Correlation by Ranz (1952)
    h_fw_values = k_f_values / D * (0.6 * Pr_values**(1/3) * Re_values**0.5) # Correlation by Yagi and Wakao (1959) with 1 < Re < 40
    h_wa_values = k_air / L * 0.59 * (0.71 * 9.81 * (T_w - T_a) * L**3 / ((T_w + T_a) / 2 * nu_air**2))**0.25 # Nusselt number correlation with 10^4 < Ra < 10^9 ; Rayleigh number calculation with Pr = 0.71 for air at 25°C
    
    beta = (k_s_values - k_f_values) / (k_s_values + 2 * k_f_values) # Parameter for fluid-solid thermal conductivity calculation
    k_eff_0 = k_f_values * (1 + 2 * beta * (1 - e) + (2 * beta**3 - 0.1 * beta) * (1 - e)**2 + (1 - e)**3 * 0.05 * np.exp(4.5 * beta)) / (1 - beta * (1 - e)) # Effective fluid-solid thermal conductivity of the bed due to conduction (Gonzo, 2002)
    f = (k_eff_0 - e * k_f_values - (1 - e) * k_s_values) / (k_f_values - k_s_values) # Tortuosity of the bed
    k_eff_f = (e + f) * k_f_values # Effective fluid thermal conductivity
    # Radiation must be added in for gaseous HTF
    k_eff_s = (1 - e - f) * k_s_values # Effective solid thermal conductivity (W/m/K)
    
    ## Boundaries energy balances
    # Fluid phase
    dTf_dt[0] = -u_values[0] * (T_f[0] - T_in) / dz + (k_eff_f[0] * (T_f[1] - T_f[0]) / dz**2 - h_fs_values[0] * a_fs * (T_f[0] - T_s[0]) - h_fw_values[0] * a_w * (T_f[0] - T_w[0])) / (e * rho_f_values[0] * Cp_f_values[0])
    dTf_dt[-1] = -u_values[-1] * (T_f[-1] - T_f[-2]) / dz + (-h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1]) - h_fw_values[-1] * a_w * (T_f[-1] - T_w[-1])) / (e * rho_f_values[-1] * Cp_f_values[-1])
    
    # Solid phase
    dTs_dt[0] = (k_eff_s[0] * (T_s[1] - T_s[0]) / dz**2 + h_fs_values[0] * a_fs * (T_f[0] - T_s[0])) / ((1 - e) * rho_s_values[0] * Cp_s_values[0])
    dTs_dt[-1] = (k_eff_s[-1] * (T_s[-2] - T_s[-1]) / dz**2 + h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1])) / ((1 - e) * rho_s_values[-1] * Cp_s_values[-1])
    
    # Wall
    dTw_dt[0] = (k_w_values[0] * (T_w[1] - T_w[0]) / dz**2 + h_fw_values[0] * a_w * (T_f[0] - T_w[0]) - h_wa_values[0] * a_w * (T_w[0] - T_a)) / (rho_w_values[0] * Cp_w_values[0])
    dTw_dt[-1] = (k_w_values[-1] * (T_w[-2] - T_w[-1]) / dz**2 + h_fw_values[-1] * a_w * (T_f[-1] - T_w[-1]) - h_wa_values[-1] * a_w * (T_w[-1] - T_a)) / (rho_w_values[-1] * Cp_w_values[-1])
    
    # # Energy balances through the bed
    dTf_dt[1:-1] = -u_values[1:-1] * (T_f[1:-1] - T_f[:-2]) / dz + (k_eff_f[1:-1] * (T_f[2:] - 2*T_f[1:-1] + T_f[:-2]) / dz**2 - h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) - h_fw_values[1:-1] * a_w * (T_f[1:-1] - T_w[1:-1])) / (e * rho_f_values[1:-1] * Cp_f_values[1:-1])
    dTs_dt[1:-1] = (k_eff_s[1:-1] * (T_s[2:] - 2*T_s[1:-1] + T_s[:-2]) / dz**2 + h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1])) / ((1 - e) * rho_s_values[1:-1] * Cp_s_values[1:-1])
    dTw_dt[1:-1] = (k_w_values[1:-1] * (T_w[2:] - 2*T_w[1:-1] + T_w[:-2]) / dz**2 + h_fw_values[1:-1] * a_w * (T_f[1:-1] - T_w[1:-1]) - h_wa_values[1:-1] * a_w * (T_w[1:-1] - T_a)) / (rho_w_values[1:-1] * Cp_w_values[1:-1])
    return np.concatenate((dTf_dt, dTs_dt, dTw_dt))

## Initial conditions
y0 = np.concatenate((np.ones(Nz) * T_f0, np.ones(Nz) * T_s0, np.ones(Nz) * T_w0))

## Time span (start, stop, number of points)
t_span = (0, 50000)
t_eval = np.linspace(0, 50000, 2000)

## Solve ODE system
solution = solve_ivp(schumann_model, t_span, y0, method='Radau', t_eval=t_eval)

## Extract results
T_f = solution.y[:Nz]
T_s = solution.y[Nz:2*Nz]
T_w = solution.y[2*Nz:]

## Plot results
# =============================================================================
## Plot temperature function of axial position at different time steps
plt.figure(figsize=(12, 8))
plt.plot(z, T_f[:, -1], label='Fluid temperature')
plt.plot(z, T_s[:, -1], label='Solid temperature')
plt.plot(z, T_w[:, -1], label='Wall temperature')
plt.xlabel('Axial position (m)')
plt.ylabel('Temperature (K)')
plt.title('Extended Schumann Model - Packed Bed Reactor with Wall Effects')
plt.legend()
plt.grid(True)
plt.show()
# =============================================================================
# Plot temperature evolution over time at different reactor positions
positions = [0, Nz//4, Nz//2, 3*Nz//4, -1]  # Start, 1/4, 1/2, 3/4, End
plt.figure(figsize=(12, 8))
for pos in positions:
    plt.plot(solution.t, T_f[pos, :], label=f'Fluid at z={z[pos]:.2f}m')
    plt.plot(solution.t, T_s[pos, :], '--', label=f'Solid at z={z[pos]:.2f}m')
#    plt.plot(solution.t, T_w[pos, :], ':', label=f'Wall at z={z[pos]:.2f}m')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Evolution Over Time at Different Reactor Positions')
plt.legend()
plt.grid(True)
plt.show()