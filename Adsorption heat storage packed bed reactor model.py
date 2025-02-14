# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Reactor parameters
L = 0.2  # Reactor length (m)
D = 0.72  # Reactor bed diameter (m)
r = D / 2
w_t = 0.2 # Wall thickness (m)
D_ext = D + w_t # Reactor external diameter (m)
r_ext = D_ext / 2
e = 0.4  # Bed void fraction
e_r = 0.321 # Particle reactive fraction (by volume) = 0.321 for 1 cm inert diameter + 2 mm reactive coating
Ds_i = 0.01 # Inert particle center diameter (m)
c_t = 0.002 # Coating thickness (m)
Ds = Ds_i + 2*c_t # Particle diameter (m)
Q_f = 0.06015 # Fluid flow rate (kg/s) (=180 m3/h at 20°C/70% RH)


## Reaction parameters and constants
R = 8.31 # ideal gas constant (J/K/mol)
dH = 3800e3 # Synthesis reaction enthalpy (J/kgw)
dH_m = 1.2e6
Ea = 4e4 # Arrhenius activation energy
# q constants
k_vs = 0.032
D_0 = 4e-7
b_0 = 5e4
a = 3.04
q_n1 = 0.84
q_n0 = -198
q_cap1 = 0.074
q_cap2 = -4.7e-5
q_cap3 = -3.9e-3

M_v = 0.018 # Water molar mass (kg/mol)

## Operating conditions
T_in = 294  # Inlet temperature (K)
T_f0 = 295  # Initial fluid temperature (K)
T_s0 = 296  # Initial solid temperature (K)
T_w0 = T_f0  # Initial wall temperature (K)
T_a = 293  # Ambient temperature (K)
T_in_charge = 453 # Charging temperature (K)

P_amb = 101325 # Ambient pressure (Pa)
P_in = P_amb # Inlet pressure (Pa)
phi_in = 0.7 # Inlet relative humidity
P_ws = np.exp(23.1964-3816.44/(T_in - 46.13)) # Antoine equation for water saturation vapor pressure (Pa)
x_in = 0.622*phi_in * P_ws / (P_amb - P_ws * phi_in) # Inlet absolute humidity (kgw/kg)
x_0 = 0 # Initial absolute humidity (kgw/kg)

a_air = 33.5e-6 # Air thermal diffusivity at 20°C (m²/s)
nu_air = 15.1e-6 # Air kinematic viscosity at 20°C (m²/s)
k_air = 25.9e-3 # Air thermal conductivity at 20°C (W/K)

## Calculated parameters
A = np.pi * r**2  # Cross-sectional area (m^2)
V = A*L # Reactor volume (m^3)

a_fs = 6 * (1 - e) / Ds  # Specific surface area for fluid-solid heat transfer (m^2/m^3)
a_w_int = 2 * r / (2*w_t * r_ext - w_t**2) # Specific internal surface area for fluid-wall heat transfer (m^2/m^3)
a_w_ext = 2 * r_ext / (2*w_t * r_ext - w_t**2)  # Specific external surface area for ambient-wall heat transfer (m^2/m^3)


## Discretization
Nz = 100  # Number of axial grid points
dz = L / (Nz - 1)  # Grid spacing
z = np.linspace(0, L, Nz)  # Axial coordinate

## Temperature-dependent property functions
def rho_f(T,AH): # Air density, kg/m^3
    return P_in / (287*T) * (1+AH) / (1+1.61*AH) # Ideal gas law

def Cp_f(x):
    return 1.82*x+1.005

def k_f(T):
    return 0.03+0*T

def mu_f(T):
    return 4.564e-8*T+4.745e-6

def rho_s(T):
    return 1900+0*T

def Cp_s(T):
    return 780+0*T

def k_s(T):
    return 0.26+0*T

def rho_w(T):
    return 180+0*T

def Cp_w(T):
    return 1e3+0*T

def k_w(T):
    return 0.07+0*T

def p_vs_values(T):
    return np.exp(23.2-3816/(T-46.1))

def phi_values(T,AH):
    """
    Returns humidity ratio W [1] as W= AH*Pin/((0.622 + AH)*P_vs)
    :param T: fluid temperature. [K]
    :param AH: absolute humidity. [kg of water vapor per kg of dry air]
    """
    return AH * P_in / ((0.622+AH)*p_vs_values(T)) # Assuming constant pressure inside the reactor

def reactor_model(t, y): 
    T_f = y[:Nz]
    T_s = y[Nz:2*Nz]
    T_w = y[2*Nz:3*Nz]
    x = y[3*Nz:4*Nz] # Absolute humidity (kgw/kgair)
    q = y[4*Nz:] # Adsorbed layer density (kgw/m^3)
    
    dTf_dt = np.zeros(Nz)
    dTs_dt = np.zeros(Nz)
    dTw_dt = np.zeros(Nz)
    dx_dt = np.zeros(Nz)
    dq_dt = np.zeros(Nz)
    
    # Temperature-dependent properties
    rho_f_values = rho_f(T_f,x)
    Cp_f_values = Cp_f(T_f)
    k_f_values = k_f(T_f)
    mu_f_values = mu_f(T_f)
    rho_s_values = rho_s(T_s)
    Cp_s_values = Cp_s(T_s)
    k_s_values = k_s(T_s)
    rho_w_values = rho_w(T_w)
    Cp_w_values = Cp_w(T_w)
    k_w_values = k_w(T_w)
    
    phi = phi_values(T_f, x)
    
    # Following equations mostly from : A review on experience feedback and numerical modeling of packed-bed thermal energy storage systems (Esence et al., 2016) => validées avec modèle sensible
    u = Q_f / (rho_f_values * e * A) # Interstitial velocity (m/s)
    u_sup = e * u # Superficial velocity (m/s) = 0.123
    Re_values = rho_f_values * u_sup * Ds / mu_f_values
    Pr_values = mu_f_values * Cp_f_values / k_f_values
    Ra_a_values = 9.81 / (T_a*nu_air*a_air) * (T_w-T_a) * L**3
    Nu_a_values = 0.56*Ra_a_values**0.25 # Nusselt values for air around the cylinder
    h_fs_values = k_f_values / Ds * (2 + 1.1 * Re_values**0.6 * Pr_values**(1/3)) # Correlation by Wakao et al. (1979)
    h_fw_values = k_f_values / L * (0.12 * Pr_values**(1/3) * Re_values**0.75) # Correlation by Kunii and Suzuki (1968) with Re > 100
    h_wa_values = k_air * Nu_a_values / L # Nusselt number correlation with 10^4 < Ra < 10^9 ; Rayleigh number calculation with Pr = 0.71 for air at 25°C
    beta = (k_s_values - k_f_values) / (k_s_values + 2 * k_f_values) # Parameter for fluid-solid thermal conductivity calculation
    k_eff_0 = k_f_values * (1 + 2 * beta * (1 - e) + (2 * beta**3 - 0.1 * beta) * (1 - e)**2 + (1 - e)**3 * 0.05 * np.exp(4.5 * beta)) / (1 - beta * (1 - e)) # Effective fluid-solid thermal conductivity of the bed due to conduction (Gonzo, 2002)
    f = (k_eff_0 - e * k_f_values - (1 - e) * k_s_values) / (k_f_values - k_s_values) # Tortuosity of the bed
    k_eff_f = (e + f + 0.5*Re_values*Pr_values) * k_f_values # Effective fluid thermal conductivity
    k_eff_s = (1 - e - f) * k_s_values # Effective solid thermal conductivity (W/m/K)
    k_m = 15*D_0 / c_t**2 * np.exp(-Ea / (R * T_f)) + k_vs * u_sup # LDF time constant, Gondre : Eq III.1 ; between 1.4 and 4 for STAID
    b = b_0 * np.exp(-dH_m * M_v / (R * T_f)) # Gondre : Eq III.2
    q_n = q_n1 * T_in_charge + q_n0 # Gondre : Eq III.4
    q_cap = q_cap1 * T_in_charge + q_cap2*T_in + q_cap3 # Gondre : Eq III.3
    q_e = b * phi * q_n / (1 + b * phi) + a*phi + q_cap * phi / (1-phi)

    #Bi = h_fs_values * Ds / (6 * k_s_values) # Biot number, must be < 0.1 for solids thermal gradient to be negligible
    
    ## Sorption
    dq_dt = k_m * (q_e - q) # Gondre : Eq II.24

    ## Vapour mass conservation
    #dx_dt[0] = -1/e*(0.*dq_dt[0] + u[0] *(x[1] - x_in) / dz)
    #dx_dt[1:-1] = -1/e*(0. * dq_dt[1:-1] + u[1:-1] * (x[1:-1] - x[0:-2]) / dz )
    #dx_dt[-1] = -1/e * 0. * dq_dt[-1] / dz
    ## Vapour mass conservation, only advection
    dx_dt[0] = - u[0] * (x[1] - x_in) / dz  #precribed at inlet
    dx_dt[1:] = - u[1:] * (x[1:] - x[0:-1]) / dz  #upwind scheme
    ## Vapour mass conservation, water consumption
    # there is e*rho_f*dz*1 kg of dry air in the cell.
    dx_dt = dx_dt - dq_dt / (e*rho_f_values)

    # Bilans à vérifier avec ceux de Gondre    

    ## Energy balances
    ## Boundaries
    # Fluid phase
    dTf_dt[0] = -u[0] * (T_f[1] - T_in) / (2*dz) + (k_eff_f[0] * (T_f[1] - 2*T_f[0] + T_in) / dz**2 - h_fs_values[0] * a_fs * (T_f[0] - T_s[0]) - h_fw_values[0] * a_w_int * (T_f[0] - T_w[0])) / (e * rho_f_values[0] * Cp_f_values[0])
    dTf_dt[-1] = (k_eff_f[-1] * (T_f[-2] - T_f[-1]) / dz**2 - h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1]) - h_fw_values[-1] * a_w_int * (T_f[-1] - T_w[-1])) / (e * rho_f_values[-1] * Cp_f_values[-1])
    
    # Solid phase
    dTs_dt[0] = (k_eff_s[0] * (T_s[1] - T_s[0]) / dz**2 + h_fs_values[0] * a_fs * (T_f[0] - T_s[0]) + dH * dq_dt[0]) / ((1 - e) * rho_s_values[0] * Cp_s_values[0])
    dTs_dt[-1] = (k_eff_s[-1] * (T_s[-2] - T_s[-1]) / dz**2 + h_fs_values[-1] * a_fs * (T_f[-1] - T_s[-1]) + dH * dq_dt[-1]) / ((1 - e) * rho_s_values[-1] * Cp_s_values[-1])
    
    # Wall
    dTw_dt[0] = (k_w_values[0] * (T_w[1] - T_w[0]) / dz**2 + h_fw_values[0] * a_w_int * (T_f[0] - T_w[0]) - h_wa_values[0] * a_w_ext * (T_w[0] - T_a)) / (rho_w_values[0] * Cp_w_values[0])
    dTw_dt[-1] = (k_w_values[-1] * (T_w[-2] - T_w[-1]) / dz**2 + h_fw_values[-1] * a_w_int * (T_f[-1] - T_w[-1]) - h_wa_values[-1] * a_w_ext * (T_w[-1] - T_a)) / (rho_w_values[-1] * Cp_w_values[-1])
    
    ## Through the bed
    dTf_dt[1:-1] = -u[1:-1] * (T_f[2:] - T_f[:-2]) / (2*dz) + (k_eff_f[1:-1] * (T_f[2:] - 2*T_f[1:-1] + T_f[:-2]) / dz**2 - h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) - h_fw_values[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1])) / (e * rho_f_values[1:-1] * Cp_f_values[1:-1])
    dTs_dt[1:-1] = (k_eff_s[1:-1] * (T_s[2:] - 2*T_s[1:-1] + T_s[:-2]) / dz**2 + h_fs_values[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) + dH * dq_dt[1:-1]) / ((1 - e) * rho_s_values[1:-1] * Cp_s_values[1:-1])
    dTw_dt[1:-1] = (k_w_values[1:-1] * (T_w[2:] - 2*T_w[1:-1] + T_w[:-2]) / dz**2 + h_fw_values[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1]) - h_wa_values[1:-1] * a_w_ext * (T_w[1:-1] - T_a)) / (rho_w_values[1:-1] * Cp_w_values[1:-1])

    return np.concatenate((dTf_dt, dTs_dt, dTw_dt, dx_dt, dq_dt))

## Initial conditions
y0 = np.concatenate((np.ones(Nz) * T_f0, np.ones(Nz) * T_s0, np.ones(Nz) * T_w0, np.ones(Nz)*x_0, np.zeros(Nz)))

## Time span (start, stop, number of points)
t_max = 60*60*3
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, 100)

## Solve ODE system
solution = solve_ivp(reactor_model, t_span, y0, method='Radau', t_eval=t_eval)

## Extract results
T_f = solution.y[:Nz]
T_s = solution.y[Nz:2*Nz]
T_w = solution.y[2*Nz:3*Nz]
x = solution.y[3*Nz:4*Nz]
q = solution.y[4*Nz:]

## Plot results
## Plot temperature function of axial position at different time steps
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

linestyles = {'_s': '-', '_f': '--'}  # Define different linestyles for solid and fluid
colors = [
    "#000000",  # Black
    "#8A2BE2",  # Violet (~400 nm)
    "#4B0082",  # Indigo (~445 nm)
    "#0000FF",  # Blue (~475 nm)
    "#00FFFF",  # Cyan (~500 nm)
    "#00FF00",  # Green (~525 nm)
    "#ADFF2F",  # Yellow-Green (~560 nm)
    "#FFFF00",  # Yellow (~580 nm)
    "#FFA500",  # Orange (~600 nm)
    "#FF4500",  # Reddish-Orange (~625 nm)
    "#FF0000",  # Red (~700 nm)
]  # Define a set of colors for different `i`

for idx, i in enumerate(np.linspace(0, 99, 11, dtype=int)):
    color = colors[idx % len(colors)]  # Cycle through colors for each `i`
    axes[0].plot(z, T_s[:, i], label=f"solid, time {i * t_max / 100}", color=color, linestyle=linestyles['_s'])
    axes[0].plot(z, T_f[:, i], label=f"fluid, time {i * t_max / 100}", color=color, linestyle=linestyles['_f'])
    #axes[0].plot(z, T_w[:, i], label=f"wall, time {i * t_max / 100}")
    axes[1].plot(z, x[:, i], label=f"time {i * t_max / 100}", color=color)
    axes[2].plot(z, q[:, i], label=f"time {i * t_max / 100}", color=color)

axes[0].set_title("Temperatures inside the reactor")
axes[0].set_xlabel("z [m]")
axes[0].set_ylabel("Temp [K]")
axes[0].legend(loc='upper left', bbox_to_anchor=(0,-0.2), ncol=5)

axes[1].set_title("Abolute humidity inside the reactor")
axes[1].set_xlabel("z [m]")
axes[1].set_ylabel("x [kg_water/kg_dryair]")
axes[1].legend()

axes[2].set_title("Reaction inside the reactor")
axes[2].set_xlabel("z [m]")
axes[2].set_ylabel("q [wtf?]")
axes[2].legend()
plt.subplots_adjust(wspace=0.3)
#plt.tight_layout()
# Show the figure
plt.show()


## Plot temperature function of time at outlet
plt.figure(figsize=(12, 8))
plt.plot(z, T_f[-1, :], label='')
## Experimental data for validation


plt.xlabel('Time [s]')
plt.ylabel('Temperature K')
plt.title('Fluid temperature at outlet')
plt.legend()
plt.grid(True)
plt.show()