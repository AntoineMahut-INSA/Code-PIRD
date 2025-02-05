# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
## Materials constants
a_air = 33.5e-6 # Air thermal diffusivity at 20°C (m²/s)
nu_air = 15.1e-6 # Air kinematic viscosity at 20°C (m²/s)
k_air = 25.9e-3 # Air thermal conductivity at 20°C (W/m/K)
rho_z = 1080 # Zeolite density, including pores (kg/m^3)
Cp_z = 830 # Zeolite heat capacity (J/kg)
k_z = 0.2 # Zeolite conductivity (W/m/K)
rho_syl = 1030 # Sylgard density (kg/m^3)
Cp_syl = 1500 # Sylgard heat capacity (J/kg)
k_syl = 0.27 # Sylgard conductivity (W/m/K)
rho_i = 2500 # Inert core density (kg/m^3)
Cp_i = 850 # Inert core heat capacity (J/kg)
k_i = 1 # Inert core conductivity (W/m/K)
M_v = 0.018 # Water molar mass (kg/mol)
M_da = 0.029 # Dry air molar mass (kg/mol)

## Reactor parameters
L = 0.2  # Reactor length (m)
D = 0.72  # Reactor bed diameter (m)
r = D / 2
w_t = 0.2 # Wall thickness (m)
D_ext = D + 2*w_t # Reactor external diameter (m)
r_ext = D_ext / 2
e = 0.4  # Bed void fraction
Ds_i = 0.01 # Inert particle center diameter (m)
rs_i = Ds_i / 2
c_t = 0.5e-3 # Coating thickness (m)
Ds = Ds_i + 2*c_t # Particle diameter (m)
rs = Ds / 2
c_v = 4/3*np.pi*(rs**3-rs_i**3) # Coating volume
Vs = 4/3*np.pi*rs**3
Vs_i = 4/3*np.pi*rs_i**3
V_c = Vs - Vs_i

f_c_v = 1 - (Ds_i/Ds)**3 # Coating volume fraction
f_z_m = 0.48 # Coating zeolite mass fraction
rho_c = 1 / (f_z_m / rho_z + (1 - f_z_m) / rho_syl) # Coating density (kg/m^3)
ms = Vs_i * rho_i + (Vs - Vs_i) * rho_c # Particle mass
f_c_m = rho_c * V_c / ms # Coating mass fraction
f_z_v = (f_z_m / rho_z) / (f_z_m / rho_z + (1 - f_z_m) / rho_syl) # Coating zeolite volume fraction


Q_f = 0.06015 # Fluid flow rate (kg/s) (=180 m3/h at 20°C/70% RH)
Ba = (1-e)*f_c_v*f_z_m*rho_c # Zeolite fraction (kg zeolite/m^3 of reactor)

## Reaction parameters and constants
R = 8.31 # ideal gas constant (J/K/mol)
dH = 3800e3 # Synthesis reaction enthalpy (J/kgw)
#dH_m = 1.2e6
Ea = 4e4 # Arrhenius activation energy

# Adsorption Gondre constants
k_vs = 0.032
D_0 = 4e-7
# b_0 = 5e4
# a = 3.04
# q_n1 = 0.84
# q_n0 = -198
# q_cap1 = 0.074
# q_cap2 = -4.7e-5
# q_cap3 = -3.9e-3


## Operating conditions
T_in = 293  # Inlet temperature (K)
T_f0 = 295  # Initial fluid temperature (K)
T_s0 = 296  # Initial solid temperature (K)
T_w0 = T_f0  # Initial wall temperature (K)
T_a = 294  # Ambient temperature (K)
T_in_charge = 453 # Charging temperature (K)

P_amb = 101325 # Ambient pressure (Pa)
P_in = P_amb # Inlet pressure (Pa)
phi_in = 0.7 # Inlet relative humidity
p_vs = np.exp(23.1964-3816.44/(T_in - 46.13)) # Antoine equation for inlet vapor saturation pressure (Pa)
p_in = phi_in * p_vs # Inlet vapor pressure (Pa)
p_0 = 0 # Initial vapor pressure (Pa)


## Calculated thermophysical parameters
#q_n = q_n1 * T_in_charge + q_n0  # Gondre : Eq III.4
#q_cap = q_cap1 * T_in_charge + q_cap2 * T_in + q_cap3  # Gondre : Eq III.3

## Calculated geometric parameters
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
def rho_da_values(T,P): # Dry air density function of T_f and total pressure (kg/m^3)
    return P / (287.1*T)

def rho_f_values(T,p_v): # Air density, kg/m^3
    return (P_in*M_da + p_v*M_v) / (8.31 * T)

def Cp_da_values(T):
    return 1.006+0*T

def Cp_v_values(T):
    return 1.88+0*T

def Cp_water_values(T): # Water specific heat capacity (J/kg/K)
    return 4.18e3+0*T

def Cp_f_values(x):
    return 1.82*x+1.005

def k_f_values(T):
    return 0.03+0*T

def mu_f_values(T):
    return 4.564e-8*T+4.745e-6

def rho_s_values(T):
    return (1-f_c_v)*rho_i + f_c_v*f_z_v*rho_z + f_c_v*(1-f_z_v)*rho_syl +0*T

def Cp_s_values(T):
    return (1-f_c_m)*Cp_i + f_c_m*f_z_m*Cp_z + f_c_v*(1-f_z_v)*Cp_syl +0*T

def k_s_values(T):
    return rs_i/rs*k_i + c_t*rs*(f_z_v*k_z + (1-f_z_v)*k_syl) + 0*T

def rho_w_values(T):
    return 180+0*T

def Cp_w_values(T):
    return 1e3+0*T

def k_w_values(T):
    return 0.07+0*T

def p_vs_values(T):
    return np.exp(23.2-3816/(T-46.1))

def phi_values(T,p_v):
    """
    Returns humidity ratio W [1] as W= p_v / p_vs
    :param T: fluid temperature. [K]
    :param p_v: Vapor pressure [Pa]
    """
    return p_v / p_vs_values(T)

def q_e_values(RH): # Affine approx. from Ferreira et al. for 45°C isotherm
    return np.select([RH < 0.3, RH > 0.4], [0.01456876 * RH, 0.1195943 * RH + 0.1211077], default=1.676895 * RH - 0.498698)    

    # for i in range(Nz):
    #     if RH[i] < 0.3:
    #         return 0.01456876*RH[i]
    #     elif RH[i] > 0.4:
    #         return 1.676895*RH[i] - 0.498698 
    #     else:
    #         return 0.1195943 * RH[i] + 0.1211077

def reactor_model(t, y): 
    T_f = y[:Nz]
    T_s = y[Nz:2*Nz]
    T_w = y[2*Nz:3*Nz]
    p = y[3*Nz:4*Nz] # Vapor pressure (Pa)
    q = y[4*Nz:] # Adsorbed layer density (kgw/m^3)
    
    dTf_dt = np.zeros(Nz)
    dTs_dt = np.zeros(Nz)
    dTw_dt = np.zeros(Nz)
    dp_dt = np.zeros(Nz)
    dq_dt = np.zeros(Nz)
    
    # Temperature-dependent properties
    rho_f = rho_f_values(T_f,p)
    Cp_f = Cp_f_values(T_f)
    k_f = k_f_values(T_f)
    mu_f = mu_f_values(T_f)
    rho_s = rho_s_values(T_s)
    Cp_s = Cp_s_values(T_s)
    k_s = k_s_values(T_s)
    rho_w = rho_w_values(T_w)
    Cp_w = Cp_w_values(T_w)
    k_w = k_w_values(T_w)
    
    phi = phi_values(T_f, p)
    rho_da = rho_da_values(T_f, P_in)
    Cp_da = Cp_da_values(T_f)
    Cp_v = Cp_v_values(T_f)
    Cp_water = Cp_water_values(T_f)
    
    # Following equations mostly from : A review on experience feedback and numerical modeling of packed-bed thermal energy storage systems (Esence et al., 2016) => validées avec modèle sensible
    u = Q_f / (rho_f * e * A) # Interstitial velocity (m/s)
    u_sup = e * u # Superficial velocity (m/s) = 0.123
    Re = rho_f * u_sup * Ds / mu_f
    Pr = mu_f * Cp_f / k_f
    Ra_a = 9.81 / (T_a*nu_air*a_air) * (T_w-T_a) * L**3
    Nu_a = 0.56*Ra_a**0.25 # Nusselt values for air around the cylinder
    h_fs = k_f / Ds * (2 + 1.1 * Re**0.6 * Pr**(1/3)) # Correlation by Wakao et al. (1979)
    h_fw = k_f / L * (0.12 * Pr**(1/3) * Re**0.75) # Correlation by Kunii and Suzuki (1968) with Re > 100
    h_wa = k_air * Nu_a / L # Nusselt number correlation with 10^4 < Ra < 10^9 ; Rayleigh number calculation with Pr = 0.71 for air at 25°C
    beta = (k_s - k_f) / (k_s + 2 * k_f) # Parameter for fluid-solid thermal conductivity calculation
    k_eff_0 = k_f * (1 + 2 * beta * (1 - e) + (2 * beta**3 - 0.1 * beta) * (1 - e)**2 + (1 - e)**3 * 0.05 * np.exp(4.5 * beta)) / (1 - beta * (1 - e)) # Effective fluid-solid thermal conductivity of the bed due to conduction (Gonzo, 2002)
    f = (k_eff_0 - e * k_f - (1 - e) * k_s) / (k_f - k_s) # Tortuosity of the bed
    k_eff_f = (e + f + 0.5*Re*Pr) * k_f # Effective fluid thermal conductivity
    k_eff_s = (1 - e - f) * k_s # Effective solid thermal conductivity (W/m/K)
    
    #Bi = h_fs * Ds / (6 * k_s) # Biot number, must be < 0.1 for solids thermal gradient to be negligible
    
    ## Gondre isotherm
    k_m = 15*D_0 / c_t**2 * np.exp(-Ea / (R * T_f)) + k_vs * u_sup # LDF time constant, Gondre : Eq III.1 ; between 1.4 and 4 for STAID
    #b = b_0 * np.exp(-dH_m * M_v / (R * T_f)) # Gondre : Eq III.2
    #q_e = b * phi * q_n / (1 + b * phi) + a*phi + q_cap * phi / (1-phi)
    
    ## Ferreira et al. isotherm at 45°C
    q_e = q_e_values(phi)
    ## Sorption
    dq_dt = k_m * (q_e - q) # Gondre : Eq II.24

    ## Vapour mass conservation
    dp_dt[0] = p[0] / T_f[0] * dTf_dt[0] - u[0] * T_f[0] / e * (T_f[0]*(p[0]-p_in)-p[0]*(T_f[0]-T_in)) / (dz*T_f[0]**2)
    dp_dt[1:] = p[1:] / T_f[1:] * dTf_dt[1:] - u[1:] * T_f[1:] / (e * dz) * (p[1:] / T_f[1:] - p[0:-1] / T_f[0:-1])
    dp_dt = dp_dt - Ba * R * T_f * dq_dt / (e*M_v)

    ## Energy balances
    ## Boundaries
    # Fluid phase
    # dTf_dt[0] =  1 / (e * rho_da[1:-1] * Cp_da[1:-1]) * (k_eff_f[0] * (T_f[1] - 2*T_f[0] + T_in) / dz**2 - h_fs[0] * a_fs * (T_f[0] - T_s[0]) - h_fw[0] * a_w_int * (T_f[0] - T_w[0]) - u[0] * (rho_da[0] * Cp_da[0] * (T_f[1] - T_in) / (2*dz) + M_v * Cp_v[0] / R * (p[1] - p_in) / (2*dz))) - M_v * Cp_v[0] / (R * rho_da[0] * Cp_da[0]) * dp_dt[0]
    dTf_dt[0] = k_eff_f[0] * (T_f[1] - 2*T_f[0] + T_in) / dz**2
    dTf_dt[0] = dTf_dt[0] - h_fs[0] * a_fs * (T_f[0] - T_s[0])
    dTf_dt[0] = dTf_dt[0] - h_fw[0] * a_w_int * (T_f[0] - T_w[0])
    dTf_dt[0] = dTf_dt[0] - u[0] * (rho_da[0] * Cp_da[0] * (T_f[1] - T_in) / (2*dz) + M_v * Cp_v[0] / R * (p[1] - p_in) / (2*dz))
    dTf_dt[0] = dTf_dt[0] / (e * rho_da[0] * Cp_da[0])
    dTf_dt[0] = dTf_dt[0] - M_v * Cp_v[0] / (R * rho_da[0] * Cp_da[0]) * dp_dt[0]
    
    #dTf_dt[-1] = 1 / (e * rho_da[-1] * Cp_da[-1]) * (- h_fs[-1] * a_fs * (T_f[-1] - T_s[-1]) - h_fw[-1] * a_w_int * (T_f[-1] - T_w[-1])) - M_v * Cp_v[-1] / (R * rho_da[-1] * Cp_da[-1]) * dp_dt[-1]
    dTf_dt[-1] = - h_fs[-1] * a_fs * (T_f[-1] - T_s[-1])
    dTf_dt[-1] = dTf_dt[-1] - h_fw[-1] * a_w_int * (T_f[-1] - T_w[-1])
    dTf_dt[-1] = dTf_dt[-1] / (e * rho_da[-1] * Cp_da[-1])
    dTf_dt[-1] = dTf_dt[-1] - M_v * Cp_v[-1] / (R * rho_da[-1] * Cp_da[-1]) * dp_dt[-1]

    # Solid phase
    dTs_dt[0] = (k_eff_s[0] * (T_s[1] - T_s[0]) / dz**2 + h_fs[0] * a_fs * (T_f[0] - T_s[0]) + Ba * (dH - Cp_water[0] * T_s[0]) * dq_dt[0]) / ((1 - e) * rho_s[0] * Cp_s[0] + Ba * q[0] * Cp_water[0])
    dTs_dt[-1] = (k_eff_s[-1] * (T_s[-2] - T_s[-1]) / dz**2 + h_fs[-1] * a_fs * (T_f[-1] - T_s[-1]) + Ba * (dH - Cp_water[-1] * T_s[-1]) * dq_dt[-1]) / ((1 - e) * rho_s[-1] * Cp_s[-1] + Ba * q[-1] * Cp_water[-1])
    
    # Wall
    dTw_dt[0] = (k_w[0] * (T_w[1] - T_w[0]) / dz**2 + h_fw[0] * a_w_int * (T_f[0] - T_w[0]) - h_wa[0] * a_w_ext * (T_w[0] - T_a)) / (rho_w[0] * Cp_w[0])
    dTw_dt[-1] = (k_w[-1] * (T_w[-2] - T_w[-1]) / dz**2 + h_fw[-1] * a_w_int * (T_f[-1] - T_w[-1]) - h_wa[-1] * a_w_ext * (T_w[-1] - T_a)) / (rho_w[-1] * Cp_w[-1])
    
    ## Through the bed
    # dTf_dt[1:-1] = 1 / (e * rho_da[1:-1] * Cp_da[1:-1]) * (k_eff_f[1:-1] * (T_f[2:] - 2*T_f[1:-1] + T_f[:-2]) / dz**2 - h_fs[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) - h_fw[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1]) - u[1:-1] * (rho_da[1:-1] * Cp_da[1:-1] * (T_f[2:] - T_f[:-2]) / (2*dz) + M_v * Cp_v[1:-1] / R * (p[2:] - p[:-2]) / (2*dz))) - M_v * Cp_v[1:-1] / (R * rho_da[1:-1] * Cp_da[1:-1]) * dp_dt[1:-1]
    dTf_dt[1:-1] = k_eff_f[1:-1] * (T_f[2:] - 2*T_f[1:-1] + T_f[:-2]) / dz**2 # Conduction
    dTf_dt[1:-1] = dTf_dt[1:-1] - h_fs[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) # Convective exchange with solid
    dTf_dt[1:-1] = dTf_dt[1:-1] - h_fw[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1]) # Convective exchange with wall
    dTf_dt[1:-1] = dTf_dt[1:-1] - u[1:-1] * (rho_da[1:-1] * Cp_da[1:-1] * (T_f[2:] - T_f[:-2]) / (2*dz) + M_v * Cp_v[1:-1] / R * (p[2:] - p[:-2]) / (2*dz)) # Transport
    dTf_dt[1:-1] = dTf_dt[1:-1] / (e * rho_da[1:-1] * Cp_da[1:-1]) # dTf_dt multiplicator
    dTf_dt[1:-1] = dTf_dt[1:-1] - M_v * Cp_v[1:-1] / (R * rho_da[1:-1] * Cp_da[1:-1]) * dp_dt[1:-1] # Vapor pressure change
    
    dTs_dt[1:-1] = (k_eff_s[1:-1] * (T_s[2:] - 2*T_s[1:-1] + T_s[:-2]) / dz**2 + h_fs[1:-1] * a_fs * (T_f[1:-1] - T_s[1:-1]) + Ba * (dH - Cp_water[1:-1] * T_s[1:-1]) * dq_dt[1:-1]) / ((1 - e) * rho_s[1:-1] * Cp_s[1:-1] + Ba * q[1:-1] * Cp_water[1:-1])
    dTw_dt[1:-1] = (k_w[1:-1] * (T_w[2:] - 2*T_w[1:-1] + T_w[:-2]) / dz**2 + h_fw[1:-1] * a_w_int * (T_f[1:-1] - T_w[1:-1]) - h_wa[1:-1] * a_w_ext * (T_w[1:-1] - T_a)) / (rho_w[1:-1] * Cp_w[1:-1])
    return np.concatenate((dTf_dt, dTs_dt, dTw_dt, dp_dt, dq_dt))

## Initial conditions
y0 = np.concatenate((np.ones(Nz) * T_f0, np.ones(Nz) * T_s0, np.ones(Nz) * T_w0, np.ones(Nz)*p_0, np.zeros(Nz)))

## Time span (start, stop, number of points)
t_max = 3600*1
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, 100)

## Solve ODE system
solution = solve_ivp(reactor_model, t_span, y0, method='Radau', t_eval=t_eval)

## Extract results
T_f = solution.y[:Nz]
T_s = solution.y[Nz:2*Nz]
T_w = solution.y[2*Nz:3*Nz]
p = solution.y[3*Nz:4*Nz]
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

# compute equilibrium sorption
#b = b_0 * np.exp(-dH_m * M_v / (R * T_f))
#q_e = b * phi_values(T_f, p) * q_n / (1 + b * phi_values(T_f, p))  #+ a*phi_values(T_f, p) + q_cap * phi_values(T_f, p) / (1-phi_values(T_f, p))


for idx, i in enumerate(np.linspace(0, 99, 11, dtype=int)):
    color = colors[idx % len(colors)]  # Cycle through colors for each `i`
    axes[0].plot(z, T_s[:, i], label=f"solid, time {i * t_max / 100}", color=color, linestyle=linestyles['_s'])
    axes[0].plot(z, T_f[:, i], label=f"fluid, time {i * t_max / 100}", color=color, linestyle=linestyles['_f'])
    #axes[0].plot(z, T_w[:, i], label=f"wall, time {i * t_max / 100}")
    axes[1].plot(z, p[:, i], label=f"time {i * t_max / 100}", color=color)
    axes[2].plot(z, q[:, i], label=f"time {i * t_max / 100}", color=color)
 #   axes[2].plot(z, q_e[:, i], label=f"time {i * t_max / 100}", color=color, linestyle=linestyles['_f'])

axes[0].set_title("Temperatures inside the reactor")
axes[0].set_xlabel("z [m]")
axes[0].set_ylabel("Temp [K]")
axes[0].legend(loc='upper left', bbox_to_anchor=(0,-0.2), ncol=5)

axes[1].set_title("Vapor Pressure inside the reactor")
axes[1].set_xlabel("z [m]")
axes[1].set_ylabel("p [Pa]")
#axes[1].legend()

axes[2].set_title("Reaction inside the reactor")
axes[2].set_xlabel("z [m]")
axes[2].set_ylabel("q [kgw/kgz]")
#axes[2].legend()
plt.subplots_adjust(wspace=0.3)
plt.tight_layout()
# Show the figure
plt.show()


## Plot temperature function of time at outlet
plt.figure(figsize=(12, 8))
plt.plot(t_eval, T_f[-1, :], label='')
## Experimental data for validation


plt.xlabel('Time [s]')
plt.ylabel('Temperature K')
plt.title('Fluid temperature at outlet')
plt.grid(True)
plt.show()