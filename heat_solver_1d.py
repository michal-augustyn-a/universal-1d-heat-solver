import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL PLOT CONFIGURATION (Matplotlib Setup)
# =============================================================================
# Standard Matplotlib text rendering instead of a system LaTeX installation
# to guarantee that this script runs seamlessly on any computer out of the box.
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['font.size'] = 12

# Predefined material database mapping real-world physical properties.
# Units: rho [kg/m^3], c_p [J/(kg*C)], k [W/(m*C)].
# D_override is used when experimental data has been pre-calibrated (like your ePAHT-CF).
MATERIAL_DATABASE = {
    '1': {
        'name': 'ePAHT-CF (Nylon Composite Specimen - Lab Test)',
        'label': 'ePAHT-CF',
        'rho': 1100.0, 'c_p': 1300.0, 'k': 0.18, 'D_override': 0.09,
        'default_length': 2.4, 'default_T_init': 20.0, 'default_T_bc': 130.0, 
        'default_T_target': 128.0, 'default_n': 15
    },
    '2': {
        'name': 'Baking Steel (Carbon Steel Pizza Plate in Oven)',
        'label': 'Baking Steel',
        'rho': 7850.0, 'c_p': 480.0, 'k': 50.0, 'D_override': None,
        'default_length': 6.0, 'default_T_init': 20.0, 'default_T_bc': 300.0, 
        'default_T_target': 225.0, 'default_n': 15
    },
    '3': {
        'name': 'Beef Steak (Thick Cut Meat Thermal Simulation)',
        'label': 'Beef Steak',
        'rho': 1050.0, 'c_p': 3400.0, 'k': 0.45, 'D_override': None,
        'default_length': 25.0, 'default_T_init': 4.0, 'default_T_bc': 200.0, 
        'default_T_target': 55.0, 'default_n': 15  # Odd node count guarantees a perfect center node
    }
}


def setup_directory(folder_name='figures'):
    """Finds the script directory and handles automated cleanup of old PNG plots."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, folder_name)

    if os.path.exists(save_path):
        print(f"\nCleaning up old simulation plots from '{folder_name}' folder...")
        for filename in os.listdir(save_path):
            file_path = os.path.join(save_path, filename)
            # Safely clear old frames without deleting non-image files
            if os.path.isfile(file_path) and filename.endswith('.png'):
                os.remove(file_path)
    else:
        os.makedirs(save_path)
        
    return save_path


def initialize_plots(n, length, length_arr, u_initial, min_u, max_u, D, time_actual, material):
    """Initializes the original 3-panel dashboard layout."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.set_figwidth(12), fig.set_figheight(6)
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.85, hspace=0.5)

    # Subplot 1: Heatmap
    ax1.set_xlabel('Cell number $n$', fontsize=11)
    ax1.set_xticks(np.arange(0, n + 0.1, 1), minor=True)
    ax1.set_yticklabels([])
    ax1.tick_params(axis='y', colors='white')

    pcm = ax1.pcolormesh([u_initial], cmap=plt.cm.jet, vmin=min_u, vmax=max_u)
    fig.colorbar(pcm, cax=ax1.inset_axes([1.05, 0.0, 0.05, 1]), label=r'Temperature [°C]',
                 ticks=np.round(np.linspace(min_u, max_u, 5), 0))
    ax1.set_title(f'Solution of 1D heat equation for $D = {D: .2f}$ '
                  r'$\frac{{mm}^2}{s}$,' f' at time $t = {time_actual: .0f}s$, material: {material}')

    # Subplot 2: Temp Profile across thickness
    ax2.grid()
    ax2.set_xlim(length_arr[0], length_arr[-1])
    ax2.set_ylim(bottom=min_u - 10, top=max_u + 10, auto=False)
    ax2.set_xlabel('Length $[mm]$')
    ax2.set_ylabel('Temperature [°C]')
    ax2.tick_params(axis='y', which='both', direction='in', left=True, right=True)
    ax2.set_yticks(np.arange(min_u - 10, max_u + 10, 10), minor=True)
    ax2.set_xticks(np.arange(0, length, max(0.1, length/10)), minor=False)
    ax2.plot(length_arr, u_initial, color='black', marker='.')

    # Subplot 3: Numerical Convergence Logs (Residuals)
    ax3.plot(0, 0, color='blue', linestyle='--', label='Relative RMS of a temperature difference')
    ax3.plot(0, 0, color='green', label='Maximum absolute relative temperature difference')
    ax3.set_yscale('log')
    ax3.set_xlabel(f'$t$ $[s]$')
    ax3.set_ylabel('Temperature change [%]')
    ax3.grid()
    ax3.legend(loc=1)
    
    return fig, ax1, ax2, ax3, pcm


def update_dashboard(fig, ax1, ax2, ax3, pcm, u_current, length, length_arr, min_u, max_u, 
                     D, time_actual, material, time_data, RMS_diff_data, Diff_max_data):
    """Redraws the dashboard configuration for live-view animation."""
    ax1.set_title(f'Solution of 1D heat equation for $D = {D: .2f}$ '
                  r'$\frac{{mm}^2}{s}$,' f' at time $t = {time_actual: .0f}s$, material: {material}')
    pcm.set_array([u_current])

    # Refresh 1D line plot profile
    ax2.cla()
    ax2.grid()
    ax2.set_xlim(length_arr[0], length_arr[-1])
    ax2.set_ylim(bottom=min_u - 10, top=max_u + 10, auto=False)
    ax2.set_xlabel('Length $[mm]$')
    ax2.set_ylabel('Temperature [°C]')
    ax2.tick_params(axis='y', which='both', direction='in', left=True, right=True)
    ax2.set_yticks(np.arange(min_u - 10, max_u + 10, 10), minor=True)
    ax2.set_xticks(np.arange(0, length, max(0.1, length/10)), minor=False)
    ax2.plot(length_arr, u_current, color='black', marker='.')

    # Append new historical data to convergence curves
    ax3.plot(time_data, RMS_diff_data, color='blue', linestyle='--')
    ax3.plot(time_data, Diff_max_data, color='green')
    plt.pause(0.1) # Small pause gives system time to render the graphics window


def get_user_config():
    """Interactive text-driven menu to load presets or input fully custom thermal configurations."""
    print("==================================================")
    print("       UNIVERSAL 1D HEAT TRANSFER SOLVER         ")
    print("==================================================")
    print("Select a profile material from the library:")
    for key, mat in MATERIAL_DATABASE.items():
        print(f"  [{key}] {mat['name']}")
    print("  [4] Custom Material Entry (Provide your own parameters)")
    
    choice = input("Enter choice (1-4, default is 1): ").strip()
    if choice not in ['1', '2', '3', '4']:
        choice = '1'
        
    if choice == '4':
        # Prompting for custom material physics parameters manually
        config = {
            'material': input("Enter material name [Default: Custom]: ").strip() or 'Custom',
            'rho': float(input("Density rho [kg/m^3]: ")),
            'c_p': float(input("Specific heat c_p [J/(kg*C)]: ")),
            'k': float(input("Thermal conductivity k [W/(m*C)]: ")),
            'D_override': None,
            'length': float(input("Object thickness [mm]: ")),
            'T_init': float(input("Initial object temperature [°C]: ")),
            'T_left': float(input("Left side boundary temperature [°C]: ")),
            'T_right': float(input("Right side boundary temperature [°C]: ")),
            'T_target': float(input("Target CORE center temperature to achieve [°C]: ")),
            'n': int(input("Accuracy - Number of nodes (Use odd numbers like 15): ")),
        }
    else:
        preset = MATERIAL_DATABASE[choice]
        print(f"\nSelected preset: {preset['name']}\nPress Enter to keep default values.")
        
        len_in = input(f"Thickness [mm] (default {preset['default_length']}): ").strip()
        t_init_in = input(f"Initial temperature [°C] (default {preset['default_T_init']}): ").strip()
        
        print("Set independent boundary temperatures (e.g., hot pan vs ambient room air):")
        t_left_in = input(f" - Left boundary temp [°C] (default {preset['default_T_bc']}): ").strip()
        t_right_in = input(f" - Right boundary temp [°C] (default {preset['default_T_bc']}): ").strip()
        
        # ASK FOR TARGET CORE TEMPERATURE: Let the user define the exact thermal milestone
        t_target_in = input(f"Target CORE center temperature [°C] (default {preset['default_T_target']}): ").strip()
        
        n_in = input(f"Accuracy - Node count / Mesh resolution (default {preset['default_n']}): ").strip()
        
        config = {
            'material': preset['label'],
            'rho': preset['rho'], 'c_p': preset['c_p'], 'k': preset['k'], 'D_override': preset['D_override'],
            'length': float(len_in) if len_in else preset['default_length'],
            'T_init': float(t_init_in) if t_init_in else preset['default_T_init'],
            'T_left': float(t_left_in) if t_left_in else preset['default_T_bc'],
            'T_right': float(t_right_in) if t_right_in else preset['default_T_bc'],
            'T_target': float(t_target_in) if t_target_in else preset['default_T_target'],
            'n': int(n_in) if n_in else preset['default_n']
        }
        
    animate_in = input("\nDo you want to see live animation frames? (y/n, default is y): ").strip().lower()
    config['animate'] = False if animate_in == 'n' else True
    return config


def run_simulation(cfg):
    """Executes the core FDM simulation loop monitoring core node progress."""
    material = cfg['material']
    length = cfg['length']
    n = cfg['n']
    T_init = cfg['T_init']
    T_left = cfg['T_left']
    T_right = cfg['T_right']
    T_target = cfg['T_target']
    animate = cfg['animate']

    # 1. Thermal Diffusivity Calculation (D = k / (rho * c_p))
    # Represents the rate at which heat distributes through a medium.
    if cfg['D_override'] is not None:
        D = cfg['D_override']
    else:
        D = cfg['k'] / (cfg['rho'] * cfg['c_p'])
        D = D * 10 ** 6  # convert from m^2/s to mm^2/s

    # 2. Grid Discretization and Stability Limit Assessment
    # FDM equations require 'dt' to respect the Courant-Friedrichs-Lewy condition (alpha <= 0.5)
    # to guarantee numerical mathematical convergence without oscillating out of bounds.
    dx = length / n
    dt = 0.5 * dx ** 2 / D
    time_step = int(0)
    time_actual = 0
    alpha = D * dt / dx ** 2
    length_arr = np.linspace(0, length, n)

    # Initialize thermal grid matrix array
    u = np.ones(n) * T_init
    u[0] = T_left
    u[-1] = T_right
    u_new = u
    min_u, max_u = min(u), max(u)

    print(f'\n[Config Log] Diffusion D: {D:.4f} mm^2/s, Safe dt: {dt:.5f} s, Core Target: {T_target}°C')

    # Data registers for convergence histories
    RMS_diff_data = np.array([], dtype=np.float64)
    Diff_max_data = np.array([], dtype=np.float64)
    time_data = np.array([], dtype=np.float64)

    save_path = setup_directory('figures')
    
    # Render and export initial plot frame state only if animation flag is active
    if animate:
        fig, ax1, ax2, ax3, pcm = initialize_plots(n, length, length_arr, u_new, min_u, max_u, D, time_actual, material)
        filename = os.path.join(save_path, f'{time_step}_{material}_Boundary_conditions.png')
        plt.savefig(filename)
        print("\nProcessing simulation with interactive GUI animation frames...")
    else:
        print("\nProcessing high-speed calculations in the background...")

    # Main Simulation Time Loop
    while True:
        u_old = u.copy()

        # Spatial loop: Finite Difference Scheme
        for i in range(1, n - 1):  
            u_new[i] = u_old[i] + alpha * (u_old[i - 1] - 2 * u_old[i] + u_old[i + 1])
            
        u_new[0] = T_left
        u_new[-1] = T_right

        # Residual logging
        RMS_diff_iter = 100 * np.sqrt((1 / n) * np.sum((u_old - u_new) ** 2)) / np.sum(u_old)
        Diff_max_iter = 100 * np.max(np.abs((u_old - u_new) / u_old))

        RMS_diff_data = np.append(RMS_diff_data, RMS_diff_iter)
        Diff_max_data = np.append(Diff_max_data, Diff_max_iter)
        time_data = np.append(time_data, time_actual)

        # Track the exact center node representing the core axis of the material
        core_node_idx = n // 2
        current_core_temp = u_new[core_node_idx]

        if animate:
            if time_step % 20 == 0:
                print(f'Time: {time_actual:.2f}s | Core Temp: {current_core_temp:.1f}°C / Target: {T_target}°C')
            update_dashboard(fig, ax1, ax2, ax3, pcm, u_new, length, length_arr, min_u, max_u, 
                             D, time_actual, material, time_data, RMS_diff_data, Diff_max_data)
            
            if time_step % 10 == 0:
                filename = os.path.join(save_path, f'{time_step}_{material}_{time_actual:.1f}s.png')
                plt.savefig(filename)

        time_actual += dt
        time_step += 1
        
        # =============================================================================
        # DYNAMIC GOAL TERMINATION CRITERION
        # =============================================================================
        # The solver checks if the core center has reached the user's customized target.
        # It handles both standard heating (T_target achieved) and steady-state fallbacks.
        has_reached_target = (T_left > T_init and current_core_temp >= T_target) or \
                             (T_left < T_init and current_core_temp <= T_target)
                             
        is_steady_state = (RMS_diff_iter < 0.0001 or Diff_max_iter < 0.01) and time_actual > 2 * dt

        if has_reached_target or is_steady_state:
            print(f'\n[Goal Achieved] Target condition met!')
            print(f'Final Core Temperature: {current_core_temp:.2f}°C')
            print(f'Total process time required: {time_actual:.2f} seconds ({time_actual / 60:.2f} minutes).')
            
            if not animate:
                fig, ax1, ax2, ax3, pcm = initialize_plots(n, length, length_arr, u_new, min_u, max_u, D, time_actual, material)
                ax3.plot(time_data, RMS_diff_data, color='blue', linestyle='--')
                ax3.plot(time_data, Diff_max_data, color='green')
                
            filename = os.path.join(save_path, f'{time_step}_{material}_Reached_Target_{T_target}C.png')
            plt.savefig(filename)
            break

    print('Calculations completed.')
    plt.show()


if __name__ == '__main__':
    user_cfg = get_user_config()
    run_simulation(user_cfg)