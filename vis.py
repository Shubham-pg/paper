import matplotlib.pyplot as plt
import numpy as np
import tc_core as tc

def plot_environment_and_cage(env_name):
    # 1. Load your environment and distance fields
    mask = tc.make_env(env_name)
    F = tc.Fields(mask)
    cage = tc.build_cage(F)

    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 3. Plot the continuous signed distance field (SDF)
    im = ax.imshow(F.sdf, cmap='viridis', origin='upper')
    plt.colorbar(im, label='Clearance Distance')

    # 4. Overlay the obstacles in black
    y_obs, x_obs = np.where(mask > 0)
    ax.scatter(x_obs, y_obs, c='black', s=1, alpha=0.5)

    # 5. Overlay your Topological Cage in stark red
    y_c, x_c = np.where(cage['skel'] > 0)
    ax.scatter(x_c, y_c, c='red', s=2, label='Topological Cage')

    # 6. Format and save
    ax.set_title(f'Topological Cage - {env_name}')
    ax.legend(loc='upper right')
    ax.axis('off') # Clean look for academic papers
    plt.tight_layout()
    
    filename = f'Figure_{env_name}_Cage.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Successfully generated {filename}')

if __name__ == '__main__':
    # Generate figures for all four experimental environments
    for env in ['E1', 'E2', 'E3', 'E4']:
        plot_environment_and_cage(env)