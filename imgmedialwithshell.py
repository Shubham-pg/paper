
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt, binary_closing, binary_dilation
from skimage.morphology import medial_axis

# --- CONFIGURATION & SAVING ---
SAVE_FILENAME = "sdf_medial_axis_map.png"
DPI = 300

# --- FLEXIBLE PARAMETERS: SDF ---
START_LEVEL = 0.5      
STEP_SIZE = 2.0        
NUM_LEVELS = 30        
COLOR_MAP = 'plasma'   
SDF_LINE_WIDTH = 0.5  

# --- FLEXIBLE PARAMETERS: MEDIAL AXIS & SHELL ---
AXIS_COLOR = '#00FFFF' # Cyan
AXIS_ALPHA = 1.0       
AXIS_THICKNESS = 2.0   
OFFSET = 12.0          
SHELL_COLOR = '#FF3333' # Red shell limit
SHELL_ALPHA = 1.0      
SHELL_THICKNESS = 2.0  

# --- FLEXIBLE PARAMETERS: VIEW ---
ZOOM_TO_FIT = True     
PADDING = 30           

def get_enclosed_highway(sdf, ridges, gx, gy):
    """Calculates the outer shell bounding the safe highway zone."""
    mag = np.hypot(gx, gy)
    parallel_mask = (ridges > 0) & (mag <= 0.17) 
    max_val = np.max(sdf[parallel_mask]) if np.any(parallel_mask) else 15.0
    limit = max_val + OFFSET
    
    highway_mask = (ridges > 0) & (sdf <= limit)
    shell_mask = (np.abs(sdf - limit) <= 1.5)
    combined = (highway_mask | shell_mask)
    return binary_closing(combined, structure=np.ones((3,3))), limit

# 1. Load Data
try:
    data = np.load("map_data.npz")
    canvas = data['canvas']
except FileNotFoundError:
    # Generate dummy data if file is missing
    canvas = np.zeros((400, 400), dtype=np.uint8)
    canvas[100:300, 100:300] = 255
    canvas = 255 - canvas

# 2. Metric Fields
sdf = distance_transform_edt(canvas == 0)
gy, gx = np.gradient(sdf)
max_dist = np.max(sdf)

# 3. Topology & Shell
skeleton, distance = medial_axis(canvas == 0, return_distance=True)
highway_mask, highway_limit = get_enclosed_highway(sdf, skeleton, gx, gy)
skeleton = skeleton & (highway_mask > 0)

# 4. Setup Plot
fig, ax = plt.subplots(figsize=(12, 12), dpi=DPI) 
ax.set_facecolor('#0a0a0a')
fig.patch.set_facecolor('#0a0a0a')

# 5. Rendering
# Faint walls
masked_walls = np.ma.masked_where(canvas == 0, canvas)
ax.imshow(masked_walls, cmap='gray', vmin=0, vmax=255, alpha=0.1) 

# SDF Contours
levels = [START_LEVEL + (i * STEP_SIZE) for i in range(NUM_LEVELS) if (START_LEVEL + i*STEP_SIZE) <= max_dist]
contours = ax.contour(sdf, levels=levels, cmap=COLOR_MAP, 
                      linewidths=SDF_LINE_WIDTH, alpha=0.6, zorder=1)

# Outer Shell
ax.contour(sdf, levels=[highway_limit], colors=[SHELL_COLOR], 
           linewidths=SHELL_THICKNESS, alpha=SHELL_ALPHA, zorder=2)

# Medial Axis (Pixel-perfect overlay)
iterations = max(0, int(AXIS_THICKNESS // 2))
skel_vis = binary_dilation(skeleton, iterations=iterations) if iterations > 0 else skeleton
rgba_skeleton = np.zeros((canvas.shape[0], canvas.shape[1], 4), dtype=np.float32)
r, g, b = mcolors.to_rgb(AXIS_COLOR)
rgba_skeleton[skel_vis, :3] = [r, g, b]
rgba_skeleton[skel_vis, 3] = AXIS_ALPHA
ax.imshow(rgba_skeleton, zorder=3, interpolation='nearest')

# 6. View & Polish
if ZOOM_TO_FIT and levels:
    y_idx, x_idx = np.where((sdf >= START_LEVEL) & (sdf <= levels[-1] + STEP_SIZE))
    if len(y_idx) > 0:
        ax.set_xlim(max(0, x_idx.min() - PADDING), min(canvas.shape[1], x_idx.max() + PADDING))
        ax.set_ylim(min(canvas.shape[0], y_idx.max() + PADDING), max(0, y_idx.min() - PADDING))

# Styling
ax.axis('off')

plt.tight_layout()

# 7. Save and Show
plt.savefig(SAVE_FILENAME, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"Image saved successfully as {SAVE_FILENAME}")
plt.show()