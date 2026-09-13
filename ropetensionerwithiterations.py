import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# --- HYPERPARAMETERS: SPRING-MASS SYSTEM ---
INITIAL_NODE_SPACING = 8.0  # (Pixels) Initialize exactly 1 node every 8 pixels
NODE_REST_DIST = 5.0        # (Pixels) Ideal spring length. (Lower than spacing = constant global tension)
W_SPRING = 0.55             # Spring stiffness
W_REPEL = 2.5               # Repulsion force from walls

# --- HYPERPARAMETERS: DISTRIBUTED POPPING & SAFETY ---
ITERATIONS = 1000           # Total physics frames for the video
POP_INTERVAL = 20           # Frames to wait between pops to let the spring-mass physics settle
POP_STRIDE = 10             # The "Uniformity" factor: Pop every 10th node across the WHOLE rope!
MAX_PENETRATION = 1.5       # Trigger an UNDO if the rope penetrates > 1.5px into the trimmed band
# -------------------------------------------

# 1. Load Data from Planner
try:
    data = np.load("planner_data.npz")
    path_pixels = data['path']
    thresh_rigid = data['thresh_rigid']
    thresh_soft = data['thresh_soft']
    sdf_rigid = data['sdf_rigid']
    sdf_soft = data['sdf_soft']
    mask = data['canvas']
except FileNotFoundError:
    print("Error: 'planner_data.npz' not found. Run the planner script first.")
    exit()

# 2. Compute Gradients for Repulsion Physics
gy_r, gx_r = np.gradient(sdf_rigid)
gy_s, gx_s = np.gradient(sdf_soft)

# 3. PROPORTIONAL NODE INITIALIZATION (Distance-based sampling)
# Calculate the exact cumulative distance along the raw pixel path
dists = np.zeros(len(path_pixels))
dists[1:] = np.hypot(np.diff(path_pixels[:, 0]), np.diff(path_pixels[:, 1]))
cum_dists = np.cumsum(dists)
total_len = cum_dists[-1]

# Interpolate nodes so they are exactly INITIAL_NODE_SPACING apart
num_nodes = max(3, int(total_len / INITIAL_NODE_SPACING))
target_dists = np.linspace(0, total_len, num_nodes)
P_y = np.interp(target_dists, cum_dists, path_pixels[:, 0])
P_x = np.interp(target_dists, cum_dists, path_pixels[:, 1])
P = np.column_stack((P_y, P_x)).astype(np.float32)

# 4. Generate the Static Base Map (Pre-rendered for fast video playback)
base_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
base_img[mask == 0] = [30, 30, 30]

# Draw the uniquely trimmed bands loaded from the file
base_img[(mask == 0) & (sdf_soft < thresh_soft)] = [80, 50, 30] 
base_img[(mask == 0) & (sdf_rigid < thresh_rigid)] = [30, 30, 100] 

base_img[mask == 128] = [100, 100, 100]  
base_img[mask == 255] = [200, 200, 200]  

# Setup the Real-Time Window
cv2.namedWindow("Real-Time Rope Tensioner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Rope Tensioner", 1200, 800)
print("Starting True Spring-Mass Simulation...")

pops_done = 0
popping_active = True
last_safe_P = P.copy()

# --- PHYSICS LOOP (VIDEO FRAMES) ---
for frame in range(ITERATIONS):
    
    # 1. DISTRIBUTED POPPING MECHANIC (Global tightening)
    if popping_active and frame % POP_INTERVAL == 0 and frame > 0:
        last_safe_P = P.copy() # Save state before we pop
        
        if len(P) > POP_STRIDE * 2: # Ensure we have enough nodes left to safely pop
            # Identify every Nth node to delete, ensuring we never delete Start (0) or Goal (-1)
            indices_to_delete = np.arange(POP_STRIDE, len(P) - 1, POP_STRIDE)
            
            P = np.delete(P, indices_to_delete, axis=0)
            pops_done += len(indices_to_delete)

    P_new = P.copy()
    max_pen_this_frame = 0.0

    # 2. CALCULATE PHYSICS FORCES
    for i in range(1, len(P)-1):
        y, x = int(P[i, 0]), int(P[i, 1])
        y = np.clip(y, 0, mask.shape[0]-1)
        x = np.clip(x, 0, mask.shape[1]-1)

        # Force A: Spring Mass Tension (Hooke's Law)
        v1 = P[i-1] - P[i]
        d1 = np.hypot(v1[0], v1[1]) + 1e-5
        f1 = (v1 / d1) * (d1 - NODE_REST_DIST)

        v2 = P[i+1] - P[i]
        d2 = np.hypot(v2[0], v2[1]) + 1e-5
        f2 = (v2 / d2) * (d2 - NODE_REST_DIST)

        tension = f1 + f2

        # Force B: External Repulsion from Obstacle Bands
        repel = np.array([0.0, 0.0])

        d_r = sdf_rigid[y, x]
        t_r = thresh_rigid[y, x]
        if d_r < t_r:
            pen = t_r - d_r
            max_pen_this_frame = max(max_pen_this_frame, pen) # Track deepest cut
            mag = np.hypot(gy_r[y,x], gx_r[y,x]) + 1e-5
            repel += np.array([gy_r[y,x], gx_r[y,x]]) / mag * pen

        d_s = sdf_soft[y, x]
        t_s = thresh_soft[y, x]
        if d_s < t_s:
            pen = t_s - d_s
            max_pen_this_frame = max(max_pen_this_frame, pen) # Track deepest cut
            mag = np.hypot(gy_s[y,x], gx_s[y,x]) + 1e-5
            repel += np.array([gy_s[y,x], gx_s[y,x]]) / mag * pen

        # Apply Net Force
        P_new[i] += W_SPRING * tension + W_REPEL * repel
        
    P = P_new

    # 3. SAFETY CHECK: THE "UNDO" MECHANIC
    if popping_active and max_pen_this_frame > MAX_PENETRATION:
        print(f"\n[FRAME {frame}] OVER-TENSIONED! Penetration: {max_pen_this_frame:.2f}px")
        print("Restoring to previous safe state and entering RELAXATION mode...")
        P = last_safe_P.copy() # UNDO the last pop!
        popping_active = False # Stop popping, let the spring settle into the optimal path

    # --- RENDER VIDEO FRAME ---
    vis = base_img.copy()
    
    cv_path = np.array([(int(c), int(r)) for r, c in P], np.int32)
    cv2.polylines(vis, [cv_path], isClosed=False, color=(0, 255, 255), thickness=3)
    
    # Draw rope nodes (masses) 
    for pt in P:
        cv2.circle(vis, (int(pt[1]), int(pt[0])), 2, (0, 165, 255), -1)
        
    cv2.circle(vis, (int(P[0,1]), int(P[0,0])), 8, (0, 255, 0), -1)
    cv2.circle(vis, (int(P[-1,1]), int(P[-1,0])), 8, (0, 0, 255), -1)

    status_text = "TENSIONING (Distributed Popping)" if popping_active else "RELAXING (Settling to optimal)"
    status_color = (0, 255, 255) if popping_active else (0, 255, 0)
    
    cv2.putText(vis, f"Frame: {frame}/{ITERATIONS} | Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(vis, f"Total Nodes Popped: {pops_done}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Max Penetration: {max_pen_this_frame:.2f}px", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    if frame == ITERATIONS - 1:
        cv2.putText(vis, "SIMULATION COMPLETE (Press any key to exit)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Rope Tensioner", vis)
    
    # Wait 10ms between frames (~100 FPS animation). Press 'ESC' to exit early.
    if cv2.waitKey(10) & 0xFF == 27: 
        break

print("Simulation finished. Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()