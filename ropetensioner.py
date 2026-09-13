import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# --- HYPERPARAMETERS: SPRING-MASS SYSTEM ---
INITIAL_NODE_SPACING = 8.0  
NODE_REST_DIST = 5.0        
W_SPRING = 0.55             
W_REPEL = 2.5               
DAMPING = 0.4               # NEW: Friction to absorb bouncy kinetic energy (jitter)

# --- HYPERPARAMETERS: DISTRIBUTED POPPING & SAFETY ---
POP_INTERVAL = 20           
POP_STRIDE = 10             
MAX_PENETRATION = 1.5       

# --- HYPERPARAMETERS: AUTO-STABILITY ---
STABILITY_THRESHOLD = 0.7  # Increased slightly to ensure smooth auto-stopping
STABLE_FRAMES_REQ = 30      
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

# 3. PROPORTIONAL NODE INITIALIZATION
dists = np.zeros(len(path_pixels))
dists[1:] = np.hypot(np.diff(path_pixels[:, 0]), np.diff(path_pixels[:, 1]))
cum_dists = np.cumsum(dists)
total_len = cum_dists[-1]

num_nodes = max(3, int(total_len / INITIAL_NODE_SPACING))
target_dists = np.linspace(0, total_len, num_nodes)
P_y = np.interp(target_dists, cum_dists, path_pixels[:, 0])
P_x = np.interp(target_dists, cum_dists, path_pixels[:, 1])
P = np.column_stack((P_y, P_x)).astype(np.float32)

# 4. Generate the Static Base Map (Pre-rendered)
base_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
base_img[mask == 0] = [30, 30, 30]

base_img[(mask == 0) & (sdf_soft < thresh_soft)] = [80, 50, 30] 
base_img[(mask == 0) & (sdf_rigid < thresh_rigid)] = [30, 30, 100] 
base_img[mask == 128] = [100, 100, 100]  
base_img[mask == 255] = [200, 200, 200]  

cv2.namedWindow("Real-Time Rope Tensioner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Rope Tensioner", 1200, 800)
print("Starting Auto-Stabilizing Spring-Mass Simulation with Damping...")

pops_done = 0
popping_active = True
last_safe_P = P.copy()
stable_frame_count = 0
frame = 0

# NEW: Track velocity for momentum and friction
V = np.zeros_like(P)

# --- PHYSICS LOOP (INFINITE UNTIL AUTO-STOP) ---
while True:
    
    # 1. DISTRIBUTED POPPING MECHANIC
    if popping_active and frame % POP_INTERVAL == 0 and frame > 0:
        last_safe_P = P.copy()
        
        if len(P) > POP_STRIDE * 2:
            indices_to_delete = np.arange(POP_STRIDE, len(P) - 1, POP_STRIDE)
            P = np.delete(P, indices_to_delete, axis=0)
            V = np.zeros_like(P) # Reset velocities so rope doesn't explode when changed
            pops_done += len(indices_to_delete)
        else:
            popping_active = False

    P_new = P.copy()
    max_pen_this_frame = 0.0
    max_movement = 0.0

    # 2. CALCULATE PHYSICS FORCES
    for i in range(1, len(P)-1):
        y, x = int(P[i, 0]), int(P[i, 1])
        y = np.clip(y, 0, mask.shape[0]-1)
        x = np.clip(x, 0, mask.shape[1]-1)

        # Force A: Spring Mass Tension
        v1 = P[i-1] - P[i]
        d1 = np.hypot(v1[0], v1[1]) + 1e-5
        f1 = (v1 / d1) * (d1 - NODE_REST_DIST)

        v2 = P[i+1] - P[i]
        d2 = np.hypot(v2[0], v2[1]) + 1e-5
        f2 = (v2 / d2) * (d2 - NODE_REST_DIST)

        tension = f1 + f2

        # Force B: External Repulsion
        repel = np.array([0.0, 0.0])

        d_r = sdf_rigid[y, x]
        t_r = thresh_rigid[y, x]
        if d_r < t_r:
            pen = t_r - d_r
            max_pen_this_frame = max(max_pen_this_frame, pen)
            mag = np.hypot(gy_r[y,x], gx_r[y,x]) + 1e-5
            repel += np.array([gy_r[y,x], gx_r[y,x]]) / mag * pen

        d_s = sdf_soft[y, x]
        t_s = thresh_soft[y, x]
        if d_s < t_s:
            pen = t_s - d_s
            max_pen_this_frame = max(max_pen_this_frame, pen)
            mag = np.hypot(gy_s[y,x], gx_s[y,x]) + 1e-5
            repel += np.array([gy_s[y,x], gx_s[y,x]]) / mag * pen

        # --- TRUE PHYSICS WITH FRICTION ---
        net_force = W_SPRING * tension + W_REPEL * repel
        
        # Apply force to Velocity, then multiply by Friction (Damping)
        V[i] = (V[i] + net_force) * DAMPING
        
        # Apply final Velocity to Position
        P_new[i] += V[i]
        
        # Track maximum true movement for Auto-Stop logic
        move_dist = np.hypot(V[i][0], V[i][1])
        max_movement = max(max_movement, move_dist)

    P = P_new

    # 3. SAFETY CHECK: THE "UNDO" MECHANIC
    if popping_active and max_pen_this_frame > MAX_PENETRATION:
        print(f"\n[FRAME {frame}] OVER-TENSIONED! Penetration: {max_pen_this_frame:.2f}px")
        print("Restoring to previous safe state and entering RELAXATION mode...")
        P = last_safe_P.copy() 
        V = np.zeros_like(P) # Kill momentum on restore
        popping_active = False 

    # 4. AUTO-STOP LOGIC (Triggers only during Relaxation Mode)
    if not popping_active:
        if max_movement < STABILITY_THRESHOLD:
            stable_frame_count += 1
        else:
            stable_frame_count = 0
            
        if stable_frame_count >= STABLE_FRAMES_REQ:
            print(f"\n[FRAME {frame}] AUTO-STOP TRIGGERED! System is perfectly stable.")
            break

    # --- RENDER VIDEO FRAME ---
    vis = base_img.copy()
    
    cv_path = np.array([(int(c), int(r)) for r, c in P], np.int32)
    cv2.polylines(vis, [cv_path], isClosed=False, color=(0, 255, 255), thickness=3)
    
    for pt in P:
        cv2.circle(vis, (int(pt[1]), int(pt[0])), 2, (0, 165, 255), -1)
        
    cv2.circle(vis, (int(P[0,1]), int(P[0,0])), 8, (0, 255, 0), -1)
    cv2.circle(vis, (int(P[-1,1]), int(P[-1,0])), 8, (0, 0, 255), -1)

    status_text = "TENSIONING (Popping)" if popping_active else "RELAXING (Settling)"
    status_color = (0, 255, 255) if popping_active else (0, 255, 0)
    
    cv2.putText(vis, f"Frame: {frame} | Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(vis, f"Nodes Popped: {pops_done}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Max Movement: {max_movement:.4f}px", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    cv2.imshow("Real-Time Rope Tensioner", vis)
    
    frame += 1
    if cv2.waitKey(10) & 0xFF == 27: 
        break

# --- FINAL FRAME DISPLAY ---
cv2.putText(vis, "AUTO-STOP: System Stable! (Press any key to exit)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Real-Time Rope Tensioner", vis)

print("Simulation finished optimally. Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()