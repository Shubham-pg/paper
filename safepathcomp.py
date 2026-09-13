import cv2
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, medial_axis
from itertools import islice

# --- HYPERPARAMETERS ---
OFFSET = 100.0           
STEP = 0.5
CLEAR_RIGID = 30.0    # Visual band distance for Rigid Walls (White)
CLEAR_SOFT = 15.0     # Visual band distance for Tight Clearance Walls (Gray)
K_PATHS = 5           # Number of distinct shortest paths to evaluate 
MAX_OVERLAP = 0.85    # Reject paths that share more than 85% of their distance with another
TRIM_MARGIN = 2.0     # Shrink the band so it rests exactly 2 pixels behind the path's tightest squeeze
# -----------------------

def get_enclosed_highway(sdf, ridges, gx, gy):
    mag = np.hypot(gx, gy)
    parallel_mask = (ridges > 0) & (mag <= 0.17) 
    max_val = np.max(sdf[parallel_mask]) if np.any(parallel_mask) else 15.0
    limit = max_val + OFFSET
    shell_mask = (np.abs(sdf - limit) <= 1.5)
    pruned_ridges = (ridges > 0) & (sdf <= limit)
    highway_mask = (pruned_ridges | shell_mask).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    highway_mask = cv2.morphologyEx(highway_mask, cv2.MORPH_CLOSE, kernel)
    return highway_mask, limit

def reach_highway(start, sdf, gx, gy, highway):
    scouts = [
        {'curr': np.array(start, dtype=float), 'dir': 1,  'active': True, 'path': []},
        {'curr': np.array(start, dtype=float), 'dir': -1, 'active': True, 'path': []}
    ]
    for _ in range(2500):
        for s in scouts:
            if not s['active']: continue
            y, x = int(s['curr'][0]), int(s['curr'][1])
            if not (0 <= y < sdf.shape[0] and 0 <= x < sdf.shape[1]):
                s['active'] = False; continue

            s['path'].append((y, x))
            y_min, y_max = max(0, y-1), min(highway.shape[0], y+2)
            x_min, x_max = max(0, x-1), min(highway.shape[1], x+2)
            patch = highway[y_min:y_max, x_min:x_max]

            if np.any(patch > 0):
                hit_y, hit_x = np.argwhere(patch > 0)[0]
                return s['path'], (y_min + hit_y, x_min + hit_x)

            mag = np.hypot(gx[y,x], gy[y,x])
            if mag < 0.05: 
                s['active'] = False; continue
            s['curr'] += [s['dir'] * gy[y,x]/mag * STEP, s['dir'] * gx[y,x]/mag * STEP]
    return [], None

# --- MAIN WORKFLOW ---
try:
    data = np.load("map_data.npz")
    mask, start_pt, goal_pt = data['canvas'], data['start'], data['goal']
except FileNotFoundError:
    print("Error: 'map_data.npz' not found. Please run the map editor first.")
    exit()

# Metric Fields
sdf_raw = distance_transform_edt(mask == 0).astype(np.float32)
gy, gx = np.gradient(sdf_raw)
raw_ridges = medial_axis(mask == 0).astype(np.uint8)

highway_mask, limit = get_enclosed_highway(sdf_raw, raw_ridges, gx, gy)
skeleton = skeletonize(highway_mask > 0).astype(np.uint8)

path_s, node_s = reach_highway(start_pt, sdf_raw, gx, gy, skeleton)
path_g, node_g = reach_highway(goal_pt, sdf_raw, gx, gy, skeleton)

# BUILD TOPOLOGICAL GRAPH
G = nx.Graph()
rows, cols = np.where(skeleton > 0)
pts = set(zip(rows, cols))

critical_nodes = set()
if node_s: critical_nodes.add(node_s)
if node_g: critical_nodes.add(node_g)

for r, c in pts:
    neighbors = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            if (r+dr, c+dc) in pts: neighbors += 1
    if neighbors != 2: critical_nodes.add((r, c))

visited = set()
for node in critical_nodes:
    r, c = node
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        neighbor = (r+dr, c+dc)
        if neighbor in pts and (node, neighbor) not in visited:
            segment = [node, neighbor]
            curr = neighbor
            while curr not in critical_nodes:
                found_next = False
                for ddr, ddc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nxt = (curr[0]+ddr, curr[1]+ddc)
                    if nxt in pts and nxt != segment[-2]:
                        segment.append(nxt)
                        curr = nxt
                        found_next = True
                        break
                if not found_next: break 
            dist = sum(np.hypot(segment[i][0]-segment[i+1][0], segment[i][1]-segment[i+1][1]) for i in range(len(segment)-1))
            G.add_edge(node, curr, weight=dist, pixels=segment)
            visited.add((node, neighbor))
            visited.add((curr, segment[-2] if len(segment)>1 else node))

for component in list(nx.connected_components(G)):
    if len(component) < 2: G.remove_nodes_from(component)

# --- NEW: COMPONENT-WISE DISTANCE FIELDS ---
# We calculate distances, but also store WHICH obstacle is closest to each pixel
sdf_rigid, idx_rigid = distance_transform_edt(mask != 255, return_indices=True)
sdf_soft, idx_soft = distance_transform_edt(mask != 128, return_indices=True)

# Assign a unique integer label to every disconnected wall
_, labels_rigid = cv2.connectedComponents((mask == 255).astype(np.uint8))
_, labels_soft = cv2.connectedComponents((mask == 128).astype(np.uint8))

# Map every pixel on the screen to the ID of the wall it is closest to
nearest_rigid_lbl = labels_rigid[idx_rigid[0], idx_rigid[1]]
nearest_soft_lbl = labels_soft[idx_soft[0], idx_soft[1]]


def draw_base_map(dynamic_t_rigid=None, dynamic_t_soft=None):
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    vis[mask == 0] = [30, 30, 30]
    
    t_soft = CLEAR_SOFT if dynamic_t_soft is None else dynamic_t_soft
    t_rigid = CLEAR_RIGID if dynamic_t_rigid is None else dynamic_t_rigid

    vis[(mask == 0) & (sdf_soft < t_soft)] = [80, 50, 30] 
    vis[(mask == 0) & (sdf_rigid < t_rigid)] = [30, 30, 100] 

    vis[mask == 128] = [100, 100, 100]  
    vis[mask == 255] = [200, 200, 200]  
    
    thick_skeleton = cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)
    vis[thick_skeleton > 0] = [180, 0, 180]
    for n in G.nodes():
        cv2.circle(vis, (n[1], n[0]), 3, (255, 0, 255), -1)
    
    cv2.circle(vis, (start_pt[1], start_pt[0]), 6, (0, 255, 0), -1)
    cv2.circle(vis, (goal_pt[1], goal_pt[0]), 6, (0, 0, 255), -1)
    return vis


if node_s in G and node_g in G:
    try:
        print(f"Finding top {K_PATHS} macroscopically distinct paths...")
        
        path_generator = nx.shortest_simple_paths(G, node_s, node_g, weight='weight')
        k_shortest_nodes = []
        accepted_paths_data = [] 
        
        # --- PHASE 0: Macroscopic Path Filter ---
        for node_path in path_generator:
            path_weight = sum(G[node_path[i]][node_path[i+1]]['weight'] for i in range(len(node_path)-1))
            edges_curr = {(min(node_path[i], node_path[i+1]), max(node_path[i], node_path[i+1])): 
                          G[node_path[i]][node_path[i+1]]['weight'] for i in range(len(node_path)-1)}
            
            is_unique = True
            for acc_path, acc_weight in accepted_paths_data:
                edges_acc = {(min(acc_path[i], acc_path[i+1]), max(acc_path[i], acc_path[i+1])) for i in range(len(acc_path)-1)}
                shared_length = sum(w for e, w in edges_curr.items() if e in edges_acc)
                if shared_length > MAX_OVERLAP * min(path_weight, acc_weight):
                    is_unique = False
                    break
                    
            if is_unique:
                k_shortest_nodes.append(node_path)
                accepted_paths_data.append((node_path, path_weight))
                print(f"  > Found distinct path {len(k_shortest_nodes)} (Length: {path_weight:.1f}px)")
                
            if len(k_shortest_nodes) >= K_PATHS:
                break
                
        paths_info = []
        for i, node_path in enumerate(k_shortest_nodes):
            hwy_pixel_path = []
            for j in range(len(node_path)-1):
                edge_data = G.get_edge_data(node_path[j], node_path[j+1])
                pixels = edge_data['pixels']
                if pixels[0] != node_path[j]: pixels = pixels[::-1]
                hwy_pixel_path.extend(pixels)

            full_path = path_s + hwy_pixel_path + path_g[::-1]
            
            sdf_vals = [sdf_rigid[int(r), int(c)] for r, c in full_path]
            lex_score = sorted([round(v, 1) for v in sdf_vals]) 
            has_overlap = any(sdf_rigid[int(r), int(c)] < CLEAR_RIGID or 
                              sdf_soft[int(r), int(c)] < CLEAR_SOFT for r, c in full_path)
            
            paths_info.append({
                'id': i + 1,
                'pixels': full_path,
                'lex_score': lex_score,
                'has_overlap': has_overlap
            })

        # --- PHASE 1: VISUALIZE EVALUATED PATHS ---
        for p in paths_info:
            vis = draw_base_map()
            cv_path = np.array([(int(c), int(r)) for r, c in p['pixels']], np.int32)
            cv2.polylines(vis, [cv_path], isClosed=False, color=(0, 255, 255), thickness=2)
            
            worst_choke = p['lex_score'][0] if p['lex_score'] else "N/A"
            cv2.putText(vis, f"Evaluating Path {p['id']}/{len(paths_info)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, f"Overlap: {p['has_overlap']} | Worst Squeeze: {worst_choke}px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, "Press ANY KEY for next path...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            cv2.namedWindow("Path Evaluation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Path Evaluation", 1200, 800)
            cv2.imshow("Path Evaluation", vis)
            cv2.waitKey(0)
        cv2.destroyWindow("Path Evaluation")
        
        # --- PHASE 2: CHOOSE SAFEST PATH ---
        safe_paths = [p for p in paths_info if not p['has_overlap']]
        if safe_paths:
            best_path = safe_paths[0] 
            print(f"Path {best_path['id']} is perfectly safe!")
        else:
            best_path = max(paths_info, key=lambda p: p['lex_score'])
            print(f"All paths intersect bands. Chose Path {best_path['id']} via Decimal Tie-Breaker.")

        # --- PHASE 3: SELECTIVE OBSTACLE-SPECIFIC TRIMMING ---
        rigid_reqs = {}
        soft_reqs = {}

        # Scan the path to find the absolute tightest squeeze FOR EACH SPECIFIC WALL ID
        for r, c in best_path['pixels']:
            r, c = int(r), int(c)
            
            r_lbl = nearest_rigid_lbl[r, c]
            d_rigid = sdf_rigid[r, c]
            if r_lbl > 0 and d_rigid < CLEAR_RIGID:
                if r_lbl not in rigid_reqs or d_rigid < rigid_reqs[r_lbl]:
                    rigid_reqs[r_lbl] = d_rigid
                    
            s_lbl = nearest_soft_lbl[r, c]
            d_soft = sdf_soft[r, c]
            if s_lbl > 0 and d_soft < CLEAR_SOFT:
                if s_lbl not in soft_reqs or d_soft < soft_reqs[s_lbl]:
                    soft_reqs[s_lbl] = d_soft

        # Generate new threshold maps. Default everything to standard clearance.
        dynamic_thresh_rigid = np.full(mask.shape, CLEAR_RIGID, dtype=np.float32)
        dynamic_thresh_soft = np.full(mask.shape, CLEAR_SOFT, dtype=np.float32)

        # Overwrite the threshold ONLY for the specific walls we intersected
        for lbl, min_dist in rigid_reqs.items():
            dynamic_thresh_rigid[nearest_rigid_lbl == lbl] = max(0.0, min_dist - TRIM_MARGIN)
            
        for lbl, min_dist in soft_reqs.items():
            dynamic_thresh_soft[nearest_soft_lbl == lbl] = max(0.0, min_dist - TRIM_MARGIN)

        # Draw the final map using our mathematically trimmed bands
        final_vis = draw_base_map(dynamic_t_rigid=dynamic_thresh_rigid, dynamic_t_soft=dynamic_thresh_soft)
        
        final_cv_path = np.array([(int(c), int(r)) for r, c in best_path['pixels']], np.int32)
        cv2.polylines(final_vis, [final_cv_path], isClosed=False, color=(0, 255, 0), thickness=3)

        cv2.putText(final_vis, f"WINNER: Path {best_path['id']} (Safest Optimum)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(final_vis, "Notice how ONLY intersected walls were selectively trimmed!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.namedWindow("Final Trimmed Path", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Final Trimmed Path", 1200, 800)
        cv2.imshow("Final Trimmed Path", final_vis)
        
        # --- PHASE 4: SAVE OUTPUT DATA FOR ROPE TENSIONER ---
        out_file = "planner_data.npz"
        np.savez(out_file, 
                 path=np.array(best_path['pixels']),
                 thresh_rigid=dynamic_thresh_rigid,
                 thresh_soft=dynamic_thresh_soft,
                 sdf_rigid=sdf_rigid,
                 sdf_soft=sdf_soft,
                 canvas=mask)
        print(f"--- SUCCESS: Planner data saved to {out_file} ---")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except nx.NetworkXNoPath:
        vis = draw_base_map()
        cv2.putText(vis, "Disconnected graph components. Path routing failed.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.namedWindow("Error", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Error", 1200, 800)
        cv2.imshow("Error", vis)
        cv2.waitKey(0)
else:
    vis = draw_base_map()
    cv2.putText(vis, "One or both scouts failed to reach the skeleton.", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.namedWindow("Error", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Error", 1200, 800)
    cv2.imshow("Error", vis)
    cv2.waitKey(0)