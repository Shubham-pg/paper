import cv2
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, medial_axis

# --- HYPERPARAMETERS ---
OFFSET = 100.0           
STEP = 0.5
# -----------------------


def get_enclosed_highway(sdf, ridges, gx, gy):
    # Find max internal clearance for the shell
    mag = np.hypot(gx, gy)
    parallel_mask = (ridges > 0) & (mag <= 0.17) 
    max_val = np.max(sdf[parallel_mask]) if np.any(parallel_mask) else 15.0
    
    limit = max_val + OFFSET
    
    # 1. Create the outer bounding shell
    shell_mask = (np.abs(sdf - limit) <= 1.5)
    
    # 2. Prune the medial axis so it doesn't extend beyond the limit
    pruned_ridges = (ridges > 0) & (sdf <= limit)
    
    # 3. Combine the pruned ridges with the shell
    highway_mask = (pruned_ridges | shell_mask).astype(np.uint8)
    
    # --- CONTINUITY FIX ---
    kernel = np.ones((3,3), np.uint8)
    highway_mask = cv2.morphologyEx(highway_mask, cv2.MORPH_CLOSE, kernel)
    
    return highway_mask, limit

def reach_highway(start, sdf, gx, gy, highway):
    """
    Simultaneous Gradient (+1) and Anti-gradient (-1) stepping.
    """
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
            
            # THE FIX: 3x3 Neighborhood Collision Check
            y_min, y_max = max(0, y-1), min(highway.shape[0], y+2)
            x_min, x_max = max(0, x-1), min(highway.shape[1], x+2)
            
            patch = highway[y_min:y_max, x_min:x_max]
            
            if np.any(patch > 0):
                # We hit the skeleton! Snap to the exact pixel.
                hit_y, hit_x = np.argwhere(patch > 0)[0]
                return s['path'], (y_min + hit_y, x_min + hit_x)
                
            mag = np.hypot(gx[y,x], gy[y,x])
            if mag < 0.05: 
                s['active'] = False; continue
                
            s['curr'] += [s['dir'] * gy[y,x]/mag * STEP, s['dir'] * gx[y,x]/mag * STEP]
            
    return [], None

# --- MAIN WORKFLOW ---
# Make sure "map_data.npz" is the correct file path
data = np.load("map_data.npz")
mask, start_pt, goal_pt = data['canvas'], data['start'], data['goal']

# 1. Metric Fields
sdf = distance_transform_edt(mask == 0).astype(np.float32)
gy, gx = np.gradient(sdf)

# 2. Topological Ridge Extraction (Replaces Hessian Logic)
raw_ridges = medial_axis(mask == 0).astype(np.uint8)

# 3. Construct Highway (Prunes dead ends based on your gradient limits)
highway_mask, limit = get_enclosed_highway(sdf, raw_ridges, gx, gy)

# 4. Skeletonize to get a 1-pixel thin graph 
skeleton = skeletonize(highway_mask > 0).astype(np.uint8)

# 5. Inject Scouts
path_s, node_s = reach_highway(start_pt, sdf, gx, gy, skeleton)
path_g, node_g = reach_highway(goal_pt, sdf, gx, gy, skeleton)

# 6. BUILD TOPOLOGICAL GRAPH
G = nx.Graph()
rows, cols = np.where(skeleton > 0)
pts = set(zip(rows, cols))

# Identify Critical Nodes (Junctions, Endpoints, and Scout Entry Points)
critical_nodes = set()
if node_s: critical_nodes.add(node_s)
if node_g: critical_nodes.add(node_g)

for r, c in pts:
    neighbors = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            if (r+dr, c+dc) in pts: neighbors += 1
    if neighbors != 2:
        critical_nodes.add((r, c))

# Trace edges between critical nodes
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

# Clean remaining noise to ensure a fully connected graph
for component in list(nx.connected_components(G)):
    if len(component) < 2:
        G.remove_nodes_from(component)

# --- VISUALIZATION ---
vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Draw ALL Medial Axes
# Dilate the 1-pixel skeleton to make it 3 pixels thick for visualization
thick_skeleton = cv2.dilate(skeleton, np.ones((2, 2), np.uint8), iterations=1)
vis[thick_skeleton > 0] = [180, 0, 180]

# Draw ALL Graph Nodes (Reverted to 3px)
for n in G.nodes():
    cv2.circle(vis, (n[1], n[0]), 3, (255, 0, 255), -1) # Magenta dots

if node_s in G and node_g in G:
    try:
        # Dijkstra
        node_path = nx.shortest_path(G, node_s, node_g, weight='weight')
        
        # Highlight Dijkstra Path Nodes (Reverted to 6px and 4px)
        for n in node_path:
            cv2.circle(vis, (n[1], n[0]), 6, (255, 255, 255), 1) # White border
            cv2.circle(vis, (n[1], n[0]), 4, (0, 165, 255), -1)  # Orange fill

        # Assemble full pixel path
        hwy_pixel_path = []
        for i in range(len(node_path)-1):
            edge_data = G.get_edge_data(node_path[i], node_path[i+1])
            pixels = edge_data['pixels']
            if pixels[0] != node_path[i]: pixels = pixels[::-1]
            hwy_pixel_path.extend(pixels)

        full_path = path_s + hwy_pixel_path + path_g[::-1]
        
        # Draw final Yellow Path (Reverted line thickness to 2)
        for i in range(len(full_path)-1):
            cv2.line(vis, (int(full_path[i][1]), int(full_path[i][0])), 
                     (int(full_path[i+1][1]), int(full_path[i+1][0])), (0, 255, 255), 2)
            
        print(f"Topological Routing Success! Total Graph Nodes: {len(G.nodes())} | Path Nodes: {len(node_path)}")
    except nx.NetworkXNoPath:
        msg = "Disconnected graph components. Path routing failed."
        print(msg)
        # Added text overlay for graph failure
        cv2.putText(vis, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
else:
    msg = "One or both scouts failed to reach the skeleton."
    print(msg)
    # Added text overlay for scout failure
    cv2.putText(vis, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Draw Start and Goal Points (Reverted to 6px)
cv2.circle(vis, (start_pt[1], start_pt[0]), 6, (0, 255, 0), -1)
cv2.circle(vis, (goal_pt[1], goal_pt[0]), 6, (0, 0, 255), -1)

output_filename = "dijkstra_graph_result3.png"
cv2.imwrite(output_filename, vis)
print(f"--- SUCCESS: Visualization saved as {output_filename} ---")

# Kept WINDOW_NORMAL so you can resize the window easily
cv2.namedWindow("Medial Axis Dijkstra Graph", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Medial Axis Dijkstra Graph", vis.shape[1], vis.shape[0])
cv2.imshow("Medial Axis Dijkstra Graph", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()