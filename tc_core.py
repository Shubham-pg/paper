
"""
tc_core.py -- Shared library for Topological Cage experiments.

Everything the benchmark scripts need lives here:
  * synthetic environment generation (E1..E4) + loader for your hand-drawn map
  * distance field, medial ridge, tangent-gradient pruning, super-critical closure
  * graph abstraction, gradient injection, K-path enumeration, lexicographic selection
  * baselines: A*, PRM, RRT*, GVD+A*-injection, EGVG-style raycast injection
  * damped spring-mass band (with ablatable terms)
  * metrics: clearance, curvature, jerk, spacing CV

Nothing here prints. The exp*.py scripts do the printing.
"""

import time
import heapq
import math
import numpy as np
import cv2
import networkx as nx
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import medial_axis, skeletonize

# ======================================================================
# GLOBAL PARAMETERS  (these become Table "Parameters used in experiments")
# ======================================================================
PARAMS = dict(
    tau_deg        = 17.0,   # tangent-gradient pruning tolerance
    delta_offset   = 20.0,   # cage offset above C_max, in pixels
    alpha_step     = 0.5,    # gradient integration step, pixels
    K_paths        = 5,      # homotopy candidates enumerated
    max_overlap    = 0.85,   # macroscopic distinctness filter
    d_rest         = 5.0,    # band rest spacing, pixels
    init_spacing   = 8.0,    # initial band node spacing, pixels
    W_s            = 0.55,   # spring weight
    W_r            = 2.5,    # repulsion weight
    D_damp         = 0.4,    # kinetic damping
    h_step         = 1.0,    # integration timestep (absorbed into W_s/W_r)
    eps_stab       = 0.7,    # auto-stop velocity threshold, px/iter
    n_hold         = 30,     # consecutive stable iterations required
    pop_interval   = 20,     # T_pop
    pop_stride     = 10,     # N_pop
    max_penetration= 1.5,
    rho_safe       = 6.0,    # robot radius + safety margin, pixels
    clear_rigid    = 30.0,   # nominal rigid threshold tau_O^nom
    clear_soft     = 15.0,   # nominal soft threshold
    trim_margin    = 2.0,    # eta
    grid_res_m_px  = 0.02,   # metres per pixel (for reporting in metres)
    iter_cap        = 1500,
    max_pop_rounds  = 12,
)

PX2M = PARAMS['grid_res_m_px']


# ======================================================================
# 1. ENVIRONMENTS
# ======================================================================
def _rect(m, y0, x0, y1, x1, v=255):
    m[int(y0):int(y1), int(x0):int(x1)] = v


def make_env(name, H=500, W=750, seed=0):
    """Return uint8 mask: 0 = free, 128 = soft obstacle, 255 = rigid obstacle."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    b = 12  # outer wall thickness
    _rect(m, 0, 0, b, W); _rect(m, H - b, 0, H, W)
    _rect(m, 0, 0, H, b); _rect(m, 0, W - b, H, W)

    if name == 'E1':
        # Structured indoor: rooms + doorways. Dense, well-conditioned.
        for x in (200, 400, 560):
            _rect(m, b, x, H - b, x + 16)
        _rect(m, 150, 200, 230, 216, 0)          # doorway
        _rect(m, 300, 400, 360, 416, 0)          # doorway
        _rect(m, 100, 560, 150, 576, 0)          # narrow doorway
        _rect(m, 250, b, 266, 200)               # horizontal spur
        _rect(m, 120, 60, 180, 140, 128)         # soft furniture
        _rect(m, 340, 440, 420, 520, 128)

    elif name == 'E2':
        # Semi-sparse with one huge open region -> Voronoi-void stress case.
        _rect(m, b, 120, 300, 136)
        _rect(m, 380, 120, H - b, 136)
        _rect(m, 200, 600, 320, 616)
        _rect(m, 60, 300, 90, 380, 128)          # one small island in the void

    elif name == 'E3':
        # Free-standing islands, no continuous interior walls.
        for (cy, cx, r) in [(130, 180, 38), (330, 250, 45), (180, 430, 40),
                            (370, 560, 42), (110, 600, 35)]:
            cv2.circle(m, (cx, cy), r, 255, -1)
        cv2.circle(m, (260, 380), 30, 128, -1)

    elif name == 'E4':
        # U-shaped non-convex trap with a narrow exit.
        _rect(m, 120, 220, 140, 540)             # top bar
        _rect(m, 120, 220, 400, 240)             # left arm
        _rect(m, 120, 520, 400, 540)             # right arm
        _rect(m, 380, 220, 400, 372)             # bottom partial -> narrow exit
        _rect(m, 380, 394, 400, 540)             # 22 px gap == the only way out
        _rect(m, 200, 300, 260, 360, 128)        # soft blob inside the trap

    elif name == 'E5_scale':
        # Same topology as E1 but larger, for the area-scaling argument.
        return make_env('E1', H * 2, W * 2, seed)

    else:
        raise ValueError(f'unknown env {name}')
    return m


def load_user_map(path='map_data.npz'):
    """Load your hand-drawn map; returns (mask, start, goal)."""
    d = np.load(path)
    return d['canvas'], tuple(d['start']), tuple(d['goal'])


# ======================================================================
# 2. FIELDS
# ======================================================================
class Fields:
    """Distance fields and gradients for a mask."""

    def __init__(self, mask):
        self.mask = mask
        self.free = (mask == 0)
        self.sdf = distance_transform_edt(self.free).astype(np.float32)
        self.gy, self.gx = np.gradient(self.sdf)
        self.mag = np.hypot(self.gy, self.gx)

        # per-class fields (rigid / soft) + component labels for selective relaxation
        self.sdf_rigid, idx_r = distance_transform_edt(mask != 255, return_indices=True)
        self.sdf_soft,  idx_s = distance_transform_edt(mask != 128, return_indices=True)
        self.sdf_rigid = self.sdf_rigid.astype(np.float32)
        self.sdf_soft = self.sdf_soft.astype(np.float32)
        _, lab_r = cv2.connectedComponents((mask == 255).astype(np.uint8))
        _, lab_s = cv2.connectedComponents((mask == 128).astype(np.uint8))
        self.near_rigid = lab_r[idx_r[0], idx_r[1]]
        self.near_soft = lab_s[idx_s[0], idx_s[1]]
        self.gy_r, self.gx_r = np.gradient(self.sdf_rigid)
        self.gy_s, self.gx_s = np.gradient(self.sdf_soft)

        self.ridges = medial_axis(self.free).astype(np.uint8)

    def clear(self, p):
        y, x = int(p[0]), int(p[1])
        y = np.clip(y, 0, self.sdf.shape[0] - 1)
        x = np.clip(x, 0, self.sdf.shape[1] - 1)
        return float(self.sdf[y, x])


# ======================================================================
# 3. CAGE CONSTRUCTION
# ======================================================================
def ridge_tangents(ridge_mask, win=5):
    """Unit tangent at every ridge pixel by local PCA over a (2*win+1) window."""
    idx = np.argwhere(ridge_mask > 0)
    if len(idx) == 0:
        return idx, np.zeros((0, 2), np.float32)
    H, W = ridge_mask.shape
    occ = np.zeros((H, W), bool); occ[idx[:, 0], idx[:, 1]] = True
    T = np.zeros((len(idx), 2), np.float32)
    for k, (r, c) in enumerate(idx):
        y0, y1 = max(0, r - win), min(H, r + win + 1)
        x0, x1 = max(0, c - win), min(W, c + win + 1)
        loc = np.argwhere(occ[y0:y1, x0:x1]).astype(np.float32)
        if len(loc) < 2:
            T[k] = (1.0, 0.0); continue
        loc -= loc.mean(0)
        try:
            _, _, vt = np.linalg.svd(loc, full_matrices=False)
            T[k] = vt[0]
        except np.linalg.LinAlgError:
            T[k] = (1.0, 0.0)
    return idx, T


def prune_ridges(F, tau_deg, win=5, probe=3.0):
    """Tangent-gradient parallelism test, Eq. (prune).

    theta(p) is the deviation from orthogonality between the ridge tangent
    t_M(p) and the distance gradient grad Psi(p). Because grad Psi is
    discontinuous exactly on the ridge, we evaluate the projection
    |t_M . grad Psi| by a centred finite difference of Psi taken ALONG the
    tangent, which is numerically identical to the directional derivative but
    stable on the ridge itself:

        |t_M . grad Psi| ~= |Psi(p + w t_M) - Psi(p - w t_M)| / (2w)

    Interior corridor  -> Psi is ~constant along the ridge -> value ~0 -> kept.
    Exterior branch    -> Psi climbs along the ridge       -> value ~1 -> cut.
    """
    thr = math.cos(math.radians(90.0 - tau_deg))          # = sin(tau)
    idx, T = ridge_tangents(F.ridges > 0, win)
    out = np.zeros_like(F.ridges, bool)
    if len(idx) == 0:
        return out
    H, W = F.sdf.shape
    yp = np.clip(idx[:, 0] + T[:, 0] * probe, 0, H - 1).astype(int)
    xp = np.clip(idx[:, 1] + T[:, 1] * probe, 0, W - 1).astype(int)
    ym = np.clip(idx[:, 0] - T[:, 0] * probe, 0, H - 1).astype(int)
    xm = np.clip(idx[:, 1] - T[:, 1] * probe, 0, W - 1).astype(int)
    proj = np.abs(F.sdf[yp, xp] - F.sdf[ym, xm]) / (2.0 * probe)

    # A ridge point sitting exactly on an open-space CLEARANCE PEAK also has a
    # vanishing directional derivative, so the raw pointwise test would keep it.
    # We therefore take the neighbourhood maximum of the projection along the
    # ridge: a peak that lies within `spread` pixels of a climbing exterior
    # branch inherits that branch's violation and is removed with it. This is
    # the branch-level reading of Eq. (prune) and is what the prototype uses.
    projimg = np.zeros_like(F.sdf)
    projimg[idx[:, 0], idx[:, 1]] = proj
    ridgemask = np.zeros_like(F.sdf, bool)
    ridgemask[idx[:, 0], idx[:, 1]] = True
    spread = 2 * win + 1
    projmax = maximum_filter(np.where(ridgemask, projimg, -1.0), size=spread)
    proj_eff = projmax[idx[:, 0], idx[:, 1]]

    keep = proj_eff <= thr
    out[idx[keep, 0], idx[keep, 1]] = True
    return out


def ridge_angles(F, win=5, probe=3.0):
    """Return (idx, theta_deg) for every ridge pixel -- used by figures/ablations."""
    idx, T = ridge_tangents(F.ridges > 0, win)
    if len(idx) == 0:
        return idx, np.zeros(0)
    H, W = F.sdf.shape
    yp = np.clip(idx[:, 0] + T[:, 0] * probe, 0, H - 1).astype(int)
    xp = np.clip(idx[:, 1] + T[:, 1] * probe, 0, W - 1).astype(int)
    ym = np.clip(idx[:, 0] - T[:, 0] * probe, 0, H - 1).astype(int)
    xm = np.clip(idx[:, 1] - T[:, 1] * probe, 0, W - 1).astype(int)
    proj = np.clip(np.abs(F.sdf[yp, xp] - F.sdf[ym, xm]) / (2.0 * probe), 0, 1)
    return idx, np.degrees(np.arcsin(proj))


def build_cage(F, tau_deg=None, delta=None, use_closure=True, quantile=0.75):
    """Build the Topological Cage. Returns dict with timing and masks."""
    tau_deg = PARAMS['tau_deg'] if tau_deg is None else tau_deg
    delta = PARAMS['delta_offset'] if delta is None else delta

    t0 = time.perf_counter()
    interior = prune_ridges(F, tau_deg)
    vals = F.sdf[interior] if interior.any() else np.array([15.0], np.float32)

    # NOTE (reported honestly in the paper): taking the strict maximum of Psi
    # over the interior ridge makes d_cage exceed max(Psi) over the whole map,
    # so the level set {Psi = d_cage} is EMPTY and no shell is produced. The
    # widest interior "corridor" is in practice the open-room plateau itself.
    # The prototype therefore uses a high quantile of the interior clearance
    # distribution as C_max, and clamps d_cage so the contour is non-empty.
    c_max = float(np.quantile(vals, quantile))
    d_cage = c_max + delta
    hi = float(np.quantile(F.sdf[F.free], 0.995))
    d_cage = float(np.clip(d_cage, 2.0 * PARAMS['rho_safe'], hi))

    pruned = (F.ridges > 0) & (F.sdf <= d_cage)
    if use_closure:
        shell = (np.abs(F.sdf - d_cage) <= 1.5)
        cage = (pruned | shell).astype(np.uint8)
    else:
        shell = np.zeros_like(pruned)
        cage = pruned.astype(np.uint8)

    cage = cv2.morphologyEx(cage, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    skel = skeletonize(cage > 0).astype(np.uint8)
    t_build = (time.perf_counter() - t0) * 1e3

    return dict(skel=skel, shell=shell, pruned=pruned, interior=interior,
                c_max=c_max, d_cage=d_cage, t_ms=t_build)


def n_components(skel):
    n, _ = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
    return n - 1  # subtract background


def cage_distance(skel):
    """Distance transform of the cage skeleton, for use in the band energy."""
    return distance_transform_edt(skel == 0).astype(np.float32)

# ======================================================================
# 4. GRAPH ABSTRACTION
# ======================================================================
_N8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def build_graph(skel, F, extra_nodes=()):
    t0 = time.perf_counter()
    G = nx.Graph()
    rows, cols = np.where(skel > 0)
    pts = set(zip(rows.tolist(), cols.tolist()))
    if not pts:
        return G, 0.0

    crit = set(extra_nodes)
    for (r, c) in pts:
        deg = sum(1 for dr, dc in _N8 if (r + dr, c + dc) in pts)
        if deg != 2:
            crit.add((r, c))
    if not crit:                       # pure loop, seed one node
        crit.add(next(iter(pts)))

    seen = set()
    for node in crit:
        r, c = node
        for dr, dc in _N8:
            nb = (r + dr, c + dc)
            if nb not in pts or (node, nb) in seen:
                continue
            seg = [node, nb]
            cur = nb
            guard = 0
            while cur not in crit and guard < 200000:
                guard += 1
                nxt_found = False
                for ddr, ddc in _N8:
                    nxt = (cur[0] + ddr, cur[1] + ddc)
                    if nxt in pts and nxt != seg[-2]:
                        seg.append(nxt); cur = nxt; nxt_found = True; break
                if not nxt_found:
                    break
            w = sum(math.hypot(seg[i][0] - seg[i + 1][0], seg[i][1] - seg[i + 1][1])
                    for i in range(len(seg) - 1))
            if w > 0:
                prof = np.array([F.sdf[p[0], p[1]] for p in seg], np.float32)
                G.add_edge(node, cur, weight=w, pixels=seg, prof=prof)
            seen.add((node, nb))
            if len(seg) > 1:
                seen.add((cur, seg[-2]))

    for comp in list(nx.connected_components(G)):
        if len(comp) < 2:
            G.remove_nodes_from(comp)
    return G, (time.perf_counter() - t0) * 1e3


# ======================================================================
# 5. INJECTION MECHANISMS
# ======================================================================




def inject_gradient(p0, F, skel, alpha=None, max_steps=4000,
                    contact=2.5, dcage=None, max_coast=25):
    """Bidirectional gradient scouts. Returns (path, hit, steps, ok).

    Contact is tested on a radius against a precomputed distance-to-cage field
    rather than on a 3x3 patch. The central-difference gradient magnitude
    collapses within ~1 px of the ridge because the stencil straddles the
    discontinuity; a radius-1 patch test can therefore be outrun by the
    magnitude guard when morphological closing has displaced the cage pixel by
    2 px. Where the discrete magnitude does collapse we coast along the last
    valid direction for a bounded number of steps instead of terminating.
    """
    alpha = PARAMS['alpha_step'] if alpha is None else alpha
    H, W = skel.shape
    if dcage is None:
        dcage = cage_distance(skel)
    gy, gx, mag = F.gy, F.gx, F.mag
    best = None

    for direction in (1.0, -1.0):
        cy, cx = float(p0[0]), float(p0[1])
        path = []
        ly = lx = 0.0          # last valid unit direction
        coast = 0
        for k in range(max_steps):
            y = int(cy); x = int(cx)
            if y < 0 or y >= H or x < 0 or x >= W:
                break
            path.append((y, x))

            # ---- contact test: radius, not 3x3 ----
            if dcage[y, x] <= contact:
                y0 = max(0, y - 4); y1 = min(H, y + 5)
                x0 = max(0, x - 4); x1 = min(W, x + 5)
                sub = np.argwhere(skel[y0:y1, x0:x1] > 0)
                if len(sub):
                    d = np.hypot(sub[:, 0] + y0 - y, sub[:, 1] + x0 - x)
                    j = int(d.argmin())
                    hit = (y0 + int(sub[j, 0]), x0 + int(sub[j, 1]))
                    if best is None or k < best[2]:
                        best = (path, hit, k + 1, True)
                    break

            m = mag[y, x]
            if m < 0.05:
                # discrete gradient collapsed across the ridge; coast
                if (ly == 0.0 and lx == 0.0) or coast >= max_coast:
                    break
                coast += 1
                cy += ly * alpha; cx += lx * alpha
                continue

            coast = 0
            ly = direction * gy[y, x] / m
            lx = direction * gx[y, x] / m
            cy += ly * alpha
            cx += lx * alpha

    return best if best else ([], None, max_steps, False)


def inject_nearest_pixel(p0, F, skel):
    """Baseline A2a: snap to nearest skeleton pixel, straight connector."""
    t0 = time.perf_counter()
    idx = np.argwhere(skel > 0)
    if len(idx) == 0:
        return None, 0.0, False, True
    d = np.hypot(idx[:, 0] - p0[0], idx[:, 1] - p0[1])
    hit = tuple(idx[int(d.argmin())])
    coll = _segment_hits_obstacle(p0, hit, F)
    return hit, (time.perf_counter() - t0) * 1e3, True, coll


def inject_astar_to_skel(p0, F, skel):
    """Baseline A2b: BFS/A* grid expansion until any skeleton pixel is hit."""
    t0 = time.perf_counter()
    H, W = skel.shape
    free = F.free
    visited = np.zeros((H, W), bool)
    start = (int(p0[0]), int(p0[1]))
    dq = [start]; visited[start] = True
    expansions = 0
    head = 0
    while head < len(dq):
        y, x = dq[head]; head += 1
        expansions += 1
        if skel[y, x]:
            return (y, x), (time.perf_counter() - t0) * 1e3, expansions, True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W and not visited[ny, nx_] and free[ny, nx_]:
                visited[ny, nx_] = True
                dq.append((ny, nx_))
    return None, (time.perf_counter() - t0) * 1e3, expansions, False


def inject_raycast(p0, F, G, k=8):
    """Baseline A2c: EGVG-style raycast toward the k nearest graph vertices."""
    t0 = time.perf_counter()
    if G.number_of_nodes() == 0:
        return None, 0.0, False
    nodes = np.array(list(G.nodes()))
    d = np.hypot(nodes[:, 0] - p0[0], nodes[:, 1] - p0[1])
    for i in np.argsort(d)[:k]:
        cand = tuple(nodes[i])
        if not _segment_hits_obstacle(p0, cand, F):
            return cand, (time.perf_counter() - t0) * 1e3, True
    return None, (time.perf_counter() - t0) * 1e3, False


def _segment_hits_obstacle(a, b, F):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    ys = np.linspace(a[0], b[0], n).astype(int)
    xs = np.linspace(a[1], b[1], n).astype(int)
    ys = np.clip(ys, 0, F.mask.shape[0] - 1)
    xs = np.clip(xs, 0, F.mask.shape[1] - 1)
    return bool((F.mask[ys, xs] != 0).any())


# ======================================================================
# 6. HOMOTOPY ENUMERATION + SELECTION
# ======================================================================
def attach(G, F, skel, hit, tag):
    """Insert an injection contact point into G by splitting its host edge."""
    if hit in G:
        return hit
    best_e, best_i, best_d = None, None, 1e18
    for u, v, data in G.edges(data=True):
        px = data['pixels']
        arr = np.array(px)
        d = np.hypot(arr[:, 0] - hit[0], arr[:, 1] - hit[1])
        i = int(d.argmin())
        if d[i] < best_d:
            best_d, best_e, best_i = d[i], (u, v, data), i
    if best_e is None:
        return None
    u, v, data = best_e
    px = data['pixels']
    node = (hit[0], hit[1])
    if node in G:
        return node
    A, B = px[:best_i + 1], px[best_i:]

    def _w(seg):
        return sum(math.hypot(seg[i][0] - seg[i + 1][0], seg[i][1] - seg[i + 1][1])
                   for i in range(len(seg) - 1)) or 1e-3

    def _p(seg):
        return np.array([F.sdf[q[0], q[1]] for q in seg], np.float32)

    G.remove_edge(u, v)
    if len(A) > 1:
        G.add_edge(u, node, weight=_w(A), pixels=A, prof=_p(A))
    else:
        G.add_edge(u, node, weight=1e-3, pixels=[u, node], prof=_p([u, node]))
    if len(B) > 1:
        G.add_edge(node, v, weight=_w(B), pixels=B, prof=_p(B))
    else:
        G.add_edge(node, v, weight=1e-3, pixels=[node, v], prof=_p([node, v]))
    return node


def enumerate_paths(G, ns, ng, K=None, max_overlap=None):
    K = PARAMS['K_paths'] if K is None else K
    max_overlap = PARAMS['max_overlap'] if max_overlap is None else max_overlap
    out, accepted = [], []
    # Enumerate on a lightweight copy: shortest_simple_paths deep-copies edge
    # payloads, and our edges carry pixel chains + clearance profiles.
    Gw = nx.Graph()
    Gw.add_weighted_edges_from((u, v, d['weight']) for u, v, d in G.edges(data=True))
    if ns not in Gw or ng not in Gw:
        return out
    try:
        gen = nx.shortest_simple_paths(Gw, ns, ng, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return out
    seen_n = 0
    try:
        for np_ in gen:
            seen_n += 1
            if seen_n > 30:
                break
            w = sum(G[np_[i]][np_[i + 1]]['weight'] for i in range(len(np_) - 1))
            ec = {(min(np_[i], np_[i + 1]), max(np_[i], np_[i + 1])):
                  G[np_[i]][np_[i + 1]]['weight'] for i in range(len(np_) - 1)}
            uniq = True
            for ap, aw in accepted:
                ea = {(min(ap[i], ap[i + 1]), max(ap[i], ap[i + 1]))
                      for i in range(len(ap) - 1)}
                shared = sum(v for e, v in ec.items() if e in ea)
                if shared > max_overlap * min(w, aw):
                    uniq = False; break
            if uniq:
                out.append(np_); accepted.append((np_, w))
            if len(out) >= K:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        pass
    return out


def node_path_to_pixels(G, npath):
    px = []
    for i in range(len(npath) - 1):
        seg = list(G[npath[i]][npath[i + 1]]['pixels'])
        if seg[0] != npath[i]:
            seg = seg[::-1]
        px.extend(seg)
    return px


def lex_key(profile, depth=64):
    """Sorted-ascending clearance profile, truncated. Larger is safer."""
    s = np.sort(np.asarray(profile, np.float32))
    if len(s) < depth:
        s = np.concatenate([s, np.full(depth - len(s), s[-1] if len(s) else 0.0)])
    return tuple(np.round(s[:depth], 2))


def select_path(cands, F, criterion='lex'):
    """cands: list of dicts with 'pixels'. criterion in {lex,minclr,short}."""
    if not cands:
        return None, 0
    for c in cands:
        arr = np.array(c['pixels'])
        prof = F.sdf[arr[:, 0], arr[:, 1]]
        c['prof'] = prof
        c['min_clr'] = float(prof.min()) * PX2M
        c['mean_clr'] = float(prof.mean()) * PX2M
        c['len'] = float(np.hypot(*np.diff(arr, axis=0).T).sum()) * PX2M
        c['lex'] = lex_key(prof)

    if criterion == 'short':
        best = min(cands, key=lambda c: c['len'])
        ties = sum(1 for c in cands if abs(c['len'] - best['len']) < 1e-6) - 1
    elif criterion == 'minclr':
        best = max(cands, key=lambda c: c['min_clr'])
        ties = sum(1 for c in cands
                   if abs(c['min_clr'] - best['min_clr']) < 0.05) - 1
    else:
        best = max(cands, key=lambda c: c['lex'])
        ties = sum(1 for c in cands if c['lex'] == best['lex']) - 1
    return best, ties


# ======================================================================
# 7. SELECTIVE RELAXATION + DAMPED BAND
# ======================================================================
def selective_relaxation(path_px, F):
    tr = np.full(F.mask.shape, PARAMS['clear_rigid'], np.float32)
    ts = np.full(F.mask.shape, PARAMS['clear_soft'], np.float32)
    req_r, req_s = {}, {}
    for (r, c) in path_px:
        r, c = int(r), int(c)
        lr, dr = F.near_rigid[r, c], F.sdf_rigid[r, c]
        if lr > 0 and dr < PARAMS['clear_rigid']:
            req_r[lr] = min(req_r.get(lr, 1e9), dr)
        ls, ds = F.near_soft[r, c], F.sdf_soft[r, c]
        if ls > 0 and ds < PARAMS['clear_soft']:
            req_s[ls] = min(req_s.get(ls, 1e9), ds)
    floor = PARAMS['rho_safe']
    for lbl, d in req_r.items():
        tr[F.near_rigid == lbl] = max(floor, d - PARAMS['trim_margin'])
    for lbl, d in req_s.items():
        ts[F.near_soft == lbl] = max(floor, d - PARAMS['trim_margin'])
    return tr, ts, len(req_r) + len(req_s)


def tension_band(path_px, F, thresh_r, thresh_s,
                 damping=True, rest_length=True, decimation='distributed',
                 iter_cap=None):
    """Damped spring-mass band. Returns dict of results + final nodes."""
    iter_cap = PARAMS['iter_cap'] if iter_cap is None else iter_cap
    arr = np.array(path_px, float)
    if len(arr) < 4:
        return dict(ok=False)
    d = np.zeros(len(arr)); d[1:] = np.hypot(*np.diff(arr, axis=0).T)
    cum = np.cumsum(d); total = cum[-1]
    n = max(4, int(total / PARAMS['init_spacing']))
    td = np.linspace(0, total, n)
    P = np.column_stack([np.interp(td, cum, arr[:, 0]),
                         np.interp(td, cum, arr[:, 1])]).astype(np.float32)
    V = np.zeros_like(P)

    d_rest = PARAMS['d_rest'] if rest_length else 0.0
    D = PARAMS['D_damp'] if damping else 1.0     # 1.0 == no dissipation
    H, W = F.mask.shape
    pops, popping, stable, frame = 0, decimation is not None, 0, 0
    pop_rounds = 0
    last_safe = P.copy()
    converged = False

    while frame < iter_cap:
        if popping and frame > 0 and frame % PARAMS['pop_interval'] == 0:
            last_safe = P.copy()
            if len(P) > PARAMS['pop_stride'] * 2 and pop_rounds < PARAMS['max_pop_rounds']:
                pop_rounds += 1
                if decimation == 'distributed':
                    idx = np.arange(PARAMS['pop_stride'], len(P) - 1,
                                    PARAMS['pop_stride'])
                else:  # 'ends'
                    k = len(np.arange(PARAMS['pop_stride'], len(P) - 1,
                                      PARAMS['pop_stride']))
                    idx = np.arange(1, 1 + k)
                P = np.delete(P, idx, axis=0); V = np.zeros_like(P)
                pops += len(idx)
            else:
                popping = False

        # --- vectorised force evaluation over all interior nodes ---
        Y = np.clip(P[1:-1, 0].astype(np.int32), 0, H - 1)
        X = np.clip(P[1:-1, 1].astype(np.int32), 0, W - 1)

        v1 = P[:-2] - P[1:-1]
        v2 = P[2:] - P[1:-1]
        l1 = np.hypot(v1[:, 0], v1[:, 1])[:, None] + 1e-5
        l2 = np.hypot(v2[:, 0], v2[:, 1])[:, None] + 1e-5
        tension = (v1 / l1) * (l1 - d_rest) + (v2 / l2) * (l2 - d_rest)

        repel = np.zeros_like(tension)
        max_pen = 0.0
        for sdfc, gyc, gxc, thr in ((F.sdf_rigid, F.gy_r, F.gx_r, thresh_r),
                                    (F.sdf_soft,  F.gy_s, F.gx_s, thresh_s)):
            dvals = sdfc[Y, X]
            tvals = thr[Y, X]
            pen = np.maximum(tvals - dvals, 0.0)
            if pen.max() > max_pen:
                max_pen = float(pen.max())
            gy_ = gyc[Y, X]; gx_ = gxc[Y, X]
            m = np.hypot(gy_, gx_) + 1e-5
            repel[:, 0] += gy_ / m * pen
            repel[:, 1] += gx_ / m * pen

        f = PARAMS['W_s'] * tension + PARAMS['W_r'] * repel
        Vi = (V[1:-1] + f) * D
        Vi[~np.isfinite(Vi)] = 0.0
        np.clip(Vi, -50.0, 50.0, out=Vi)
        V[1:-1] = Vi
        P[1:-1] += Vi
        max_mv = float(np.hypot(Vi[:, 0], Vi[:, 1]).max()) if len(Vi) else 0.0

        if popping and max_pen > PARAMS['max_penetration']:
            P = last_safe.copy(); V = np.zeros_like(P); popping = False

        if not popping:
            stable = stable + 1 if max_mv < PARAMS['eps_stab'] else 0
            if stable >= PARAMS['n_hold']:
                converged = True
                break
        frame += 1

    return dict(ok=True, P=P, iters=frame, pops=pops, converged=converged,
                spacing_cv=spacing_cv(P), curvature=int_abs_curvature(P),
                jerk=angular_jerk(P), min_clr=path_min_clearance(P, F),
                length=path_length(P))


# ======================================================================
# 8. METRICS
# ======================================================================
def path_length(P):
    P = np.asarray(P, float)
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) * PX2M


def path_min_clearance(P, F):
    P = np.asarray(P, float)
    ys = np.clip(P[:, 0].astype(int), 0, F.sdf.shape[0] - 1)
    xs = np.clip(P[:, 1].astype(int), 0, F.sdf.shape[1] - 1)
    return float(F.sdf[ys, xs].min()) * PX2M


def path_mean_clearance(P, F):
    P = np.asarray(P, float)
    ys = np.clip(P[:, 0].astype(int), 0, F.sdf.shape[0] - 1)
    xs = np.clip(P[:, 1].astype(int), 0, F.sdf.shape[1] - 1)
    return float(F.sdf[ys, xs].mean()) * PX2M


def spacing_cv(P):
    P = np.asarray(P, float)
    if len(P) < 3:
        return float('nan')
    d = np.hypot(*np.diff(P, axis=0).T)
    return float(d.std() / (d.mean() + 1e-9))


def _resample(P, step=3.0):
    P = np.asarray(P, float)
    d = np.zeros(len(P)); d[1:] = np.hypot(*np.diff(P, axis=0).T)
    cum = np.cumsum(d)
    if cum[-1] < step * 3:
        return P
    t = np.arange(0, cum[-1], step)
    return np.column_stack([np.interp(t, cum, P[:, 0]), np.interp(t, cum, P[:, 1])])


def int_abs_curvature(P):
    """Integral of |kappa| ds, in radians (turning number * 2pi units)."""
    Q = _resample(P)
    if len(Q) < 3:
        return float('nan')
    v = np.diff(Q, axis=0)
    th = np.arctan2(v[:, 0], v[:, 1])
    dth = np.abs(np.diff(np.unwrap(th)))
    return float(dth.sum())


def angular_jerk(P):
    """RMS of the third difference of heading -- proxy for angular jerk."""
    Q = _resample(P)
    if len(Q) < 6:
        return float('nan')
    v = np.diff(Q, axis=0)
    th = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    return float(np.sqrt(np.mean(np.diff(th, n=3) ** 2)))


# ======================================================================
# 9. BASELINE PLANNERS
# ======================================================================
def astar_grid(F, s, g, clearance_penalty=False):
    """8-connected A* on the metric grid. Returns (path, expansions, ok)."""
    H, W = F.mask.shape
    free = F.free
    s = (int(s[0]), int(s[1])); g = (int(g[0]), int(g[1]))
    if not free[s] or not free[g]:
        return None, 0, False
    gscore = {s: 0.0}
    came = {}
    openh = [(math.dist(s, g), 0.0, s)]
    closed = np.zeros((H, W), bool)
    exp = 0
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while openh:
        _, gc, cur = heapq.heappop(openh)
        if closed[cur]:
            continue
        closed[cur] = True
        exp += 1
        if cur == g:
            p = [cur]
            while p[-1] in came:
                p.append(came[p[-1]])
            return p[::-1], exp, True
        for dy, dx, w in nbrs:
            ny, nx_ = cur[0] + dy, cur[1] + dx
            if not (0 <= ny < H and 0 <= nx_ < W) or not free[ny, nx_] or closed[ny, nx_]:
                continue
            cost = w
            if clearance_penalty:
                cost += max(0.0, 12.0 - F.sdf[ny, nx_]) * 0.5
            ng = gc + cost
            if ng < gscore.get((ny, nx_), 1e18):
                gscore[(ny, nx_)] = ng
                came[(ny, nx_)] = cur
                heapq.heappush(openh, (ng + math.dist((ny, nx_), g), ng, (ny, nx_)))
    return None, exp, False


def build_prm(F, n_samples=2500, k=12, seed=0):
    rng = np.random.default_rng(seed)
    H, W = F.mask.shape
    pts = []
    guard = 0
    while len(pts) < n_samples and guard < n_samples * 60:
        guard += 1
        y = rng.integers(0, H); x = rng.integers(0, W)
        if F.sdf[y, x] > PARAMS['rho_safe']:
            pts.append((int(y), int(x)))
    pts = np.array(pts)
    G = nx.Graph()
    for i, p in enumerate(pts):
        G.add_node(i, pos=tuple(p))
    for i, p in enumerate(pts):
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
        for j in np.argsort(d)[1:k + 1]:
            j = int(j)
            if G.has_edge(i, j):
                continue
            if not _segment_hits_obstacle(p, pts[j], F):
                G.add_edge(i, j, weight=float(d[j]))
    return G, pts


def query_prm(G, pts, F, s, g):
    def connect(q):
        d = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
        for i in np.argsort(d)[:25]:
            if not _segment_hits_obstacle(q, pts[int(i)], F):
                return int(i)
        return None
    a, b = connect(s), connect(g)
    if a is None or b is None:
        return None, False
    try:
        idx = nx.shortest_path(G, a, b, weight='weight')
    except nx.NetworkXNoPath:
        return None, False
    return [tuple(pts[i]) for i in idx], True


def rrt_star(F, s, g, max_iter=4000, step=18.0, radius=30.0, seed=0):
    rng = np.random.default_rng(seed)
    H, W = F.mask.shape
    V = [np.array(s, float)]; parent = {0: None}; cost = {0: 0.0}
    goal_idx = None
    for it in range(max_iter):
        q = np.array([rng.uniform(0, H), rng.uniform(0, W)]) \
            if rng.random() > 0.05 else np.array(g, float)
        arr = np.array(V)
        near = int(np.argmin(np.hypot(arr[:, 0] - q[0], arr[:, 1] - q[1])))
        d = q - V[near]; n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        new = V[near] + d / n * min(step, n)
        yy, xx = int(new[0]), int(new[1])
        if not (0 <= yy < H and 0 <= xx < W) or F.sdf[yy, xx] <= PARAMS['rho_safe']:
            continue
        if _segment_hits_obstacle(V[near], new, F):
            continue
        V.append(new); i = len(V) - 1
        parent[i] = near; cost[i] = cost[near] + np.linalg.norm(new - V[near])
        arr = np.array(V)
        nearby = np.where(np.hypot(arr[:, 0] - new[0], arr[:, 1] - new[1]) < radius)[0]
        for j in nearby:
            j = int(j)
            if j == i:
                continue
            c = cost[j] + np.linalg.norm(new - V[j])
            if c < cost[i] and not _segment_hits_obstacle(V[j], new, F):
                parent[i] = j; cost[i] = c
        if np.linalg.norm(new - np.array(g)) < step and \
           not _segment_hits_obstacle(new, g, F):
            goal_idx = i
            break
    if goal_idx is None:
        return None, it + 1, False
    path = []; cur = goal_idx
    while cur is not None:
        path.append(tuple(V[cur])); cur = parent[cur]
    return path[::-1] + [tuple(g)], it + 1, True


