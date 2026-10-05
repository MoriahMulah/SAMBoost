"""
mesh_building.py
================
Triangular mesh, projection matrix and SPDE matrices, in Python (SciPy only).

The construction is modelled on R-INLA (inla.nonconvex.hull + inla.mesh.2d)
but is not identical to it:

  1. Inner boundary: a non-convex hull around the observations
     (build_nonconvex_hull).
  2. Outer boundary: the inner hull pushed outwards by outer_expansion
     (expand_hull), so that the field is not distorted near the data edge.
  3. Nodes: points on both boundaries (spacing max_edge_inner / max_edge_outer)
     plus observation locations thinned so that no two nodes are closer than
     cutoff (Poisson-disc thinning). Interior nodes are therefore a subset of
     the observation locations.
  4. Quality: observation nodes that cause triangles with an angle below
     min_angle are removed one at a time, as long as this reduces the number
     of such triangles (build_quality_mesh). No new points are inserted.

Then:
  build_A_matrix          barycentric projection from mesh nodes to points
  assemble_spde_matrices  FEM stiffness G and lumped mass C
  compute_Q               Matern (nu = 1) precision Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G)
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from scipy.sparse import csr_matrix, diags


# >> 1. Non-convex hull  (R-INLA: inla.nonconvex.hull)

def build_nonconvex_hull(
    points: np.ndarray,
    concavity: float = 0.05,
    n_boundary_pts: int = 200,
) -> np.ndarray:
    """
    Build a non-convex (alpha-shape) hull around a point cloud.

    Modelled on R-INLA's inla.nonconvex.hull(points, convex=-concavity).

    Algorithm
    ---------
    1. Delaunay triangulate the points.
    2. Keep triangles whose circumradius < threshold_radius.
       threshold_radius = median_NN_distance / concavity.
       Larger concavity -> tighter hull; concavity=0 -> convex hull.
    3. Extract boundary edges (edges in exactly one kept triangle).
    4. Order boundary edges into a closed polygon.
    5. Resample to n_boundary_pts evenly-spaced vertices.

    Parameters
    ----------
    points         : (N, 2)  observation coordinates
    concavity      : float   hull tightness.
                             0.00 -> convex hull
                             0.03 -> slight concavity (inner boundary)
                             0.10 -> more concave  (outer boundary seed)
    n_boundary_pts : int     number of vertices in the output polygon

    Returns
    -------
    hull_pts : (n_boundary_pts, 2)  ordered boundary polygon vertices
    """
    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    nn_median = np.median(nn_dists[:, 1])

    if concavity <= 0 or nn_median < 1e-12:
        return _convex_hull_polygon(points, n_boundary_pts)

    threshold_radius = nn_median / concavity
    tri  = Delaunay(points)
    cr   = _circumradii(points, tri.simplices)
    keep = cr < threshold_radius

    if keep.sum() < 3:
        return _convex_hull_polygon(points, n_boundary_pts)

    boundary_edges = _extract_boundary_edges(tri.simplices[keep])
    if len(boundary_edges) == 0:
        return _convex_hull_polygon(points, n_boundary_pts)

    polygon = _order_boundary_edges(boundary_edges, points)
    if polygon is None or len(polygon) < 3:
        return _convex_hull_polygon(points, n_boundary_pts)

    return _resample_polygon_n(polygon, n_boundary_pts)


def expand_hull(hull_pts: np.ndarray, distance: float) -> np.ndarray:
    """
    Expand a hull polygon outward by `distance` units.

    Used to create the outer boundary:
        outer = expand_hull(inner_hull, distance=outer_expansion)

    Each vertex is moved outward along the averaged normal of its two
    adjacent edges. This plays the role of R-INLA's second, wider hull.

    Parameters
    ----------
    hull_pts : (K, 2)  ordered boundary polygon (counterclockwise)
    distance : float   outward expansion in coordinate units

    Returns
    -------
    expanded : (K, 2)
    """
    n   = len(hull_pts)
    out = np.empty_like(hull_pts)

    for i in range(n):
        p_prev = hull_pts[(i - 1) % n]
        p_curr = hull_pts[i]
        p_next = hull_pts[(i + 1) % n]

        e1 = p_curr - p_prev
        e2 = p_next - p_curr
        e1_n = e1 / (np.linalg.norm(e1) + 1e-14)
        e2_n = e2 / (np.linalg.norm(e2) + 1e-14)

        # Outward normal for a counter-clockwise polygon: right normal [dy, -dx]
        n1 = np.array([e1_n[1], -e1_n[0]])
        n2 = np.array([e2_n[1], -e2_n[0]])
        normal = (n1 + n2)
        nm = np.linalg.norm(normal)
        if nm > 1e-10:
            normal /= nm

        out[i] = p_curr + distance * normal

    return out


# >> 2. Quality mesh with iterative node removal  (R-INLA: inla.mesh.2d)

def build_quality_mesh(
    obs_locs: np.ndarray,
    inner_hull: np.ndarray,
    outer_hull: np.ndarray,
    cutoff: float = 0.02,
    min_angle: float = 21.0,
    n_interior: int = None,
    max_edge_inner: float = 1.5,
    max_edge_outer: float = 1.0,
    max_removal_iters: int = 50,
    max_candidates: int = 20,
    random_state=None,
) -> np.ndarray:
    """
    Build a Delaunay mesh from obs_locs with iterative node removal for quality.

    Design principle
    ----------------
    Interior nodes come ONLY from obs_locs (Poisson-disk filtered).
    Quality is improved by REMOVING the interior obs-point nodes most
    responsible for bad triangles, until the min_angle target is met
    or no further improvement is possible without removing more nodes.

    Unlike Ruppert refinement, no synthetic points are inserted: the
    observation nodes that cause bad triangles are removed instead, so the
    interior nodes remain a subset of the observation locations.

    Algorithm
    ---------
    1. Seed: boundary nodes (inner + outer hull at max_edge spacing) +
             interior nodes (obs_locs Poisson-disk filtered by cutoff,
             optionally capped at n_interior).
    2. Apply global cutoff.
    3. Triangulate. Find bad interior triangles (min_angle < threshold).
    4. Score each interior (non-frozen) node by how many bad triangles
       it appears in.
    5. Try removing top candidates (up to max_candidates) in order of score.
       Only accept a removal that strictly reduces the bad triangle count.
    6. Repeat from 3. Halt if: target met, no removable culprits, or
       max_candidates consecutive iterations yield no improvement.

    Boundary nodes (inner + outer hull) are FROZEN - never removed.

    Parameters
    ----------
    obs_locs          : (N, 2)
    inner_hull        : (K1, 2)  inner boundary polygon
    outer_hull        : (K2, 2)  outer boundary polygon
    cutoff            : float    min node spacing; controls interior density
    min_angle         : float    target minimum triangle angle (degrees)
    n_interior        : int or None
                                 Hard cap on interior nodes before removal.
                                 Recommended for N > 10 000.
    max_edge_inner    : float    boundary node spacing on inner hull
    max_edge_outer    : float    boundary node spacing on outer hull
    max_removal_iters : int      max removal iterations
    max_candidates    : int      how many candidate nodes to trial per iter
    random_state      : int or None

    Returns
    -------
    mesh_nodes : (M, 2)
    """
    rng = np.random.default_rng(random_state)

    # Step 1: boundary nodes (frozen - never removed)
    inner_boundary = _resample_polygon_spacing(inner_hull, max_edge_inner)
    outer_boundary = _resample_polygon_spacing(outer_hull, max_edge_outer)
    n_frozen       = len(inner_boundary) + len(outer_boundary)

    # Step 2: interior nodes from obs_locs
    interior = _poisson_disk(obs_locs, cutoff, rng)
    if n_interior is not None and len(interior) > n_interior:
        idx      = rng.choice(len(interior), size=n_interior, replace=False)
        interior = interior[idx]

    # Boundary nodes first so _apply_cutoff always preserves them
    nodes   = np.vstack([inner_boundary, outer_boundary, interior])
    nodes   = _apply_cutoff(nodes, cutoff)
    nf      = n_frozen   # number of boundary nodes, which are never removed

    # Steps 3-6: iterative verified removal
    stall = 0
    for _ in range(max_removal_iters):
        tri        = Delaunay(nodes)
        min_angles, _ = _triangle_quality_batch(nodes, tri.simplices)
        centroids  = nodes[tri.simplices].mean(axis=1)
        in_outer   = _points_in_polygon(centroids, outer_hull)

        bad        = in_outer & (min_angles < min_angle)
        n_bad      = bad.sum()

        if n_bad == 0:
            break

        # Score: count how many bad triangles each removable node appears in
        bad_verts     = tri.simplices[bad].ravel()
        removable     = bad_verts[bad_verts >= nf]
        if len(removable) == 0:
            break   # all bad triangles pinned to frozen boundary nodes

        ids, counts = np.unique(removable, return_counts=True)
        # Try top max_candidates in decreasing bad-triangle count
        order     = np.argsort(-counts)
        improved  = False
        for i in order[:max_candidates]:
            cand       = ids[i]
            test_nodes = np.delete(nodes, cand, axis=0)
            test_nf    = nf - (1 if cand < nf else 0)

            # Quick check: does removal reduce bad count?
            tri_t      = Delaunay(test_nodes)
            ma_t, _    = _triangle_quality_batch(test_nodes, tri_t.simplices)
            io_t       = _points_in_polygon(
                             test_nodes[tri_t.simplices].mean(axis=1), outer_hull)
            if (io_t & (ma_t < min_angle)).sum() < n_bad:
                nodes   = test_nodes
                nf      = test_nf
                stall   = 0
                improved = True
                break

        if not improved:
            stall += 1
            if stall >= max_candidates:
                break   # no candidate in top max_candidates helps -> halt

    return nodes


# >> 3. Full pipeline  (R-INLA one-call equivalent)

def build_inla_mesh(
    obs_locs: np.ndarray,
    concavity_inner: float = 0.03,
    concavity_outer: float = 0.10,
    outer_expansion: float = None,
    cutoff: float = 0.02,
    min_angle: float = 21.0,
    n_interior: int = None,
    max_edge_inner: float = 1.5,
    max_edge_outer: float = 1.0,
    n_hull_pts: int = 200,
    max_removal_iters: int = 50,
    max_candidates: int = 20,
    random_state=None,
) -> dict:
    """
    Full mesh pipeline in one call, modelled on the R-INLA calls:
        bound1 = inla.nonconvex.hull(pts, convex=-concavity_inner)
        bound2 = inla.nonconvex.hull(pts, convex=-concavity_outer)
        mesh   = inla.mesh.2d(pts, boundary=list(bound1, bound2),
                              cutoff=cutoff, min.angle=min_angle,
                              max.edge=c(max_edge_inner, max_edge_outer))
    Here the outer boundary is the inner hull expanded by outer_expansion.

    Interior nodes come ONLY from obs_locs (Poisson-disk filtered).
    Quality is improved by iteratively removing the obs-point node most
    responsible for bad triangles (min_angle < threshold), rather than
    inserting synthetic circumcenter points.

    Node count: <= n_interior + n_boundary, and decreases during
    refinement. Use node_counts in the output to inspect the breakdown.

    Parameters
    ----------
    obs_locs           : (N, 2)   observation coordinates (lon, lat)
    concavity_inner    : float    inner hull tightness  (R: convex=-0.03)
    concavity_outer    : float    not used (the outer hull is the expanded
                                  inner hull); kept for compatibility
    outer_expansion    : float    outward distance of outer hull beyond inner.
                                  Default: 3 * max_edge_outer.
                                  For lon/lat data: 0.05-0.3 degrees
    cutoff             : float    minimum interior node spacing.
                                  For lon/lat data: 0.002-0.05 degrees
    min_angle          : float    target minimum interior triangle angle (deg).
                                  Same as R-INLA min.angle. Default 21.
                                  Nodes causing triangles below this are removed.
    n_interior         : int or None
                                  Hard cap on interior nodes before refinement.
                                  Recommended for N > 10 000. None = no cap.
    max_edge_inner     : float    node spacing along the inner hull
    max_edge_outer     : float    node spacing along the outer hull
                                  (neither limits edge lengths inside the domain)
    n_hull_pts         : int      hull polygon resolution
    max_removal_iters  : int      max removal iterations (default 50)
    max_candidates     : int      candidates tried per iteration (default 20)
    random_state       : int or None

    Returns
    -------
    dict:
        mesh_nodes   : (M, 2)    final mesh node coordinates
        inner_hull   : (K1, 2)   inner boundary polygon
        outer_hull   : (K2, 2)   outer boundary polygon
        tri          : Delaunay  final triangulation
        quality      : dict      mesh quality statistics
        node_counts  : dict      node source breakdown
    """
    if outer_expansion is None:
        outer_expansion = 3.0 * max_edge_outer

    inner_hull = build_nonconvex_hull(obs_locs, concavity_inner, n_hull_pts)
    outer_hull = expand_hull(inner_hull, outer_expansion)

    mesh_nodes = build_quality_mesh(
        obs_locs          = obs_locs,
        inner_hull        = inner_hull,
        outer_hull        = outer_hull,
        cutoff            = cutoff,
        min_angle         = min_angle,
        n_interior        = n_interior,
        max_edge_inner    = max_edge_inner,
        max_edge_outer    = max_edge_outer,
        max_removal_iters = max_removal_iters,
        max_candidates    = max_candidates,
        random_state      = random_state,
    )

    tri = Delaunay(mesh_nodes)

    min_angles, max_edges = _triangle_quality_batch(mesh_nodes, tri.simplices)
    centroids = mesh_nodes[tri.simplices].mean(axis=1)
    in_outer  = _points_in_polygon(centroids, outer_hull)
    in_inner  = _points_in_polygon(centroids, inner_hull)

    def _ss(arr, fn):
        return float(fn(arr)) if len(arr) > 0 else float('nan')

    quality = {
        'n_nodes'              : int(len(mesh_nodes)),
        'n_triangles'          : int(len(tri.simplices)),
        'n_interior_triangles' : int(in_outer.sum()),
        'min_angle_interior'   : _ss(min_angles[in_outer], np.min),
        'mean_min_angle'       : _ss(min_angles[in_outer], np.mean),
        'max_edge_interior'    : _ss(max_edges[in_inner], np.max),
        'pct_bad_angle'        : _ss(min_angles[in_outer] < min_angle,
                                     lambda x: x.mean() * 100),
    }

    n_inner_b = len(_resample_polygon_spacing(inner_hull, max_edge_inner))
    n_outer_b = len(_resample_polygon_spacing(outer_hull, max_edge_outer))
    node_counts = {
        'inner_boundary' : n_inner_b,
        'outer_boundary' : n_outer_b,
        'interior_obs'   : max(0, len(mesh_nodes) - n_inner_b - n_outer_b),
        'total'          : int(len(mesh_nodes)),
    }

    return {
        'mesh_nodes'  : mesh_nodes,
        'inner_hull'  : inner_hull,
        'outer_hull'  : outer_hull,
        'tri'         : tri,
        'quality'     : quality,
        'node_counts' : node_counts,
    }


# >> Private geometry helpers

def _circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Circumradius of each triangle. (n_tri,)"""
    p  = points[simplices]
    a  = np.linalg.norm(p[:,1]-p[:,0], axis=1)
    b  = np.linalg.norm(p[:,2]-p[:,1], axis=1)
    c  = np.linalg.norm(p[:,0]-p[:,2], axis=1)
    x0,y0 = p[:,0,0], p[:,0,1]
    x1,y1 = p[:,1,0], p[:,1,1]
    x2,y2 = p[:,2,0], p[:,2,1]
    area = 0.5 * np.abs(x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (a * b * c) / (4.0 * area)
    r[area < 1e-14] = np.inf
    return r


def _triangle_quality_batch(points, simplices):
    """
    Returns (min_angles, max_edges) for all triangles.
    min_angles : (n_tri,)  minimum interior angle in degrees
    max_edges  : (n_tri,)  longest edge length
    """
    p  = points[simplices]
    e0 = np.linalg.norm(p[:,1]-p[:,0], axis=1)
    e1 = np.linalg.norm(p[:,2]-p[:,1], axis=1)
    e2 = np.linalg.norm(p[:,0]-p[:,2], axis=1)

    eps = 1e-14
    # Angle at each vertex opposite the edge not touching it
    cos_v0 = (e0**2 + e2**2 - e1**2) / (2*e0*e2 + eps)
    cos_v1 = (e0**2 + e1**2 - e2**2) / (2*e0*e1 + eps)
    cos_v2 = (e1**2 + e2**2 - e0**2) / (2*e1*e2 + eps)

    A0 = np.degrees(np.arccos(np.clip(cos_v0, -1, 1)))
    A1 = np.degrees(np.arccos(np.clip(cos_v1, -1, 1)))
    A2 = np.degrees(np.arccos(np.clip(cos_v2, -1, 1)))

    min_angles = np.minimum(np.minimum(A0, A1), A2)
    max_edges  = np.maximum(np.maximum(e0, e1), e2)
    return min_angles, max_edges


def _extract_boundary_edges(simplices: np.ndarray) -> list:
    """Edges belonging to exactly one triangle."""
    from collections import Counter
    ec = Counter()
    for tri in simplices:
        for k in range(3):
            ec[tuple(sorted([tri[k], tri[(k+1)%3]]))] += 1
    return [e for e, cnt in ec.items() if cnt == 1]


def _order_boundary_edges(edges: list, points: np.ndarray):
    """Order boundary edges into a closed polygon. Returns coordinates or None."""
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    start = edges[0][0]
    path  = [start]
    prev, cur = None, start

    for _ in range(len(edges) + 1):
        nbs = [nb for nb in adj[cur] if nb != prev]
        if not nbs or nbs[0] == start:
            break
        path.append(nbs[0])
        prev, cur = cur, nbs[0]

    return points[path] if len(path) >= 3 else None


def _resample_polygon_n(polygon: np.ndarray, n: int) -> np.ndarray:
    """Resample polygon to exactly n evenly-spaced vertices."""
    closed  = np.vstack([polygon, polygon[0]])
    diffs   = np.diff(closed, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    total   = lengths.sum()
    if total < 1e-12:
        return np.tile(polygon[0], (n, 1))
    t_orig  = np.concatenate([[0], np.cumsum(lengths)]) / total
    t_new   = np.linspace(0, 1, n, endpoint=False)
    xs = np.interp(t_new, t_orig, closed[:, 0])
    ys = np.interp(t_new, t_orig, closed[:, 1])
    return np.column_stack([xs, ys])


def _resample_polygon_spacing(polygon: np.ndarray, max_spacing: float) -> np.ndarray:
    """Resample polygon so no edge exceeds max_spacing."""
    closed  = np.vstack([polygon, polygon[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total   = lengths.sum()
    n       = max(3, int(np.ceil(total / max_spacing)))
    return _resample_polygon_n(polygon, n)


def _convex_hull_polygon(points: np.ndarray, n: int) -> np.ndarray:
    """Fallback: evenly-resampled convex hull."""
    hull = ConvexHull(points)
    return _resample_polygon_n(points[hull.vertices], n)


def _points_in_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Ray-casting point-in-polygon test. Returns boolean (N,).
    Vectorised over all pts simultaneously.
    """
    n      = len(polygon)
    inside = np.zeros(len(pts), dtype=bool)
    px, py = pts[:, 0], pts[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cross = ((yi > py) != (yj > py)) & \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-14) + xi)
        inside ^= cross
        j = i
    return inside


def _apply_cutoff(nodes: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Greedy Poisson-disk filter: keep nodes with minimum spacing = cutoff.

    O(N log N) average via query_ball_point on the full candidate tree.
    Guarantees zero spacing violations.

    Algorithm: build KDTree over ALL candidates, iterate in order,
    accept a node if not yet rejected, then mark all its neighbours
    within cutoff as rejected.
    """
    if len(nodes) <= 1:
        return nodes
    tree     = cKDTree(nodes)
    rejected = np.zeros(len(nodes), dtype=bool)
    kept     = np.zeros(len(nodes), dtype=bool)
    for i in range(len(nodes)):
        if rejected[i]:
            continue
        kept[i] = True
        for nb in tree.query_ball_point(nodes[i], cutoff - 1e-10):
            if nb != i:
                rejected[nb] = True
    return nodes[kept]


def _poisson_disk(points: np.ndarray, cutoff: float, rng) -> np.ndarray:
    """Shuffle then apply cutoff filter (random Poisson disk sampling)."""
    idx = rng.permutation(len(points))
    return _apply_cutoff(points[idx], cutoff)


# >> Projection matrix A

def build_A_matrix(obs_locs: np.ndarray, mesh_nodes: np.ndarray) -> csr_matrix:
    """
    Build the barycentric projection matrix A.

    For each observation i, finds the triangle it falls in and computes
    its 3 barycentric weights w.r.t. the triangle's vertices. Row i of A
    has exactly 3 non-zeros (the weights), summing to 1.

    A maps mesh-node values to observation-level values:
        z_obs = A @ z_nodes

    Parameters
    ----------
    obs_locs   : (N, 2)  observation coordinates
    mesh_nodes : (M, 2)  mesh node coordinates (from build_inla_mesh)

    Returns
    -------
    A : (N, M) csr_matrix  barycentric projection matrix

    Notes
    -----
    Observations outside the mesh get all-zero rows in A and a warning.
    This does not happen with meshes from build_inla_mesh, whose outer
    boundary lies beyond the data.
    """
    N = len(obs_locs)
    M = len(mesh_nodes)

    tri = Delaunay(mesh_nodes)
    s   = tri.find_simplex(obs_locs)   # (N,), -1 if outside the mesh

    valid   = s >= 0
    n_out   = (~valid).sum()
    if n_out > 0:
        import warnings
        warnings.warn(
            f"{n_out} observation(s) fall outside the mesh and will have "
            "all-zero rows in A. Check that the mesh covers all observations.",
            RuntimeWarning,
        )

    s_v     = s[valid]

    # Barycentric coordinates from scipy's affine transform: tri.transform[k]
    # has shape (3, 2), rows 0-1 the inverse 2x2 matrix T, row 2 the third
    # vertex r3. Then (w1, w2) = T @ (x - r3) and w3 = 1 - w1 - w2.
    Xt      = tri.transform[s_v]                                  # (N_valid, 3, 2)
    r3      = tri.points[tri.simplices[s_v, 2]]                   # (N_valid, 2)
    Y       = obs_locs[valid] - r3                                 # (N_valid, 2)
    w12     = np.einsum('ijk,ik->ij', Xt[:, :2, :], Y)            # (N_valid, 2)
    weights = np.c_[w12, 1.0 - w12.sum(axis=1)]                   # (N_valid, 3)

    # Clip tiny numerical negatives to 0 (floating point from degenerate triangles)
    weights = np.clip(weights, 0.0, None)
    # Re-normalise to ensure rows sum exactly to 1
    row_sums = weights.sum(axis=1, keepdims=True)
    weights  = weights / np.where(row_sums > 0, row_sums, 1.0)

    # Build sparse matrix
    valid_idx = np.where(valid)[0]                                 # original row indices
    rows = np.repeat(valid_idx, 3)
    cols = tri.simplices[s_v].flatten()
    data = weights.flatten()

    A = csr_matrix((data, (rows, cols)), shape=(N, M))
    return A


# >> SPDE matrices: G (stiffness) and C (mass) - fully vectorised

def assemble_spde_matrices(mesh_nodes: np.ndarray):
    """
    Assemble the FEM stiffness matrix G and mass matrix C for the SPDE
    discretisation on a Delaunay triangulation.

    Fully vectorised - no Python loops over triangles.

    Parameters
    ----------
    mesh_nodes : (M, 2)  mesh node coordinates

    Returns
    -------
    G : (M, M) csr_matrix  stiffness matrix
    C : (M, M) diags       mass matrix (diagonal)

    Mathematical background
    -----------------------
    For each triangle T with vertices p0, p1, p2 and area |T|:

    Mass matrix (lumped diagonal approximation):
        C[i, i] += |T| / 3   for each vertex i of T

    Stiffness matrix (from linear basis functions):
        G[i, j] += (grad phi_i * grad phi_j) * |T|

    where grad phi_i = (1/2|T|) * (edge opposite node i, rotated by 90 degrees)
    equivalently: the contribution G_loc[i,j] = v_i * v_j / (4*|T|)
    where v_i is the edge OPPOSITE to node i: v_i = p_{i+2} - p_{i+1}
    """
    tri      = Delaunay(mesh_nodes)
    simplices = tri.simplices           # (n_tri, 3)  node indices
    n_nodes  = len(mesh_nodes)
    n_tri    = len(simplices)

    # Node coordinates for all triangles at once
    # p: (n_tri, 3, 2) - coords of each vertex of each triangle
    p = mesh_nodes[simplices]           # (n_tri, 3, 2)

    # Triangle areas (vectorised)
    # Area = 0.5 |x0(y1-y2) + x1(y2-y0) + x2(y0-y1)|
    x0, y0 = p[:, 0, 0], p[:, 0, 1]
    x1, y1 = p[:, 1, 0], p[:, 1, 1]
    x2, y2 = p[:, 2, 0], p[:, 2, 1]
    areas = 0.5 * np.abs(x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1))  # (n_tri,)

    # Guard against degenerate (zero-area) triangles
    degenerate = areas < 1e-14
    if degenerate.any():
        import warnings
        warnings.warn(
            f"{degenerate.sum()} degenerate triangle(s) found (area < 1e-14). "
            "These will contribute zero to G and C.",
            RuntimeWarning,
        )
    areas = np.where(degenerate, 1.0, areas)   # avoid /0; contribution is 0

    # Mass matrix C (diagonal)
    # Each vertex of each triangle contributes area/3 to its diagonal entry
    C_diag = np.zeros(n_nodes)
    # simplices: (n_tri, 3) - add area/3 for each of the 3 vertices
    contrib = areas / 3.0               # (n_tri,)
    for local_v in range(3):
        np.add.at(C_diag, simplices[:, local_v], contrib)
    # Zero out degenerate triangle contributions
    # (already handled: contrib=0 if area was degenerate before replacement)

    # Stiffness matrix G (vectorised)
    # Edge vectors opposite each vertex:
    #   v[k, i] = p[k, (i+2)%3] - p[k, (i+1)%3]   shape (n_tri, 3, 2)
    v = np.stack([
        p[:, (i+2) % 3, :] - p[:, (i+1) % 3, :]
        for i in range(3)
    ], axis=1)                          # (n_tri, 3, 2)

    # Local stiffness: G_loc[k, i, j] = dot(v[k,i], v[k,j]) / (4 * area[k])
    # Compute all 9 (i,j) pairs for each triangle:
    #   dot products: (n_tri, 3, 3)
    vvt   = np.einsum('nid,njd->nij', v, v)              # (n_tri, 3, 3)
    G_loc = vvt / (4.0 * areas[:, None, None])            # (n_tri, 3, 3)

    # Zero out degenerate triangles
    G_loc[degenerate] = 0.0

    # Assemble into global G using all 9 (i,j) pairs per triangle
    # Row and column indices for all entries
    # local_i, local_j: each in {0,1,2}
    li, lj = np.meshgrid(range(3), range(3), indexing='ij')  # (3,3) grids
    li = li.ravel()   # [0,0,0,1,1,1,2,2,2]
    lj = lj.ravel()   # [0,1,2,0,1,2,0,1,2]

    # Global node indices: (n_tri, 9) for rows and cols
    row_idx = simplices[:, li]          # (n_tri, 9)
    col_idx = simplices[:, lj]          # (n_tri, 9)
    g_data  = G_loc[:, li, lj]         # (n_tri, 9)

    G = csr_matrix(
        (g_data.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(n_nodes, n_nodes),
    )
    # Sum duplicate entries (scipy csr_matrix does this automatically)

    C = diags(C_diag, format="csr")

    return G, C


# >> Precision matrix Q

def compute_Q(
    G: csr_matrix,
    C: diags,
    kappa: float,
    tau: float = 1.0,
) -> csr_matrix:
    """
    Compute the SPDE precision matrix Q.

    For the Matern nu=1 field on R^2:
        Q = tau^2 * (kappa^4*C + 2 kappa^2*G + G*C^-1*G)

    Parameters
    ----------
    G     : (M, M) stiffness matrix from assemble_spde_matrices()
    C     : (M, M) mass matrix from assemble_spde_matrices()
    kappa : float  inverse range; practical range rho = sqrt(8) / kappa
    tau   : float  variance scale; marginal variance = 1 / (4 pi kappa^2 tau^2)

    Returns
    -------
    Q : (M, M) csr_matrix  sparse symmetric positive definite precision matrix

    Parameter interpretation
    ------------------------
    kappa: controls how fast spatial correlation decays.
        Small kappa (e.g. 0.1) -> very long-range, global smoothing
        Large kappa (e.g. 5.0) -> short-range, local smoothing
        Rule of thumb: set kappa ~ sqrt(8) / desired_range_in_coord_units

    tau: controls the overall variance of the spatial field.
        Large tau -> small variance, small tau -> large variance.
        The models fix tau = 1 / (sqrt(4 pi) kappa) so that the variance is 1.
    """
    C_inv = diags(1.0 / C.diagonal(), format="csr")

    term1 = (kappa ** 4) * C
    term2 = 2.0 * (kappa ** 2) * G
    term3 = G @ C_inv @ G

    Q = (tau ** 2) * (term1 + term2 + term3)
    return Q.tocsr()

