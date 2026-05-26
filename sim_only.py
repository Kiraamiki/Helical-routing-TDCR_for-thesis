import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.optimize import least_squares


# =========================================================
# 1) Robot parameters
# =========================================================
@dataclass
class RobotParams:
    num_disks: int = 6
    disk_spacing: float = 0.05
    r_disk: float = 0.020

    # three tendons (k=0 is your pulled one)
    alpha_0: np.ndarray = field(default_factory=lambda: np.array([0.0, 2*np.pi/3, 4*np.pi/3]))

    # base -> tip total helical offset
    helix_angle: float = np.deg2rad(240)

    # material
    E: float = 0.50e9
    G: float = 0.17e9
    rad_rod: float = 0.0025

    @property
    def L(self):
        return self.num_disks * self.disk_spacing

    @property
    def K(self):
        I = np.pi * self.rad_rod**4 / 4.0
        J = np.pi * self.rad_rod**4 / 2.0
        return np.diag([self.E * I, self.E * I, self.G * J])

    @property
    def K_inv(self):
        return np.linalg.inv(self.K)


# =========================================================
# 2) Utilities
# =========================================================
def skew(v):
    return np.array([[0.0,   -v[2],  v[1]],
                     [v[2],   0.0,  -v[0]],
                     [-v[1],  v[0],  0.0]])

def project_points(P, plane="xz"):
    if plane == "xy":
        return P[:, 0], P[:, 1]
    if plane == "xz":
        return P[:, 0], P[:, 2]
    if plane == "yz":
        return P[:, 1], P[:, 2]
    raise ValueError("plane must be 'xy', 'xz', or 'yz'")


# =========================================================
# 3) Discrete disk angles (with assembly error)
#    alpha_disks shape: (num_disks+1, 3)
# =========================================================
def build_alpha_disks(p: RobotParams, delta_list=None):
    if delta_list is None:
        delta_list = np.ones(p.num_disks) * (p.helix_angle / p.num_disks)
    delta_list = np.array(delta_list, dtype=float)

    if len(delta_list) != p.num_disks:
        raise ValueError("delta_list length must equal num_disks")

    alpha_disks = np.zeros((p.num_disks + 1, 3), dtype=float)
    alpha_disks[0, :] = p.alpha_0

    acc = 0.0
    for j in range(1, p.num_disks + 1):
        acc += delta_list[j - 1]
        alpha_disks[j, :] = p.alpha_0 + acc

    return alpha_disks


def random_delta_list(num_disks, total_angle, deg_min=30, deg_max=60, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(np.deg2rad(deg_min), np.deg2rad(deg_max), size=num_disks)
    raw_sum = np.sum(raw)
    if raw_sum <= 1e-12:
        return np.ones(num_disks) * (total_angle / num_disks)
    return raw * (total_angle / raw_sum)


# =========================================================
# 4) Friction / capstan-like tension propagation
# =========================================================
def compute_segment_tensions_per_tendon(T0, alpha_k, p: RobotParams, mu=0.6):
    """
    T0: base tension (scalar)
    alpha_k: (num_disks+1,) angles for tendon k at each disk
    return: Tseg (num_disks,) effective segment tensions from base to tip
    """
    n = p.num_disks
    Tseg = np.zeros(n, dtype=float)
    if T0 <= 0.0:
        return Tseg

    Tseg[0] = float(T0)

    # build segment direction vectors dp_j
    dp_list = []
    for j in range(n):
        a0 = float(alpha_k[j])
        a1 = float(alpha_k[j + 1])
        r0 = np.array([p.r_disk * np.cos(a0), p.r_disk * np.sin(a0), 0.0])
        r1 = np.array([p.r_disk * np.cos(a1), p.r_disk * np.sin(a1), 0.0])
        dp = (r1 - r0) + np.array([0.0, 0.0, p.disk_spacing])
        dp_list.append(dp)

    # propagate tension along disks
    for j in range(n - 1):
        v1 = dp_list[j]
        v2 = dp_list[j + 1]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
        c = np.dot(v1, v2) / denom
        c = float(np.clip(c, -1.0, 1.0))
        dpsi = np.arccos(c)  # turning angle
        Tseg[j + 1] = Tseg[j] * np.exp(-mu * dpsi)

    return Tseg


# =========================================================
# 5) Forward statics (discrete disks + friction)
# =========================================================
def solve_forward_statics(
    Tensions,
    p: RobotParams,
    N=80,
    delta_list=None,
    enforce_no_twist=True,
    mu_friction=0.6,
):
    Tensions = np.array(Tensions, dtype=float).reshape(3)

    ds = p.L / (N - 1)
    s_eval = np.linspace(0.0, p.L, N)

    alpha_disks = build_alpha_disks(p, delta_list)
    assert alpha_disks.shape == (p.num_disks + 1, 3)

    # Precompute per-segment effective tension for each tendon
    # Tseg[k][j] => tendon k, segment j
    Tseg = []
    for k in range(3):
        Tseg_k = compute_segment_tensions_per_tendon(
            float(Tensions[k]),
            alpha_disks[:, k],   # (num_disks+1,)
            p,
            mu=mu_friction
        )
        Tseg.append(Tseg_k)

    pos = np.zeros(3)
    R = np.eye(3)

    p_list = np.zeros((N, 3))
    R_list = np.zeros((N, 3, 3))
    R_list[0] = R

    for i in range(N - 1):
        s = float(s_eval[i])

        j = int(np.floor(s / p.disk_spacing))
        j = int(np.clip(j, 0, p.num_disks - 1))

        M_local = np.zeros(3)

        for k in range(3):
            T = float(Tseg[k][j])   # ✅ 用该段有效张力
            if T <= 0.0:
                continue

            a0 = float(alpha_disks[j, k])
            a1 = float(alpha_disks[j + 1, k])

            r0 = np.array([p.r_disk * np.cos(a0), p.r_disk * np.sin(a0), 0.0])
            r1 = np.array([p.r_disk * np.cos(a1), p.r_disk * np.sin(a1), 0.0])

            # segment hole position: constant within segment (NO interpolation)
            r_vec = r0

            # tendon direction as straight segment between disk holes
            dp = (r1 - r0) + np.array([0.0, 0.0, p.disk_spacing])
            t_hat = dp / (np.linalg.norm(dp) + 1e-12)

            f_local = T * t_hat
            M_local += np.cross(r_vec, f_local)

        u = p.K_inv @ M_local

        # Twist handling
        if enforce_no_twist:
            u[2] = 0.0

        # Integrate backbone
        pos = pos + (R @ np.array([0.0, 0.0, 1.0])) * ds
        R = R + (R @ skew(u)) * ds

        # Re-orthonormalize
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        p_list[i + 1] = pos
        R_list[i + 1] = R

    return p_list, R_list


# =========================================================
# 6) Solve tensions from dL (length changes)
#    calc_len uses SAME discrete geometry (r_vec=r0)
# =========================================================
def solve_shape_from_dL(dL_target, params: RobotParams, delta_list=None, enforce_no_twist=True, mu_friction=0.6):
    dL_target = np.array(dL_target, dtype=float).reshape(3)

    alpha_disks = build_alpha_disks(params, delta_list)

    def calc_len(p_list, R_list):
        ls = np.zeros(3)

        prev = np.zeros((3, 3))
        j0 = 0
        for k in range(3):
            a0 = float(alpha_disks[j0, k])
            r0 = np.array([params.r_disk * np.cos(a0), params.r_disk * np.sin(a0), 0.0])
            prev[k] = p_list[0] + R_list[0] @ r0

        Nn = len(p_list)
        s_vals = np.linspace(0.0, params.L, Nn)

        for i in range(1, Nn):
            s = float(s_vals[i])
            j = int(np.floor(s / params.disk_spacing))
            j = int(np.clip(j, 0, params.num_disks - 1))

            for k in range(3):
                a0 = float(alpha_disks[j, k])
                r0 = np.array([params.r_disk * np.cos(a0), params.r_disk * np.sin(a0), 0.0])  # SAME
                curr = p_list[i] + R_list[i] @ r0
                ls[k] += np.linalg.norm(curr - prev[k])
                prev[k] = curr

        return ls

    # Base (zero tension) length
    p0, R0 = solve_forward_statics(
        np.array([0.0, 0.0, 0.0]),
        params,
        N=120,
        delta_list=delta_list,
        enforce_no_twist=enforce_no_twist,
        mu_friction=mu_friction
    )
    L0 = calc_len(p0, R0)

    L_target = L0 + dL_target
    is_active = dL_target < -1e-6

    # Optional pretension (leave 0 for now)
    T_pre = np.array([0.0, 0.0, 0.0])

    def residual(T_guess):
        T_guess = np.maximum(np.array(T_guess, dtype=float), 0.0)

        p_curr, R_curr = solve_forward_statics(
            T_guess,
            params,
            N=80,
            delta_list=delta_list,
            enforce_no_twist=enforce_no_twist,
            mu_friction=mu_friction
        )
        L_curr = calc_len(p_curr, R_curr)

        res = []
        for k in range(3):
            if is_active[k]:
                res.append((L_curr[k] - L_target[k]) * 1000.0)  # mm
            else:
                res.append((T_guess[k] - T_pre[k]) * 1.0)
        return np.array(res, dtype=float)

    T0 = np.zeros(3)
    T0[is_active] = 20.0

    sol = least_squares(residual, T0, bounds=(0.0, np.inf), method="trf")

    p_fin, R_fin = solve_forward_statics(
        sol.x,
        params,
        N=160,
        delta_list=delta_list,
        enforce_no_twist=enforce_no_twist,
        mu_friction=mu_friction
    )
    return p_fin, R_fin, sol.x


# =========================================================
# 7) Simulation-only plotting: XY / XZ / YZ projections
# =========================================================
def run_simulation_projections_only(delta_list=None, enforce_no_twist=True, mu_friction=0.8):
    params = RobotParams()

    sim_steps = np.array([-0.010, -0.015,-0.020, -0.025,-0.030, -0.035,-0.040, -0.045,-0.050, -0.055,-0.060])
    planes = [("XY view", "xy"), ("XZ view", "xz"), ("YZ view", "yz")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors = plt.cm.Blues(np.linspace(0.25, 0.95, len(sim_steps)))

    for ax, (title, plane) in zip(axes, planes):
        for i, dL in enumerate(sim_steps):
            p, _, Tsol = solve_shape_from_dL(
                np.array([dL, 0.0, 0.0]),
                params,
                delta_list=delta_list,
                enforce_no_twist=enforce_no_twist,
                mu_friction=mu_friction
            )
            u, v = project_points(p, plane=plane)
            ax.plot(u, v, linestyle="--", linewidth=2.0, alpha=0.8, color=colors[i])
            ax.scatter(u[-1], v[-1], s=18, color=colors[i])

        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(plane[0].upper() + " (m)")
        ax.set_ylabel(plane[1].upper() + " (m)")

    labels = [f"dL={dL*1000:.0f}mm" for dL in sim_steps]
    fig.legend(labels, loc="upper center", ncol=len(sim_steps), frameon=True)

    fig.suptitle(
        f"Simulation Backbone Projections (L={params.L:.2f}m, discrete disks, no_twist={enforce_no_twist}, mu={mu_friction})",
        y=1.02, fontsize=14
    )
    plt.tight_layout()
    plt.show()


# =========================================================
# 8) Main
# =========================================================
if __name__ == "__main__":
    params = RobotParams()

    # ====== YOU ONLY TUNE THIS ======
    MU_FRICTION = 0.8   # try 0.2, 0.4, 0.6, 0.8, 1.0, 1.2
    ENFORCE_NO_TWIST = True

    # Choose ONE assembly realization
    delta = random_delta_list(
        params.num_disks,
        total_angle=params.helix_angle,
        deg_min=30,
        deg_max=60,
        seed=2
    )

    run_simulation_projections_only(delta_list=delta, enforce_no_twist=ENFORCE_NO_TWIST, mu_friction=MU_FRICTION)

    # If you want multiple assembly states (envelope):
    # for seed in [0, 1, 2, 3, 4]:
    #     delta = random_delta_list(params.num_disks, params.helix_angle, 30, 60, seed=seed)
    #     run_simulation_projections_only(delta_list=delta, enforce_no_twist=ENFORCE_NO_TWIST, mu_friction=MU_FRICTION)
