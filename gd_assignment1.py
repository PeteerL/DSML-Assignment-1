"""DSML Lab Assignment 1 - Gradient Descent as a Dynamical System.

Run:  python gd_assignment1.py
Prints every number quoted in DSML_Assignment_1_Solution.md and writes the
figures to ./figures/.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close(fig)


# Reusable optimizer
def gradient_descent(f, grad_f, x0, eta, tol=1e-6, max_iter=10000, max_norm=1e6):
    """Iterate x_{k+1} = x_k - eta * grad_f(x_k).

    tol=None -> comparison mode (exactly max_iter updates unless a safety check fires).
    Returns a dict with histories (copies of every state, including x0), the
    number of completed updates and a stop reason.
    """
    x = np.array(x0, dtype=float).reshape(-1).copy()
    g = np.asarray(grad_f(x), dtype=float).reshape(-1)
    xs, losses, gnorms = [x.copy()], [float(f(x))], [float(np.linalg.norm(g))]

    def finite(x, loss, g):
        return np.all(np.isfinite(x)) and np.isfinite(loss) and np.all(np.isfinite(g))

    status, n_updates = None, 0
    if not finite(x, losses[0], g):
        status = "nonfinite"
    elif tol is not None and gnorms[0] < tol:
        status = "converged"

    while status is None:
        if n_updates >= max_iter:
            status = "max_iter"
            break
        x = x - eta * g
        n_updates += 1
        loss = float(f(x))
        g = np.asarray(grad_f(x), dtype=float).reshape(-1)
        xs.append(x.copy())
        losses.append(loss)
        gnorms.append(float(np.linalg.norm(g)))
        if not finite(x, loss, g):
            status = "nonfinite"
        elif np.linalg.norm(x) > max_norm:
            status = "safety_stop"
        elif tol is not None and gnorms[-1] < tol:
            status = "converged"

    return {
        "x": np.array(xs),
        "loss": np.array(losses),
        "grad_norm": np.array(gnorms),
        "n_updates": n_updates,
        "status": status,
    }


def converge(f, g, x0, eta):
    return gradient_descent(f, g, x0, eta, tol=1e-6, max_iter=10000)


def compare(f, g, x0, eta):
    return gradient_descent(f, g, x0, eta, tol=None, max_iter=40)


def summary(r):
    return (f"x_final={np.array2string(r['x'][-1], precision=10)}, "
            f"loss={r['loss'][-1]:.10g}, |grad|={r['grad_norm'][-1]:.3e}, "
            f"updates={r['n_updates']}, status={r['status']}")


def nz(a):
    """Replace exact zeros by NaN so they are omitted on a log axis."""
    a = np.array(a, dtype=float)
    a[a == 0] = np.nan
    return a



# Problem 1
print("=" * 70, "\nPROBLEM 1")
f1 = lambda x: float(x[0] ** 2 + 3 * x[0] + 2)
g1 = lambda x: np.array([2 * x[0] + 3])
xs1 = -1.5
fs1 = f1([xs1])
print(f"x* = {xs1}, f(x*) = {fs1}")

# Task 1.1
r = converge(f1, g1, [10.0], 0.1)
print("Task 1.1:", summary(r))
xx = np.linspace(-14, 12, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(xx, xx ** 2 + 3 * xx + 2, "k-", lw=1, label="$f(x)=x^2+3x+2$")
p = r["x"][:13, 0]
ax.plot(p, p ** 2 + 3 * p + 2, "o-", color="tab:blue", ms=5, label="iterates $k=0..12$")
ax.plot(p[0], f1([p[0]]), "s", color="tab:green", ms=9, label="$x_0=10$")
ax.plot(xs1, fs1, "*", color="tab:red", ms=15, label="$x^*=-1.5$")
ax.set_xlim(-4, 11); ax.set_ylim(-5, 140)
ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$"); ax.set_title(r"Task 1.1: descent path, $\eta=0.1$")
ax.legend(); ax.grid(alpha=.3)
save(fig, "p1_task11_path.png")

# Task 1.2
etas = [0.1, 0.5, 0.6, 0.9, 1.0, 1.1]
runs = {eta: compare(f1, g1, [10.0], eta) for eta in etas}
print("Task 1.2 comparison runs (40 updates):")
for eta in etas:
    rr = runs[eta]
    print(f"  eta={eta}: q={1 - 2 * eta:+.1f}, x_40={rr['x'][-1, 0]:.6g}, "
          f"|e_40|={abs(rr['x'][-1, 0] - xs1):.3e}, f_40-f*={rr['loss'][-1] - fs1:.3e}, "
          f"updates={rr['n_updates']}, status={rr['status']}")
print("Task 1.2 convergence runs:")
for eta in etas:
    rr = converge(f1, g1, [10.0], eta)
    print(f"  eta={eta}: {summary(rr)}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
k = np.arange(41)
for eta in [0.1, 0.5, 0.6, 0.9, 1.0]:
    axs[0].plot(k, runs[eta]["x"][:, 0], "o-", ms=3, label=rf"$\eta={eta}$")
axs[0].axhline(xs1, color="k", ls="--", label="$x^*$")
axs[0].set_title("Bounded cases"); axs[0].set_xlabel("$k$"); axs[0].set_ylabel("$x_k$")
axs[0].legend(); axs[0].grid(alpha=.3)
axs[1].plot(k, runs[1.1]["x"][:, 0], "o-", ms=3, color="tab:brown", label=r"$\eta=1.1$")
axs[1].axhline(xs1, color="k", ls="--", label="$x^*$")
axs[1].set_title("Growing case"); axs[1].set_xlabel("$k$"); axs[1].set_ylabel("$x_k$")
axs[1].legend(); axs[1].grid(alpha=.3)
save(fig, "p1_task12_xk.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
styles = {0.1: ("o-", 6), 0.5: ("D", 9), 0.6: ("o-", 3), 0.9: ("x--", 6), 1.0: ("o-", 3), 1.1: ("o-", 3)}
for eta in etas:
    st, ms = styles[eta]
    ax.semilogy(k, nz(np.abs(runs[eta]["x"][:, 0] - xs1)), st, ms=ms, mfc="none" if eta == 0.1 else None,
                label=rf"$\eta={eta}$")
ax.set_xlabel("$k$"); ax.set_ylabel(r"$|x_k-x^*|$"); ax.set_title("Task 1.2: error (log scale, exact zeros omitted; $\\eta=0.1$ and $0.9$ coincide)")
ax.legend(); ax.grid(alpha=.3, which="both")
save(fig, "p1_task12_error.png")

# Task 1.3
print("Task 1.3:")
fig, ax = plt.subplots(figsize=(7, 4.5))
for x0 in [10.0, -5.0, 0.0]:
    rr = converge(f1, g1, [x0], 0.1)
    print(f"  x0={x0}: |e0|={abs(x0 - xs1)}, {summary(rr)}")
    ax.plot(rr["x"][:, 0], ".-", ms=3, label=f"$x_0={x0:g}$ ({rr['n_updates']} updates)")
ax.axhline(xs1, color="k", ls="--", label="$x^*$")
ax.set_xlim(0, 90); ax.set_xlabel("$k$"); ax.set_ylabel("$x_k$"); ax.set_title(r"Task 1.3: $\eta=0.1$")
ax.legend(); ax.grid(alpha=.3)
save(fig, "p1_task13_init.png")

# Problem 2
print("=" * 70, "\nPROBLEM 2")
f2 = lambda x: float(x[0] ** 2 + x[1] ** 2 + 3 * x[0] + 4 * x[1] + 5)
g2 = lambda x: np.array([2 * x[0] + 3, 2 * x[1] + 4])
xs2 = np.array([-1.5, -2.0])
print(f"x* = {xs2}, f(x*) = {f2(xs2)}")
H2 = np.array([[2.0, 0.0], [0.0, 2.0]])

X1g, X2g = np.meshgrid(np.linspace(-10, 7, 300), np.linspace(-10, 7, 300))
Z2 = X1g ** 2 + X2g ** 2 + 3 * X1g + 4 * X2g + 5
levels2 = np.linspace(f2(xs2), 140, 20)


def contour_axes(ax, title):
    cs = ax.contour(X1g, X2g, Z2, levels=levels2, cmap="viridis", linewidths=.8)
    ax.plot(*xs2, "*", color="tab:red", ms=15, label=r"$\mathbf{x}^*$", zorder=5)
    ax.set_xlim(-10, 7); ax.set_ylim(-10, 7); ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title)
    return cs


# Task 2.1
r = converge(f2, g2, [5.0, 5.0], 0.1)
print("Task 2.1:", summary(r))
fig, ax = plt.subplots(figsize=(6, 6))
contour_axes(ax, r"Task 2.1: $\eta=0.1$, $\mathbf{x}_0=(5,5)$")
ax.plot(r["x"][:, 0], r["x"][:, 1], "o-", ms=3, color="tab:blue", label="iterates")
ax.plot(5, 5, "s", color="tab:green", ms=9, label=r"$\mathbf{x}_0$")
ax.legend(loc="lower right")
save(fig, "p2_task21_contour.png")

# Task 2.2
print("Task 2.2:")
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
runs2 = {}
for ax, eta in zip(axs, [0.1, 0.5, 1.0]):
    rr = compare(f2, g2, [5.0, 5.0], eta)
    runs2[eta] = rr
    rc = converge(f2, g2, [5.0, 5.0], eta)
    print(f"  eta={eta}: multiplier={1 - 2 * eta:+.1f}; 40-update run: x_40={rr['x'][-1]}, "
          f"||e_40||={np.linalg.norm(rr['x'][-1] - xs2):.3e}, status={rr['status']}; "
          f"convergence run: {summary(rc)}")
    contour_axes(ax, rf"$\eta={eta}$ (40 updates)")
    ax.plot(rr["x"][:, 0], rr["x"][:, 1], "o-", ms=4, color="tab:blue", label="iterates")
    ax.plot(5, 5, "s", color="tab:green", ms=9, label=r"$\mathbf{x}_0$")
    ax.legend(loc="lower right")
save(fig, "p2_task22_contours.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
for eta in [0.1, 0.5, 1.0]:
    ax.semilogy(np.arange(41), nz(np.linalg.norm(runs2[eta]["x"] - xs2, axis=1)), "o-", ms=3,
                label=rf"$\eta={eta}$")
ax.set_xlabel("$k$"); ax.set_ylabel(r"$\|\mathbf{x}_k-\mathbf{x}^*\|_2$")
ax.set_title("Task 2.2: distance to minimizer (zeros omitted)"); ax.legend(); ax.grid(alpha=.3, which="both")
save(fig, "p2_task22_error.png")

# Task 2.3
print("Task 2.3:")
fig, ax = plt.subplots(figsize=(6, 6))
contour_axes(ax, r"Task 2.3: $\eta=0.1$, three initializations")
for x0, c in zip([(5, 5), (-5, 5), (-5, -5)], ["tab:blue", "tab:orange", "tab:purple"]):
    rr = converge(f2, g2, x0, 0.1)
    e0 = np.array(x0, float) - xs2
    print(f"  x0={x0}: ||e0||={np.linalg.norm(e0):.4f}, direction e0/||e0||={e0 / np.linalg.norm(e0)}, "
          f"{summary(rr)}")
    ax.plot(rr["x"][:, 0], rr["x"][:, 1], "o-", ms=3, color=c, label=f"$x_0={x0}$: {rr['n_updates']} upd.")
ax.legend(loc="lower right", fontsize=8)
save(fig, "p2_task23_inits.png")

# Task 2.4
print("Task 2.4: H =", H2.tolist())
for eta in [0.1, 0.5, 1.0]:
    print(f"  eta={eta}: eig(I - eta H) = {np.linalg.eigvalsh(np.eye(2) - eta * H2)}")
rr = runs2[1.0]
print("  eta=1.0 first states:", rr["x"][:4].tolist())
# collinearity check for eta = 0.1: cross product of e_k with e_0
e = runs2[0.1]["x"] - xs2
cross = e[:, 0] * e[0, 1] - e[:, 1] * e[0, 0]
print(f"  eta=0.1 max |e_k x e_0| over 40 updates = {np.max(np.abs(cross)):.2e}")


# Problem 3, Dataset 1
print("=" * 70, "\nPROBLEM 3 - DATASET 1")
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)
n = len(X)
A1 = np.column_stack([X, np.ones(n)])


def mse_loss(A, y):
    return lambda th: float(np.mean((A @ th - y) ** 2))


def mse_grad(A, y):
    # r = A th - y ;  dL/dth = (2/n) A^T r   (all coefficients updated from the same residuals)
    return lambda th: (2.0 / len(y)) * A.T @ (A @ th - y)


L1, G1 = mse_loss(A1, Y), mse_grad(A1, Y)
# explicit scalar form, used only to check the matrix form
th_test = np.array([0.7, -0.3])
res = th_test[0] * X + th_test[1] - Y
scalar_grad = np.array([2 / n * np.sum(res * X), 2 / n * np.sum(res)])
print("scalar vs matrix gradient max diff:", np.max(np.abs(scalar_grad - G1(th_test))))

H1 = 2 / n * A1.T @ A1
lam1 = np.linalg.eigvalsh(H1)
eta1 = 1 / lam1.max()
print("H =", H1.tolist(), " eigenvalues =", lam1)
print(f"eta = 1/lmax = {eta1:.10f}, bound 2/lmax = {2 / lam1.max():.10f}")
print("multipliers 1 - eta*lam:", 1 - eta1 * lam1)
r3a = converge(L1, G1, [0.0, 0.0], eta1)
print("Task 3A.1 convergence run:", summary(r3a))
r3a_bad = compare(L1, G1, [0.0, 0.0], 0.1)
print("eta=0.1 multipliers:", 1 - 0.1 * lam1)
print("Task 3A.2 eta=0.1 comparison:", summary(r3a_bad))
print("   loss at k=0,10,20,30,40:", r3a_bad["loss"][[0, 10, 20, 30, 40]])

theta_ref = np.linalg.lstsq(A1, Y, rcond=None)[0]
mse_ref = np.mean((A1 @ theta_ref - Y) ** 2)
print(f"lstsq theta_ref = {theta_ref}, mse_ref = {mse_ref:.3e}")
print(f"|theta_GD - theta_ref| = {np.abs(r3a['x'][-1] - theta_ref)}, "
      f"MSE_GD - mse_ref = {r3a['loss'][-1] - mse_ref:.3e}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(X, Y, "o", ms=8, label="observations")
xl = np.linspace(0, 6, 50)
w, b = r3a["x"][-1]
ax.plot(xl, w * xl + b, "-", label=rf"GD fit $\hat y={w:.6f}x{b:+.6f}$")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title("Task 3A.3: Dataset 1 fit"); ax.legend(); ax.grid(alpha=.3)
save(fig, "p3a_fit.png")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].semilogy(nz(r3a["loss"]), "-", label=r"$\eta=1/\lambda_{\max}$")
axs[0].set_xlabel("$k$"); axs[0].set_ylabel("training MSE"); axs[0].set_title(f"Convergent run ({r3a['n_updates']} updates)")
axs[0].grid(alpha=.3, which="both"); axs[0].legend()
axs[1].semilogy(r3a_bad["loss"], "o-", ms=3, color="tab:red", label=r"$\eta=0.1$")
axs[1].set_xlabel("$k$"); axs[1].set_ylabel("training MSE"); axs[1].set_title("40-update run, $\\eta=0.1$")
axs[1].grid(alpha=.3, which="both"); axs[1].legend()
save(fig, "p3a_loss.png")


# Problem 3, Dataset 2
print("=" * 70, "\nPROBLEM 3 - DATASET 2")
X1 = np.array([1, 2, 3, 4, 5], dtype=float)
X2 = np.array([2, 3, 4, 5, 6], dtype=float)
Y2 = np.array([2, 3, 5, 6, 8], dtype=float)
A2 = np.column_stack([X1, X2, np.ones(n)])
v = np.array([1.0, -1.0, 1.0])
print("A v =", A2 @ v, " rank(A) =", np.linalg.matrix_rank(A2))
H3 = 2 / n * A2.T @ A2
lam3 = np.linalg.eigvalsh(H3)
eta3 = 1 / lam3.max()
print("H =", H3.tolist())
print("eigenvalues of H =", lam3, "  H v =", H3 @ v)
print(f"eta = 1/lmax = {eta3:.10f}, bound 2/lmax = {2 / lam3.max():.10f}")
print("eig(I - eta H) =", np.linalg.eigvalsh(np.eye(3) - eta3 * H3))

L2, G2 = mse_loss(A2, Y2), mse_grad(A2, Y2)
rA = converge(L2, G2, [0.0, 0.0, 0.0], eta3)
rB = converge(L2, G2, [1.0, -1.0, 1.0], eta3)
print("Task 3B.2 from 0      :", summary(rA))
print("Task 3B.3 from (1,-1,1):", summary(rB))
thA, thB = rA["x"][-1], rB["x"][-1]
print("difference thB - thA =", thB - thA)
print("v.theta along run A (first, last):", v @ rA["x"][0], v @ thA,
      " max drift:", np.max(np.abs(rA["x"] @ v - v @ rA["x"][0])))
print("v.theta along run B (first, last):", v @ rB["x"][0], v @ thB,
      " max drift:", np.max(np.abs(rB["x"] @ v - v @ rB["x"][0])))
print("alpha, beta (A):", thA[0] + thA[1], thA[1] + thA[2], "  (B):", thB[0] + thB[1], thB[1] + thB[2])

th_ref2 = np.linalg.lstsq(A2, Y2, rcond=None)
print("lstsq returns: theta =", th_ref2[0], " residuals array =", th_ref2[1], " rank =", th_ref2[2])
mse_ref2 = np.mean((A2 @ th_ref2[0] - Y2) ** 2)
print(f"mse_ref = {mse_ref2:.12f}")
print("predictions ref:", A2 @ th_ref2[0])
print("predictions A  :", A2 @ thA, " max diff vs ref:", np.max(np.abs(A2 @ thA - A2 @ th_ref2[0])))
print("predictions B  :", A2 @ thB, " max diff vs ref:", np.max(np.abs(A2 @ thB - A2 @ th_ref2[0])))
print(f"MSE_A - mse_ref = {rA['loss'][-1] - mse_ref2:.3e}, MSE_B - mse_ref = {rB['loss'][-1] - mse_ref2:.3e}")
print("|thA - theta_ref| =", np.abs(thA - th_ref2[0]))
x_new = np.array([0.0, 0.0, 1.0])
print("prediction at (0,0): A ->", x_new @ thA, " B ->", x_new @ thB)

# figures for 3B.3
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
names = ["$w_1$", "$w_2$", "$b$"]
for j, c in enumerate(["tab:blue", "tab:orange", "tab:green"]):
    axs[0].plot(rA["x"][:, j], "-", color=c, label=names[j] + r" from $\theta_0$")
    axs[0].plot(rB["x"][:, j], "--", color=c, label=names[j] + r" from $\tilde\theta_0$")
axs[0].set_xscale("symlog", linthresh=10); axs[0].set_title("Parameter histories")
axs[0].set_xlabel("$k$"); axs[0].legend(fontsize=7, ncol=2); axs[0].grid(alpha=.3)
predA, predB = rA["x"] @ A2.T, rB["x"] @ A2.T
for i in range(n):
    axs[1].plot(predA[:, i], "-", color=f"C{i}", label=rf"$\hat y_{i + 1}$ ($\theta_0$)")
    axs[1].plot(predB[:, i], "--", color=f"C{i}")
axs[1].set_xscale("symlog", linthresh=10); axs[1].set_title(r"Prediction histories (solid $\theta_0$, dashed $\tilde\theta_0$)")
axs[1].set_xlabel("$k$"); axs[1].legend(fontsize=7); axs[1].grid(alpha=.3)
axs[2].semilogy(rA["loss"] - mse_ref2 + 1e-18, "-", label=r"from $\theta_0=(0,0,0)$")
axs[2].semilogy(rB["loss"] - mse_ref2 + 1e-18, "--", label=r"from $\tilde\theta_0=(1,-1,1)$")
axs[2].set_title(r"Loss gap $L(\theta_k)-L_{\rm ref}$"); axs[2].set_xlabel("$k$"); axs[2].legend(); axs[2].grid(alpha=.3, which="both")
save(fig, "p3b_histories.png")

# figure for 3B.4
fig = plt.figure(figsize=(13, 5.5))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
g1_, g2_ = np.meshgrid(np.linspace(-0.5, 6, 20), np.linspace(-0.5, 7, 20))
ax3.plot_surface(g1_, g2_, thA[0] * g1_ + thA[1] * g2_ + thA[2], alpha=.35, color="tab:blue")
ax3.scatter(X1, X2, Y2, color="tab:red", s=40, depthshade=False, label="observations")
ll = np.linspace(-0.5, 6, 2)
ax3.plot(ll, ll + 1, 0 * ll, "k--", label="$x_2=x_1+1$ (input line)")
ax3.plot(ll, ll + 1, thA[0] * ll + thA[1] * (ll + 1) + thA[2], "r-", lw=2, label="fit along data line")
ax3.scatter([0], [0], [thA[2]], color="tab:blue", marker="^", s=60, label=r"$(0,0)$: plane from $\theta_0$")
ax3.scatter([0], [0], [thB[2]], color="tab:purple", marker="v", s=60, label=r"$(0,0)$: model from $\tilde\theta_0$")
ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_zlabel("$y$")
ax3.set_title(r"Fitted plane (from $\theta_0=0$)"); ax3.legend(fontsize=7, loc="upper left")
ax3.view_init(elev=20, azim=-60)
ax = fig.add_subplot(1, 2, 2)
ax.plot(X1, Y2, "o", ms=8, label="observed $y$")
ax.plot(X1, A2 @ thA, "s-", label=r"fitted, $\theta_0=(0,0,0)$")
ax.plot(X1, A2 @ thB, "x--", ms=10, label=r"fitted, $\tilde\theta_0=(1,-1,1)$")
ax.set_xlabel("$x_1$ (with $x_2=x_1+1$)"); ax.set_ylabel("$y$"); ax.set_title("Observed vs fitted outputs")
ax.legend(); ax.grid(alpha=.3)
save(fig, "p3b_fit.png")

# Optional extension E3
print("=" * 70, "\nE3")
P_v = np.outer(v, v) / (v @ v)
for name, rr in [("A", rA), ("B", rB)]:
    th_star_proj = rr["x"] - (rr["x"] @ v)[:, None] * v / (v @ v)  # component orthogonal to v
    # distance to the solution set {theta_ref + t v}
    d = np.linalg.norm((rr["x"] - th_ref2[0]) @ (np.eye(3) - P_v), axis=1)
    print(f"run {name}: dist to minimizer set at k=0: {d[0]:.4f}, final: {d[-1]:.3e}")
print("predicted limit from (1,-1,1): theta_ref + (v.theta0/|v|^2) v =", th_ref2[0] + (v @ np.array([1, -1, 1.0])) / (v @ v) * v)
