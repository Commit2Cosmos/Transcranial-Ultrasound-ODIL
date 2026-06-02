from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch  # type: ignore[reportMissingImports]
import scipy.optimize as scopt
import time


# this will store the data of the variables on our grid
@dataclass
class Wavefield:
    # spatial data
    x_0: float = -1.0
    x_I: float = 1.0
    I: int = 25
    L: float = field(init=False)
    dx: float = field(init=False)

    # temporal data
    t_0: float = 0.0
    t_N: float = 1.0
    N: int = 25
    T: float = field(init=False)
    dt: float = field(init=False)

    # actual data
    init: np.ndarray | None = None  # data to initialise with
    _data: np.ndarray = field(init=False)  # internal data
    _shape: Tuple[int, int] = field(init=False)  # length of domain object

    def __post_init__(self):
        # compute lengths and spacings
        self.L = self.x_I - self.x_0
        self.dx = self.L / (self.I - 1)

        self.T = self.t_N - self.t_0
        self.dt = self.T / (self.N - 1)

        self._shape = (self.N, self.I)  # t, x
        if self.init is None:
            self._data = np.zeros(shape=(self._shape), dtype=float).flatten()
        else:
            self._data = np.asarray(self.init, dtype=float).flatten()

    # convention: space index i, time index n
    def __getitem__(self, key):
        i, n = key
        return self._data[i + self.I * n]

    def __setitem__(self, key, value):
        i, n = key
        self._data[i + self.I * n] = value

    def __sub__(self, other):
        return self._data - other._data

    def __pow__(self, other):
        return self._data**other

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    def show(self, title: str = "Exact solution") -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        umax = np.abs(self._data).max()
        im = ax.imshow(
            self._data.reshape(self.N, self.I),
            extent=(self.x_0, self.x_I, self.t_N, self.t_0),  # [xmin, xmax, tmax, tmin]
            cmap="RdBu_r",
            vmin=-umax,
            vmax=umax,  # symmetric since u is bounded in [-0.5, 0.5]
            aspect=4,
        )

        plt.colorbar(im, ax=ax, label="u(x, t)")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


class DiscretePDELoss:
    def __init__(self, u: Wavefield):
        self.u = u
        self.history = {
            "R_pde": [],
            "R_ic": [],
            "R_vel": [],
            "R_left": [],
            "R_right": [],
            "L": [],
        }

    # evaluate full loss and gradient
    def evaluate(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        p = torch.tensor(params, requires_grad=True, dtype=torch.float64)
        r = self._residuals(p)
        L = (r**2).sum()
        L.backward()
        # p.grad can be None in some static-analysis scenarios
        grad = p.grad if p.grad is not None else torch.zeros_like(p)
        return L.item(), grad.numpy()

    def _residuals(self, p_flat: torch.Tensor) -> torch.Tensor:
        r = self._compute_residuals(p_flat)
        self._log(r)
        return r

    def _compute_residuals(self, p_flat: torch.Tensor) -> torch.Tensor:
        p = p_flat.reshape(self.u.N, self.u.I)

        # PDE residual
        utt = (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / self.u.dt**2
        uxx = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / self.u.dx**2
        R_pde = (utt - uxx).ravel()

        # boundary conditions
        xs = torch.linspace(self.u.x_0, self.u.x_I, self.u.I, dtype=torch.float64)
        ts = torch.linspace(self.u.t_0, self.u.t_N, self.u.N, dtype=torch.float64)

        # allocate for BCs
        g_ic = torch.zeros(self.u.I, dtype=torch.float64)
        g_vel = torch.zeros(self.u.I, dtype=torch.float64)
        g_left = torch.zeros(self.u.N, dtype=torch.float64)
        g_right = torch.zeros(self.u.N, dtype=torch.float64)

        # evaluate exact soln
        for k in range(1, 6):
            kpi = k * torch.pi
            g_ic += 0.1 * (torch.cos((xs + 0.5) * kpi) + torch.cos((xs - 0.5) * kpi))
            g_vel += (
                0.1 * kpi * (torch.sin((xs + 0.5) * kpi) - torch.sin((xs - 0.5) * kpi))
            )  # u_t(x,0) = 0
            g_left += 0.1 * (
                torch.cos((self.u.x_0 - ts + 0.5) * kpi)
                + torch.cos((self.u.x_0 + ts - 0.5) * kpi)
            )
            g_right += 0.1 * (
                torch.cos((self.u.x_I - ts + 0.5) * kpi)
                + torch.cos((self.u.x_I + ts - 0.5) * kpi)
            )

        # compute residuals
        R_ic = p[0, :] - g_ic
        R_vel = (p[1, :] - p[0, :]) / self.u.dt - g_vel
        R_left = p[:, 0] - g_left
        R_right = p[:, -1] - g_right

        return torch.cat([R_pde, R_ic, R_vel, R_left, R_right])

    def _log(self, r: torch.Tensor) -> None:
        n_pde = (self.u.N - 2) * (self.u.I - 2)  # only get interiors
        n_ic = self.u.I
        n_vel = self.u.I
        n_left = self.u.N
        n_right = self.u.N

        idx = 0
        self.history["R_pde"].append(r[idx : idx + n_pde].norm().item())
        idx += n_pde
        self.history["R_ic"].append(r[idx : idx + n_ic].norm().item())
        idx += n_ic
        self.history["R_vel"].append(r[idx : idx + n_vel].norm().item())
        idx += n_vel
        self.history["R_left"].append(r[idx : idx + n_left].norm().item())
        idx += n_left
        self.history["R_right"].append(r[idx : idx + n_right].norm().item())
        self.history["L"].append((r**2).sum().item())

    # plot residual history
    def plot_history(self, title: str = "Residual history"):
        if len(self.history["R_pde"]) == 0:
            raise BufferError(
                "Detected history of length 0. Please optimise before trying to plot."
            )
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        titles = [
            r"$\|\mathcal{R}_\mathrm{PDE}\|^2$",
            r"$\|\mathcal{R}_\mathrm{ic}\|$",
            r"$\|\mathcal{R}_\mathrm{vel}\|$",
            r"$\|\mathcal{R}_\mathrm{left}\|$",
            r"$\|\mathcal{R}_\mathrm{right}\|$",
            r"$L(u_i^n)$",
        ]

        for ax, t, key in zip(axs.ravel(), titles, self.history.keys()):
            data = self.history[key]
            if not data:
                ax.set_visible(False)
                continue
            ax.plot(np.arange(len(data)), data)  # iters, data
            ax.set_xlabel("Function evaluation")
            ax.set_title(t)
            ax.set_yscale("log")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


def run_odil_newton(u, functional):
    x = torch.tensor(u._data.copy(), dtype=torch.float64)

    J = torch.func.jacfwd(functional._residuals)(x)  # compute once since constant

    for step in range(10):
        r = functional._compute_residuals(x)  # compute residuals
        functional._log(r)
        delta = torch.linalg.lstsq(
            J, -r
        ).solution  # solve least squares problem, equivalent to newton
        x = x + delta  # update
        print(f"Step {step}: |r| = {r.norm().item():.2e}")
        if delta.norm() < 1e-12:
            break

    return x.detach().numpy()


def run_odil(
    init_values="zeros",
    method="L-BFGS-B",
    field_plot_title="ODIL",
    history_plot_title="Residual history",
    maxiter=50000,
    ftol=1e-12,
    gtol=1e-8,
) -> Wavefield:
    # setup
    if init_values == "zeros":
        u = Wavefield()  # instantiate with zeros
    elif init_values == "rand":
        u = Wavefield(init=np.random.rand(25, 25))
    elif isinstance(init_values, np.ndarray):
        u = Wavefield(init=init_values)
    else:
        raise ValueError(
            "init_values must be one of 'zeros', 'rand', or np.ndarray, got",
            type(init_values),
        )

    functional = DiscretePDELoss(u)

    if method in ["L-BFGS-B", "Newton-CG"]:
        opts = (
            {"maxiter": maxiter}
            if method == "Newton-CG"
            else {
                "maxiter": maxiter,
                "maxfun": maxiter * 200,
                "ftol": ftol,
                "gtol": gtol,
            }
        )

        start = time.time()
        result = scopt.minimize(
            functional.evaluate,
            x0=u._data.copy(),
            jac=True,  # evaluate returns loss, grad together
            method=method,
            options=opts,
        )
        end = time.time()

        convergence = result["success"]
        print(
            f"Convergence: {' achieved' if convergence else ' failed'}",
        )
        print("Exit reason:", result.message)
        if convergence:
            print("Iterations:", result.nit)
        print("Wall time:", end - start)

        u._data = result.x
        u.show(field_plot_title)

    elif method == "Newton":
        result = run_odil_newton(u, functional)
        u._data = result
        u.show(field_plot_title)

    functional.plot_history(history_plot_title)

    return u


if __name__ == "__main__":
    print("Running ODIL with Newton's method")
    run_odil(
        init_values="zeros",
        method="Newton",
        field_plot_title="ODIL (Newton's method)",
        history_plot_title="Residual history, Newton",
    )
