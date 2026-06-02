import numpy as np
import matplotlib.pyplot as plt


# get d'Alemberts exact solution of the 1D wave equation
def get_exact(tt: np.ndarray, x: np.ndarray) -> np.ndarray:
    u = np.zeros((tt.size, x.size))  # solution vector
    ii = [1, 2, 3, 4, 5]

    # solve for all t
    for t_idx, t in enumerate(tt):
        # compute harmonics
        for i in ii:
            k = i * np.pi
            u[t_idx] += np.cos((x - t + 0.5) * k)  # leftward
            u[t_idx] += np.cos(
                (x + t - 0.5) * k
            )  # rightward NOTE: they specify + 0.5 but plot - 0.5 for this part

    u /= 10  # 1/10 prefactor from the paper
    return u


def plot_1d(
    uu: np.ndarray, xx: np.ndarray, tt: np.ndarray, title: str = "Exact solution"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    umax = max(abs(np.max(uu)), abs(np.min(uu)))

    im = ax.imshow(
        uu,
        extent=(xx.min(), xx.max(), tt.max(), tt.min()),  # [xmin, xmax, tmax, tmin]
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


if __name__ == "__main__":
    print("Geting ODIL reference solution")
    # ODIL reference plot: 25 x 25 grid
    t = np.linspace(0, 1, 25)
    x = np.linspace(-1, 1, 25)

    u = get_exact(t, x)
    plot_1d(u, x, t, title="ODIL reference")
