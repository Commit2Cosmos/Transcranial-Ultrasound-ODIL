import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from IPython.display import clear_output
import time
from dataclasses import dataclass


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # set the device


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_exact(tt: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
    u = torch.zeros_like(tt)
    for k in range(1, 6):
        u += torch.cos((xx - tt + 0.5) * k * torch.pi)
        u += torch.cos((xx + tt - 0.5) * k * torch.pi)
    return u / 10.0


# time derivative of reference solution
# removes the need for autograd in the velocity bc
def get_exact_ut(tt: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
    u_t = torch.zeros_like(tt)
    for k in range(1, 6):
        u_t += k * torch.pi * torch.sin((xx - tt + 0.5) * k * torch.pi)
        u_t -= k * torch.pi * torch.sin((xx + tt - 0.5) * k * torch.pi)
    return u_t / 10.0


class PINN(nn.Module):
    def __init__(self, n_layers=2, n_neurons=25):
        super().__init__()
        activation = nn.Tanh  # tanh activation function as specified

        layers = [nn.Linear(2, n_neurons), activation()]  # input layer

        # build layers
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]

        layers += [nn.Linear(n_neurons, 1)]  # output layer

        self.net = nn.Sequential(*layers)  # sequential object to encapsulate layers

    def forward(self, t, x):
        tx = torch.cat([t, x], dim=1)  # shape: (N, 2)
        return self.net(tx)  # shape: (N, 1)


def collocation_points(
    device, t_0=0.0, t_N=1.0, x_0=-1.0, x_I=1.0, n_interior=8192, n_boundary=768
):
    n_each = n_boundary // 4

    # define random interior collocation points
    tx_physics = torch.rand(
        n_interior, 2, device=device
    )  # generate random (t, x) points of shape (n_interior, 2)

    # sample and scale to domain limits
    tx_physics[:, 0] = tx_physics[:, 0] * (t_N - t_0) + t_0  # t in [t_0, t_N]
    tx_physics[:, 1] = tx_physics[:, 1] * (x_I - x_0) + x_0  # x in [x_0, x_I]
    tx_physics.requires_grad_(True)  # required for autograd

    # decompose into t, x
    t_physics = tx_physics[:, 0:1]
    x_physics = tx_physics[:, 1:2]

    # define boundary/initial collocation points
    # u(t=0, x)
    x_ic = torch.rand(n_each, 1, device=device) * (x_I - x_0) + x_0
    t_ic = torch.zeros(n_each, 1, device=device)

    # u(t, x=-1)
    t_left = torch.rand(n_each, 1, device=device) * (t_N - t_0) + t_0
    x_left = torch.full((n_each, 1), x_0, device=device)  # x_0 = -1.0

    # u(t, x=1.0)
    t_right = torch.rand(n_each, 1, device=device) * (t_N - t_0) + t_0
    x_right = torch.full((n_each, 1), x_I, device=device)

    # u_t(t=0, x)
    x_vel = torch.rand(n_each, 1, device=device) * (x_I - x_0) + x_0
    t_vel = torch.zeros(n_each, 1, device=device).requires_grad_(
        True
    )  # needs grad for u_t

    return (
        t_physics,
        x_physics,
        t_ic,
        x_ic,
        t_left,
        x_left,
        t_right,
        x_right,
        t_vel,
        x_vel,
    )


# boundary/initial losses
# u(t=0, x)
def loss_ic(model, t_ic, x_ic):
    u = model(t_ic, x_ic)
    return torch.mean((u - get_exact(t_ic, x_ic)) ** 2)


# u(t, x=-1)
def loss_left(model, t_left, x_left):
    u = model(t_left, x_left)
    return torch.mean((u - get_exact(t_left, x_left)) ** 2)


# u(t, x=L)
def loss_right(model, t_right, x_right):
    u = model(t_right, x_right)
    return torch.mean((u - get_exact(t_right, x_right)) ** 2)


# u_t(t=0, x)
def loss_velocity(model, t_vel, x_vel):
    u = model(t_vel, x_vel)
    dudt = torch.autograd.grad(u, t_vel, torch.ones_like(u), create_graph=True)[0]
    u_t_ref = get_exact_ut(t_vel, x_vel)
    return torch.mean((dudt - u_t_ref) ** 2)


# physics loss: the PDE residual
# uses autograd to compute derivatives of network output u
def loss_pde(
    model: nn.Module, t_physics: torch.Tensor, x_physics: torch.Tensor
) -> torch.Tensor:
    # compute output
    u = model(t_physics, x_physics)

    # time
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(
        dudt, t_physics, torch.ones_like(dudt), create_graph=True
    )[0]

    # space
    dudx = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(
        dudx, x_physics, torch.ones_like(dudx), create_graph=True
    )[0]

    # compute loss
    return torch.mean((d2udt2 - d2udx2) ** 2)


def training_plot(
    device: torch.device,
    model: nn.Module,
    xx: torch.Tensor,
    tt: torch.Tensor,
    step: int,
) -> None:
    TT, XX = torch.meshgrid(tt.squeeze(), xx.squeeze(), indexing="ij")
    t_flat = TT.reshape(-1, 1).to(device)
    x_flat = XX.reshape(-1, 1).to(device)

    with torch.no_grad():
        u = model(t_flat, x_flat).detach().squeeze().cpu().reshape(len(tt), len(xx))
    uu_exact = get_exact(TT, XX).detach().cpu()

    error = torch.abs(u - uu_exact) / uu_exact.std()
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    umax = u.abs().max().item()
    
    im0 = axs[0].imshow(
        u,
        extent=[xx.min().item(), xx.max().item(), tt.max().item(), tt.min().item()],
        cmap="RdBu_r",
        vmin=-umax,
        vmax=umax,
        aspect=4,
    )
    im1 = axs[1].imshow(
        error,
        extent=[xx.min().item(), xx.max().item(), tt.max().item(), tt.min().item()],
        cmap="RdBu_r",
        aspect=4,
    )

    plt.colorbar(im0, ax=axs[0], label=r"$\mathcal{N}(t, x, \mathbf{\theta})$")
    plt.colorbar(
        im1,
        ax=axs[1],
        label=r"$\mathcal{N}(t, x, \mathbf{\theta}) - u_\text{exact}(t, x)$",
    )

    for ax in axs:
        ax.set_xlabel("x")
    axs[0].set_ylabel("t")
    axs[0].set_title("PINN prediction")
    axs[1].set_title("Relative L1 test error")

    fig.suptitle(f"Epoch {step + 1}, L2 error: {error.mean().item():.2%}")
    plt.tight_layout()


def plot_losses(losses: dict):
    if len(losses["L"]) == 0:
        raise BufferError(
            "Detected history of length 0. Please train before trying to plot."
        )
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    titles = [
        r"$L(u_i^n)$",
        r"$\|\mathcal{R}_\mathrm{PDE}\|$",
        r"$\|\mathcal{R}_\mathrm{IC}\|$",
        r"$\|\mathcal{R}_\mathrm{left}\|$",
        r"$\|\mathcal{R}_\mathrm{right}\|$",
        r"$\|\mathcal{R}_\mathrm{vel}\|$",
    ]

    for ax, t, key in zip(axs.ravel(), titles, losses.keys()):
        data = losses[key]
        ax.semilogy(data)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title(t + f", current = {data[-1]:.3e}")

    fig.suptitle("Training history per component")
    fig.tight_layout()


@dataclass
class PINNConfig:
    seed: int = 7
    device: torch.device = get_device()
    layers: int = 2
    neurons: int = 25
    lr: float = 0.1
    epochs: int = 100
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    left_bc_weight: float = 1.0
    right_bc_weight: float = 1.0
    velocity_bc_weight: float = 1.0
    t_0: float = 0.0
    t_N: float = 1.0
    x_0: float = -1.0
    x_I: float = 1.0

    def show(self):
        print("PINN Configuration:")
        print(f"Layers: {self.layers}. Neurons per layer: {self.neurons}.")
        print(f"Learning rate: {self.lr}. Epochs: {self.epochs}.")
        print(
            f"PDE weight: {self.pde_weight}.\nIC weight: {self.ic_weight}. "
            + f"\nLeft BC weight: {self.left_bc_weight}."
            + f"\nRight BC weight: {self.right_bc_weight}. "
            + f"\nVelocity BC weight: {self.velocity_bc_weight}."
        )
        print(f"Seed: {self.seed}. Device: {self.device}.")


def train(config: PINNConfig = PINNConfig()) -> PINN:
    set_seed(config.seed)
    device = config.device

    pinn = PINN(n_layers=config.layers, n_neurons=config.neurons).to(device)
    t_physics, x_physics, t_ic, x_ic, t_left, x_left, t_right, x_right, t_vel, x_vel = (
        collocation_points(device, config.t_0, config.t_N, config.x_0, config.x_I)
    )

    t_test = torch.linspace(config.t_0, config.t_N, 100).to(device)
    x_test = torch.linspace(config.x_0, config.x_I, 100).to(device)

    lbfgs = torch.optim.LBFGS(
        pinn.parameters(),
        lr=config.lr,
        max_iter=250,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    history = {"L": [], "PDE": [], "IC": [], "BC-Left": [], "BC-Right": [], "Vel": []}

    def closure():
        lbfgs.zero_grad()
        ic_loss = loss_ic(pinn, t_ic, x_ic)
        left_loss = loss_left(pinn, t_left, x_left)
        right_loss = loss_right(pinn, t_right, x_right)
        velocity_loss = loss_velocity(pinn, t_vel, x_vel)
        pde_loss = loss_pde(pinn, t_physics, x_physics)
        loss = (
            config.ic_weight * ic_loss
            + config.left_bc_weight * left_loss
            + config.right_bc_weight * right_loss
            + config.velocity_bc_weight * velocity_loss
            + config.pde_weight * pde_loss
        )
        loss.backward()
        return loss

    start = time.time()
    for i in range(config.epochs):
        loss = lbfgs.step(closure)
        history["L"].append(loss.item())

        with torch.enable_grad():
            ic_loss = loss_ic(pinn, t_ic, x_ic)
            left_loss = loss_left(pinn, t_left, x_left)
            right_loss = loss_right(pinn, t_right, x_right)
            vel_loss = loss_velocity(pinn, t_vel, x_vel)
            pde_loss = loss_pde(pinn, t_physics, x_physics)
            history["IC"].append(ic_loss.item())
            history["BC-Left"].append(left_loss.item())
            history["BC-Right"].append(right_loss.item())
            history["Vel"].append(vel_loss.item())
            history["PDE"].append(pde_loss.item())

        if ((i + 1) % 10 == 0) or (i == 0):
            clear_output(wait=True)
            training_plot(device, pinn, x_test, t_test, i)
            plot_losses(history)
            plt.show()

            print(f"Elapsed time: {time.time() - start:.1f}s")

    return pinn


if __name__ == "__main__":
    device = get_device()
    set_seed()

    # domain parameters

    model = PINN()
    summary(
        model.net,
        input_size=(1, 2),
        col_names=["input_size", "output_size", "num_params"],
    )

    # network parameters
    default = PINNConfig()

    print("Training...")
    train(default)
