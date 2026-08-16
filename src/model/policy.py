import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, dropout: float = 0.0):
        super().__init__()
        # Keep the no-dropout structure byte-identical to the original so
        # checkpoints saved before dropout was added still load cleanly.
        if dropout > 0:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, act_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, act_dim)
            )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_log_prob(self, obs, act):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(act).sum(dim=-1)


class VNetwork(nn.Module):
    """State-only value network for preference-based learning.

    Used by the V-AWR pipeline (scripts/04f_train_v_awr.py) where the
    safety signal is encoded as state occupancy rather than action choice.
    Bradley-Terry over summed V values per segment teaches V which states
    appear in preferred (safe) vs dispreferred (unsafe) trajectories;
    AWR then turns that V into an action policy via one-step advantages.
    """

    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class VEnsemble(nn.Module):
    """K independent V networks averaged at inference.

    Addresses BT-loss underspecification: the loss

        L = -log sigmoid(sum_t V(s_t^+) - sum_t V(s_t^-))

    only constrains segment SUMS of V, not per-state V values. Each member
    of the ensemble has a different random init and converges to a
    different point in the null space of zero-sum-per-segment
    perturbations. Averaging cancels out idiosyncratic per-state errors
    while preserving the BT-supervised segment-sum signal.

    Forward returns the mean V across members, so this class is a drop-in
    replacement for VNetwork in AWR call sites that read V(s).
    """

    def __init__(self, obs_dim, hidden_dim=256, K=3):
        super().__init__()
        self.K = int(K)
        # nn.ModuleList(VNetwork) — each VNetwork constructor advances the
        # global RNG so members get distinct random inits.
        self.members = nn.ModuleList(
            [VNetwork(obs_dim, hidden_dim) for _ in range(self.K)]
        )

    def forward(self, obs):
        # Mean V across ensemble. Shape: same as a single VNetwork forward.
        return torch.stack([m(obs) for m in self.members], dim=0).mean(dim=0)

    def forward_all(self, obs):
        # Return per-member outputs; shape [K, *batch_shape].
        return torch.stack([m(obs) for m in self.members], dim=0)


class QNetwork(nn.Module):
    """State-action value network. Used by the V-IQL pipeline
    (scripts/04g_train_v_iql.py) to bootstrap V_θ's per-state preference
    value into a long-horizon action-conditioned critic via offline TD.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class QEnsemble(nn.Module):
    """K independent Q networks. K-coupled IQL pipeline (Phase 13): each Q_k
    is bootstrapped with its own V_theta_k as reward and its own V_psi_k as
    next-state baseline. This propagates BT-underdetermination uncertainty
    (from V_theta) through the Q estimate AND adds Q's own bootstrap
    variance — both surface in std_k(Q_k(s,a)).

    forward() returns the ensemble mean (drop-in for QNetwork); forward_all
    returns per-member [K, batch] for disagreement-pessimism advantage:
        A_pess(s, a) = mean_k Q_k(s, a) - λ * std_k Q_k(s, a) - V_psi_mean(s)
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256, K=3):
        super().__init__()
        self.K = int(K)
        self.members = nn.ModuleList(
            [QNetwork(obs_dim, act_dim, hidden_dim) for _ in range(self.K)]
        )

    def forward(self, obs, act):
        return torch.stack([m(obs, act) for m in self.members], dim=0).mean(dim=0)

    def forward_all(self, obs, act):
        return torch.stack([m(obs, act) for m in self.members], dim=0)

class DynamicsEnsemble(nn.Module):
    """K_dyn one-step dynamics models f(s, a) -> delta_s (Phase B'.3).

    One-step ONLY, by design: Phase 3 showed long-horizon optimization of a
    safety-only value collapses to stationarity, and one-step myopia is
    protective. A one-step model avoids compounding rollout error entirely
    while unlocking the capability AWR lacks: evaluating actions the dataset
    did not take, via V(s + f(s, a)).

    Ensemble disagreement (std over members of the predicted next state)
    doubles as an out-of-distribution signal for candidate actions: OOD
    (s, a) pairs get high disagreement -> pessimized advantage -> the
    counterfactual weight falls back toward the data action.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256, K=5):
        super().__init__()
        self.K = int(K)
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            ) for _ in range(self.K)
        ])

    def forward_all(self, obs, act):
        """Predicted next states per member: [K, *batch, obs_dim]."""
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([obs + m(x) for m in self.members], dim=0)

    def forward(self, obs, act):
        """Mean predicted next state."""
        return self.forward_all(obs, act).mean(dim=0)
