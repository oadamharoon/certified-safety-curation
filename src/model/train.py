"""BC pre-training and CPL training loops with optional wandb logging."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _wlog(run, payload: dict) -> None:
    if run is None:
        return
    try:
        import wandb
        wandb.log(payload)
    except Exception:
        pass


def bc_pretrain(
    policy,
    obs: torch.Tensor,
    act: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    ds = TensorDataset(obs, act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        running = 0.0
        for bo, ba in loader:
            bo, ba = bo.to(device), ba.to(device)
            log_p = policy.get_log_prob(bo, ba)
            loss = -log_p.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        avg = running / len(loader)
        _wlog(wandb_run, {"bc/epoch": epoch, "bc/loss": avg})
        if epoch % log_every == 0 or epoch == 1:
            print(f"  BC epoch {epoch:3d}/{epochs} | loss = {avg:.4f}")


def train_cpl(
    policy,
    pref_obs_A: torch.Tensor, pref_act_A: torch.Tensor,
    pref_obs_B: torch.Tensor, pref_act_B: torch.Tensor,
    pref_labels: torch.Tensor,
    bc_obs: torch.Tensor, bc_act: torch.Tensor,
    epochs: int, batch_size: int, lr: float,
    lambda_bc: float, temp: float, device,
    log_every: int = 5,
    wandb_run=None,
    conservative_bias: float = 1.0,
) -> None:
    """CPL = Bradley-Terry CE on summed log-probs of preferred vs dispreferred
    segments, plus a small BC auxiliary loss on the offline data.

    conservative_bias is CPL's lambda in (0, 1]: the preferred segment's
    score is scaled by lambda (Hejna et al., Eq. 4); 1.0 recovers the
    symmetric variant.
    """
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = pref_labels.size(0)
    bc_loader = DataLoader(TensorDataset(bc_obs, bc_act),
                           batch_size=batch_size, shuffle=True)
    bc_iter = iter(bc_loader)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        run_cpl = run_bc = run_acc = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            oA = pref_obs_A[idx].to(device)
            aA = pref_act_A[idx].to(device)
            oB = pref_obs_B[idx].to(device)
            aB = pref_act_B[idx].to(device)
            y = pref_labels[idx].to(device)

            B, T, _ = oA.shape
            logp_A = policy.get_log_prob(
                oA.reshape(B * T, -1), aA.reshape(B * T, -1)
            ).view(B, T).sum(dim=1) / temp
            logp_B = policy.get_log_prob(
                oB.reshape(B * T, -1), aB.reshape(B * T, -1)
            ).view(B, T).sum(dim=1) / temp

            logits = torch.stack([logp_A, logp_B], dim=1)  # [B, 2]
            if conservative_bias != 1.0:
                # scale the PREFERRED segment's score by lambda (CPL Eq. 4)
                sel = F.one_hot(y, num_classes=2).float()
                logits = logits * (1.0 + (conservative_bias - 1.0) * sel)
            loss_cpl = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean()

            try:
                bo, ba = next(bc_iter)
            except StopIteration:
                bc_iter = iter(bc_loader)
                bo, ba = next(bc_iter)
            bo, ba = bo.to(device), ba.to(device)
            loss_bc = -policy.get_log_prob(bo, ba).mean()
            loss = loss_cpl + lambda_bc * loss_bc

            opt.zero_grad(); loss.backward(); opt.step()

            run_cpl += loss_cpl.item()
            run_bc += loss_bc.item()
            run_acc += acc.item()
            n_batches += 1

        avg_cpl = run_cpl / max(1, n_batches)
        avg_bc = run_bc / max(1, n_batches)
        avg_acc = run_acc / max(1, n_batches)
        _wlog(wandb_run, {
            "cpl/epoch": epoch,
            "cpl/loss_cpl": avg_cpl,
            "cpl/loss_bc": avg_bc,
            "cpl/pref_accuracy": avg_acc,
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  CPL epoch {epoch:3d}/{epochs} | "
                  f"L_cpl={avg_cpl:.4f}  L_bc={avg_bc:.4f}  acc={avg_acc:.3f}")


def train_v_preference(
    v_net,
    pref_obs_A: torch.Tensor, pref_obs_B: torch.Tensor,
    pref_labels: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Bradley-Terry loss on summed state values per segment.

    L = CE([V_A_sum, V_B_sum], preference_label)

    where V_A_sum = Sum_t V(s_t^A) over the 30-step segment A. Actions are
    NOT used — this is the diagnostic for "safety is state-occupancy."
    """
    opt = torch.optim.Adam(v_net.parameters(), lr=lr)
    n = pref_labels.size(0)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        run_loss = run_acc = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            oA = pref_obs_A[idx].to(device)
            oB = pref_obs_B[idx].to(device)
            y = pref_labels[idx].to(device)

            B, T, _ = oA.shape
            vA = v_net(oA.reshape(B * T, -1)).view(B, T).sum(dim=1)
            vB = v_net(oB.reshape(B * T, -1)).view(B, T).sum(dim=1)
            logits = torch.stack([vA, vB], dim=1)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); run_acc += acc.item(); n_batches += 1

        avg_loss = run_loss / max(1, n_batches)
        avg_acc = run_acc / max(1, n_batches)
        _wlog(wandb_run, {
            "v/epoch": epoch, "v/loss": avg_loss, "v/pref_accuracy": avg_acc,
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  V epoch {epoch:3d}/{epochs} | "
                  f"loss={avg_loss:.4f}  acc={avg_acc:.3f}")


def train_v_supervised(
    v_net,
    seg_obs: torch.Tensor, seg_target: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Direct regression of per-segment cumulative cost onto state-sum-V.

    Phase 10c baseline. Trains V_supervised so that Sum_t V(s_t) over a
    segment matches the segment's target value, where target = -total_cost
    (negative so high V = safer, matching V_preference's convention).

    If V_supervised in AWR matches or beats V_preference, the preference
    pipeline isn't adding value over a direct cost signal — we'd be
    measuring the preference bottleneck, not the safety insight.

    seg_obs:    shape (n_segments, T, obs_dim)
    seg_target: shape (n_segments,) -- typically -total_cost
    """
    opt = torch.optim.Adam(v_net.parameters(), lr=lr)
    n = seg_target.size(0)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        run_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bo = seg_obs[idx].to(device)        # (B, T, D)
            bt = seg_target[idx].to(device)     # (B,)
            B, T, _ = bo.shape
            v_sum = v_net(bo.reshape(B * T, -1)).view(B, T).sum(dim=1)
            loss = F.mse_loss(v_sum, bt)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); n_batches += 1

        avg = run_loss / max(1, n_batches)
        _wlog(wandb_run, {"v_sup/epoch": epoch, "v_sup/loss": avg})
        if epoch % log_every == 0 or epoch == 1:
            print(f"  V-sup epoch {epoch:3d}/{epochs} | mse={avg:.4f}")


def train_v_ensemble(
    ensemble,
    pref_obs_A: torch.Tensor, pref_obs_B: torch.Tensor,
    pref_labels: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device,
    weight_decay: float = 0.0,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Train K independent V members of a VEnsemble on the same preference
    data. Each member has its own optimizer; gradients don't cross between
    members, so they explore different points in the BT loss's null space.

    weight_decay > 0 directly regularizes V_theta toward zero, constraining
    the BT-loss null space and reducing inter-member drift (the same effect
    that ensemble averaging produces, but per-member).
    """
    K = ensemble.K
    opts = [torch.optim.Adam(ensemble.members[k].parameters(), lr=lr,
                             weight_decay=weight_decay)
            for k in range(K)]
    n = pref_labels.size(0)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        run_loss = [0.0] * K
        run_acc = [0.0] * K
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            oA = pref_obs_A[idx].to(device)
            oB = pref_obs_B[idx].to(device)
            y = pref_labels[idx].to(device)
            B, T, _ = oA.shape
            for k in range(K):
                m = ensemble.members[k]
                vA = m(oA.reshape(B * T, -1)).view(B, T).sum(dim=1)
                vB = m(oB.reshape(B * T, -1)).view(B, T).sum(dim=1)
                logits = torch.stack([vA, vB], dim=1)
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                opts[k].zero_grad(); loss.backward(); opts[k].step()
                run_loss[k] += loss.item(); run_acc[k] += acc.item()
            n_batches += 1

        if epoch % log_every == 0 or epoch == 1:
            losses = [run_loss[k] / max(1, n_batches) for k in range(K)]
            accs   = [run_acc[k]  / max(1, n_batches) for k in range(K)]
            print(f"  V-ens epoch {epoch:3d}/{epochs} | "
                  f"losses=[{', '.join(f'{l:.4f}' for l in losses)}] | "
                  f"accs=[{', '.join(f'{a:.3f}' for a in accs)}]")
            _wlog(wandb_run, {
                "v_ens/epoch": epoch,
                **{f"v_ens/loss_k{k}": losses[k] for k in range(K)},
                **{f"v_ens/acc_k{k}":  accs[k]  for k in range(K)},
                "v_ens/loss_mean": sum(losses) / K,
                "v_ens/acc_mean":  sum(accs)   / K,
            })


def train_awr_policy(
    policy, v_net,
    obs_all: torch.Tensor, act_all: torch.Tensor, next_obs_all: torch.Tensor,
    epochs: int, batch_size: int, lr: float, beta: float, device,
    weight_clip: float = 20.0,
    normalize_adv: bool = False,
    adaptive_beta: bool = False,
    disagreement_lambda: float = 0.0,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Advantage-Weighted Regression: fine-tune `policy` from BC using
    one-step advantages computed from a frozen state-value network V.

        A(s, a) = V(s') - V(s)
        A_norm  = (A - mean(A)) / (std(A) + eps)        # if normalize_adv
        w       = exp(A / beta), clipped to [0, weight_clip]
        L       = - mean(w * log pi(a|s))

    The policy starts from BC and is steered toward actions whose successor
    state has higher V (= more "preferred" by the V-preference signal).
    Fully offline: walks transitions from the cached trajectories.

    `normalize_adv` (per-batch z-score on advantages) makes β interpretable
    across critic architectures: the weight distribution is then driven by
    the *shape* of the advantage distribution rather than its absolute
    scale. Off by default for V-AWR backward compatibility — set True on
    callers that want a scale-invariant β (V-IQL does).
    """
    v_net.eval()
    for p in v_net.parameters():
        p.requires_grad = False

    # Adaptive beta: scale the nominal beta by std(V_theta(s)) over the
    # full training transition set, computed ONCE before training. The
    # intuition: BT loss doesn't constrain V's absolute magnitude, so V_std
    # varies across envs / seeds. With beta_eff = beta * V_std, exp(A/beta)
    # produces comparable weight distributions across tasks without per-env
    # beta tuning.
    if adaptive_beta:
        with torch.no_grad():
            sample_n = min(20000, len(obs_all))
            idx = torch.randperm(len(obs_all))[:sample_n]
            v_vals = []
            for i in range(0, sample_n, batch_size):
                bo = obs_all[idx[i:i+batch_size]].to(device)
                v_vals.append(v_net(bo).cpu())
            v_std = float(torch.cat(v_vals).std().item())
        beta_eff = beta * max(v_std, 1e-6)
        print(f"  [adaptive-beta] V_std = {v_std:.4f}, "
              f"beta_nominal = {beta:.3f}, beta_eff = {beta_eff:.4f}", flush=True)
    else:
        beta_eff = beta

    # Disagreement-pessimistic AWR: when the ensemble disagrees on the
    # advantage at (s, s'), the policy update is down-weighted. Targets the
    # exact failure mode the null-space diagnostic identified: ensemble
    # members agree on broad rank (Spearman 0.63) but disagree on the tail,
    # and exp(A/beta) amplifies tail disagreement into policy divergence.
    # Pessimistic advantage: A_pess = A_mean - lambda * A_std (across K).
    # disagreement_lambda is a fixed regularization coefficient (in the same
    # spirit as CQL's alpha or EDAC's ensemble penalty weight) — not a
    # constraint multiplier. Auto-tuning it via cost-budget Lagrangian was
    # explored and failed because disagreement is a variance proxy, not a
    # cost actuator (see Phase 12e-12 notes). Fix it at 1.0 across tasks;
    # if it errs, it errs toward conservatism (safe-but-lower-R).
    pessimism_enabled = (disagreement_lambda > 0.0 and
                         hasattr(v_net, "forward_all"))
    if pessimism_enabled:
        print(f"  [disagreement-pessimism] lambda = {disagreement_lambda:.3f}",
              flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ds = TensorDataset(obs_all, act_all, next_obs_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_loss = 0.0
        run_w_mean = run_w_std = run_w_max = run_a_mean = 0.0
        run_a_disagreement = 0.0
        n_batches = 0
        for bo, ba, bno in loader:
            bo = bo.to(device); ba = ba.to(device); bno = bno.to(device)
            with torch.no_grad():
                if pessimism_enabled:
                    # Per-member advantages across the K ensemble members.
                    V_all_s  = v_net.forward_all(bo)    # [K, batch]
                    V_all_sp = v_net.forward_all(bno)   # [K, batch]
                    A_all = V_all_sp - V_all_s          # [K, batch]
                    A = A_all.mean(dim=0)               # [batch]
                    A_std = A_all.std(dim=0)            # [batch]
                    run_a_disagreement += A_std.mean().item()
                    A_pess = A - disagreement_lambda * A_std
                else:
                    A = v_net(bno) - v_net(bo)
                    A_pess = A
                if normalize_adv:
                    A_pess = (A_pess - A_pess.mean()) / (A_pess.std() + 1e-6)
                w = torch.exp(A_pess / beta_eff).clamp(max=weight_clip)
            log_p = policy.get_log_prob(bo, ba)
            loss = -(w * log_p).mean()

            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item()
            run_w_mean += w.mean().item()
            run_w_std += w.std().item()
            run_w_max += w.max().item()
            run_a_mean += A.mean().item()
            n_batches += 1

        avg_loss = run_loss / max(1, n_batches)
        avg_a_dis = run_a_disagreement / max(1, n_batches) if pessimism_enabled else 0.0
        _wlog(wandb_run, {
            "awr/epoch": epoch, "awr/loss": avg_loss,
            "awr/w_mean": run_w_mean / max(1, n_batches),
            "awr/w_std":  run_w_std  / max(1, n_batches),
            "awr/w_max":  run_w_max  / max(1, n_batches),
            "awr/A_mean": run_a_mean / max(1, n_batches),
            "awr/A_disagreement": avg_a_dis,
        })
        if epoch % log_every == 0 or epoch == 1:
            extra = f" A_dis={avg_a_dis:.3f}" if pessimism_enabled else ""
            print(f"  AWR epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f}  "
                  f"w: mean={run_w_mean/n_batches:.3f} std={run_w_std/n_batches:.3f} "
                  f"max={run_w_max/n_batches:.2f}  A_mean={run_a_mean/n_batches:+.3f}{extra}")


def train_q_td(
    q_net, q_target, v_theta,
    obs: torch.Tensor, act: torch.Tensor,
    next_obs: torch.Tensor, next_act: torch.Tensor,
    epochs: int, batch_size: int, lr: float,
    gamma: float, tau: float, device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Offline TD-learn Q with V_θ as the per-state preference reward:

        y(s, a, s', a') = V_θ(s) + γ * Q_target(s', a')

    where a' is the data (behavior-policy) next-action. Soft-updates the
    target network each batch with mixing coefficient `tau`. Q learns to
    estimate the discounted cumulative state-preference value of the
    trajectory following (s, a) — i.e. the long-horizon information that
    one-step V-AWR misses.
    """
    v_theta.eval()
    q_target.eval()
    for p in v_theta.parameters():
        p.requires_grad = False
    for p in q_target.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    ds = TensorDataset(obs, act, next_obs, next_act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_loss = run_y = run_q = 0.0
        n_batches = 0
        for bo, ba, bno, bna in loader:
            bo = bo.to(device); ba = ba.to(device)
            bno = bno.to(device); bna = bna.to(device)
            with torch.no_grad():
                q_next = q_target(bno, bna)
                v_now = v_theta(bo)
                y = v_now + gamma * q_next
            q_pred = q_net(bo, ba)
            loss = F.mse_loss(q_pred, y)

            opt.zero_grad(); loss.backward(); opt.step()

            # Soft-update target
            with torch.no_grad():
                for tp, sp in zip(q_target.parameters(), q_net.parameters()):
                    tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

            run_loss += loss.item()
            run_y += y.mean().item()
            run_q += q_pred.mean().item()
            n_batches += 1

        avg = run_loss / max(1, n_batches)
        _wlog(wandb_run, {
            "qtd/epoch": epoch, "qtd/loss": avg,
            "qtd/y_mean": run_y / max(1, n_batches),
            "qtd/q_mean": run_q / max(1, n_batches),
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  Q-TD epoch {epoch:3d}/{epochs} | loss={avg:.4f}  "
                  f"y_mean={run_y/n_batches:+.3f}  q_mean={run_q/n_batches:+.3f}")


def train_iql_q_vpsi(
    q_net, q_target, v_psi, v_theta,
    obs: torch.Tensor, act: torch.Tensor,
    next_obs: torch.Tensor, next_act: torch.Tensor,
    epochs: int, batch_size: int, lr: float,
    gamma: float, tau_polyak: float, tau_expectile: float, device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Proper IQL training of Q and V_psi using V_theta as per-step reward.

    Fixes the wrong-baseline bug in 04g (Phase 7d) where the advantage was
    Q(s,a) - V_theta(s) — Q being discounted-cumulative and V_theta being
    per-state preference value, so the subtraction was scale-mismatched
    and required ad-hoc normalization to make beta meaningful.

    Standard IQL formulation here:
        r(s,a,s') := V_theta(s')              # per-step reward = next-state V
        Q(s,a)    -> r + gamma * V_psi(s')    # TD bootstrap
        V_psi(s)  <- expectile_tau(Q_target(s, a)) under behavior actions
        A(s,a)    := Q(s,a) - V_psi(s)        # PROPER centered advantage

    V_psi is the soft-max over data actions (controlled by tau_expectile);
    Q_target is a Polyak-averaged copy of Q.

    Setting V_theta to an ensemble (Phase 10a) launders the BT-underspec
    per-state noise out before IQL bootstraps it into Q.
    """
    v_theta.eval()
    q_target.eval()
    for p in v_theta.parameters():
        p.requires_grad = False
    for p in q_target.parameters():
        p.requires_grad = False

    opt_q = torch.optim.Adam(q_net.parameters(), lr=lr)
    opt_v = torch.optim.Adam(v_psi.parameters(), lr=lr)

    ds = TensorDataset(obs, act, next_obs, next_act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_q = run_v = run_adv = 0.0
        n_batches = 0
        for bo, ba, bno, bna in loader:
            bo = bo.to(device); ba = ba.to(device)
            bno = bno.to(device); bna = bna.to(device)

            # --- V_psi update: expectile of Q_target(s, a) under data actions ---
            with torch.no_grad():
                q_for_v = q_target(bo, ba)
            v_pred = v_psi(bo)
            u = q_for_v - v_pred
            # Asymmetric L2: weight = tau on positive u, (1-tau) on negative u.
            # tau_expectile near 1 pulls V_psi toward the upper tail of Q.
            w_exp = torch.where(u > 0,
                                torch.full_like(u, tau_expectile),
                                torch.full_like(u, 1.0 - tau_expectile))
            v_loss = (w_exp * u.pow(2)).mean()
            opt_v.zero_grad(); v_loss.backward(); opt_v.step()

            # --- Q update: TD with V_theta as reward, V_psi as bootstrap ---
            with torch.no_grad():
                r = v_theta(bno)              # next-state preference value
                v_next = v_psi(bno)
                y = r + gamma * v_next
            q_pred = q_net(bo, ba)
            q_loss = F.mse_loss(q_pred, y)
            opt_q.zero_grad(); q_loss.backward(); opt_q.step()

            # --- Polyak update of Q_target ---
            with torch.no_grad():
                for tp, sp in zip(q_target.parameters(), q_net.parameters()):
                    tp.data.mul_(1.0 - tau_polyak).add_(sp.data, alpha=tau_polyak)
                adv = q_pred - v_psi(bo).detach()

            run_q += q_loss.item()
            run_v += v_loss.item()
            run_adv += adv.mean().item()
            n_batches += 1

        avg_q = run_q / max(1, n_batches)
        avg_v = run_v / max(1, n_batches)
        avg_a = run_adv / max(1, n_batches)
        _wlog(wandb_run, {
            "iql/epoch": epoch,
            "iql/q_loss": avg_q,
            "iql/v_psi_expectile_loss": avg_v,
            "iql/A_mean": avg_a,
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  IQL epoch {epoch:3d}/{epochs} | "
                  f"Q-MSE={avg_q:.4f}  V_psi-exp={avg_v:.4f}  A_mean={avg_a:+.3f}")


def train_iql_q_vpsi_ensemble(
    q_ens, q_target_ens, v_psi_ens, v_theta_ens,
    obs: torch.Tensor, act: torch.Tensor,
    next_obs: torch.Tensor, next_act: torch.Tensor,
    epochs: int, batch_size: int, lr: float,
    gamma: float, tau_polyak: float, tau_expectile: float, device,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """K-coupled IQL training (Phase 13). Each (Q_k, V_psi_k) pipeline is
    bootstrapped with V_theta_k as per-step reward — independent across k —
    so std_k(Q_k(s,a)) captures BT-underdet (propagated through V_theta) AND
    Q's own bootstrap variance.

    The update is identical to train_iql_q_vpsi but vectorized across K:
        V_psi_k(s) <- expectile_tau( Q_target_k(s, a) ) for each k
        Q_k(s,a)   -> V_theta_k(s') + gamma * V_psi_k(s')
        Q_target_k <- Polyak(Q_k)
    """
    v_theta_ens.eval()
    q_target_ens.eval()
    for p in v_theta_ens.parameters():
        p.requires_grad = False
    for p in q_target_ens.parameters():
        p.requires_grad = False

    K = q_ens.K
    opts_q = [torch.optim.Adam(m.parameters(), lr=lr) for m in q_ens.members]
    opts_v = [torch.optim.Adam(m.parameters(), lr=lr) for m in v_psi_ens.members]

    ds = TensorDataset(obs, act, next_obs, next_act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_q = run_v = run_adv_mean = run_adv_std = 0.0
        n_batches = 0
        for bo, ba, bno, bna in loader:
            bo = bo.to(device); ba = ba.to(device)
            bno = bno.to(device); bna = bna.to(device)

            # --- V_psi update per member: expectile of Q_target_k(s, a) ---
            with torch.no_grad():
                q_for_v = q_target_ens.forward_all(bo, ba)   # [K, batch]
            v_loss_sum = 0.0
            for k in range(K):
                v_pred_k = v_psi_ens.members[k](bo)
                u = q_for_v[k] - v_pred_k
                w_exp = torch.where(u > 0,
                                    torch.full_like(u, tau_expectile),
                                    torch.full_like(u, 1.0 - tau_expectile))
                v_loss_k = (w_exp * u.pow(2)).mean()
                opts_v[k].zero_grad(); v_loss_k.backward(); opts_v[k].step()
                v_loss_sum += v_loss_k.item()

            # --- Q update per member: TD with V_theta_k, V_psi_k bootstrap ---
            with torch.no_grad():
                v_theta_next_all = v_theta_ens.forward_all(bno)  # [K, batch]
                v_psi_next_all = v_psi_ens.forward_all(bno)      # [K, batch]
                y_all = v_theta_next_all + gamma * v_psi_next_all  # [K, batch]
            q_loss_sum = 0.0
            for k in range(K):
                q_pred_k = q_ens.members[k](bo, ba)
                q_loss_k = F.mse_loss(q_pred_k, y_all[k])
                opts_q[k].zero_grad(); q_loss_k.backward(); opts_q[k].step()
                q_loss_sum += q_loss_k.item()

            # --- Polyak target update per member ---
            with torch.no_grad():
                for k in range(K):
                    for tp, sp in zip(q_target_ens.members[k].parameters(),
                                      q_ens.members[k].parameters()):
                        tp.data.mul_(1.0 - tau_polyak).add_(sp.data, alpha=tau_polyak)
                # Disagreement diagnostic: std_k of advantage over batch
                q_all = q_ens.forward_all(bo, ba)         # [K, batch]
                v_all = v_psi_ens.forward_all(bo)         # [K, batch]
                adv_all = q_all - v_all                   # [K, batch]
                run_adv_mean += adv_all.mean().item()
                run_adv_std += adv_all.std(dim=0).mean().item()

            run_q += q_loss_sum / K
            run_v += v_loss_sum / K
            n_batches += 1

        _wlog(wandb_run, {
            "iql/epoch": epoch,
            "iql/q_loss_mean": run_q / max(1, n_batches),
            "iql/v_psi_expectile_loss_mean": run_v / max(1, n_batches),
            "iql/A_mean": run_adv_mean / max(1, n_batches),
            "iql/A_disagreement": run_adv_std / max(1, n_batches),
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  IQL-ens epoch {epoch:3d}/{epochs} | "
                  f"Q-MSE={run_q/n_batches:.4f}  V_psi-exp={run_v/n_batches:.4f}  "
                  f"A_mean={run_adv_mean/n_batches:+.3f}  "
                  f"A_dis={run_adv_std/n_batches:.3f}", flush=True)


def train_awr_q_ensemble_pess(
    policy, q_ens, v_psi_ens,
    obs: torch.Tensor, act: torch.Tensor,
    epochs: int, batch_size: int, lr: float, beta: float, device,
    weight_clip: float = 20.0,
    normalize_adv: bool = True,
    disagreement_lambda: float = 0.0,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """AWR with disagreement-pessimistic Q-advantage (Phase 13).

    For each (s, a) transition, K coupled critics give:
        A_k(s, a) = Q_k(s, a) - V_psi_k(s)

    Pessimistic advantage uses the ensemble:
        A_pess(s, a) = mean_k A_k - λ * std_k A_k

    The std term penalizes states where the ensemble disagrees — the
    epistemic-uncertainty pessimism transferred from V-AWR (Phase 12e-8) to
    the IQL bridge, but operating on the action-aware Q-advantage instead
    of the state-only V-difference.
    """
    q_ens.eval()
    v_psi_ens.eval()
    for p in q_ens.parameters():
        p.requires_grad = False
    for p in v_psi_ens.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ds = TensorDataset(obs, act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    pessimism_enabled = disagreement_lambda > 0.0
    if pessimism_enabled:
        print(f"  [Q-ens-pessimism] lambda = {disagreement_lambda:.3f}",
              flush=True)

    for epoch in range(1, epochs + 1):
        run_loss = run_w_mean = run_w_std = run_w_max = 0.0
        run_a_mean = run_a_dis = 0.0
        n_batches = 0
        for bo, ba in loader:
            bo = bo.to(device); ba = ba.to(device)
            with torch.no_grad():
                Q_all = q_ens.forward_all(bo, ba)        # [K, batch]
                V_all = v_psi_ens.forward_all(bo)        # [K, batch]
                A_all = Q_all - V_all                    # [K, batch]
                A = A_all.mean(dim=0)                    # [batch]
                A_std = A_all.std(dim=0)                 # [batch]
                run_a_dis += A_std.mean().item()
                if pessimism_enabled:
                    A_pess = A - disagreement_lambda * A_std
                else:
                    A_pess = A
                if normalize_adv:
                    A_pess = (A_pess - A_pess.mean()) / (A_pess.std() + 1e-6)
                w = torch.exp(A_pess / beta).clamp(max=weight_clip)
            log_p = policy.get_log_prob(bo, ba)
            loss = -(w * log_p).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item()
            run_w_mean += w.mean().item()
            run_w_std += w.std().item()
            run_w_max += w.max().item()
            run_a_mean += A.mean().item()
            n_batches += 1

        _wlog(wandb_run, {
            "awr/epoch": epoch,
            "awr/loss": run_loss / max(1, n_batches),
            "awr/w_mean": run_w_mean / max(1, n_batches),
            "awr/w_std":  run_w_std  / max(1, n_batches),
            "awr/w_max":  run_w_max  / max(1, n_batches),
            "awr/A_mean": run_a_mean / max(1, n_batches),
            "awr/A_disagreement": run_a_dis / max(1, n_batches),
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  AWR-Qens epoch {epoch:3d}/{epochs} | "
                  f"loss={run_loss/n_batches:.4f}  "
                  f"w: mean={run_w_mean/n_batches:.3f} "
                  f"std={run_w_std/n_batches:.3f} "
                  f"max={run_w_max/n_batches:.2f}  "
                  f"A_mean={run_a_mean/n_batches:+.3f}  "
                  f"A_dis={run_a_dis/n_batches:.3f}",
                  flush=True)


def train_awr_with_q(
    policy, q_net, v_theta,
    obs: torch.Tensor, act: torch.Tensor,
    epochs: int, batch_size: int, lr: float, beta: float, device,
    weight_clip: float = 20.0,
    normalize_adv: bool = True,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """AWR with A(s, a) = Q(s, a) - V_θ(s).

    Q encodes discounted cumulative state-preference value of acting `a`;
    V_θ is the baseline state value. Their difference is the advantage of
    this specific action over the "average" action implied by V_θ.

    `normalize_adv` (per-batch z-score) is on by default here because Q's
    scale depends on γ-cumulant of V over the horizon, which can vary
    several × across critic architectures. Normalization lets β remain
    interpretable across V-AWR (one-step) and V-IQL (long-horizon).
    """
    q_net.eval()
    v_theta.eval()
    for p in q_net.parameters():
        p.requires_grad = False
    for p in v_theta.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ds = TensorDataset(obs, act)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_loss = 0.0
        run_w_mean = run_w_std = run_w_max = run_a_mean = 0.0
        n_batches = 0
        for bo, ba in loader:
            bo = bo.to(device); ba = ba.to(device)
            with torch.no_grad():
                A = q_net(bo, ba) - v_theta(bo)
                if normalize_adv:
                    A = (A - A.mean()) / (A.std() + 1e-6)
                w = torch.exp(A / beta).clamp(max=weight_clip)
            log_p = policy.get_log_prob(bo, ba)
            loss = -(w * log_p).mean()

            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item()
            run_w_mean += w.mean().item()
            run_w_std += w.std().item()
            run_w_max += w.max().item()
            run_a_mean += A.mean().item()
            n_batches += 1

        avg = run_loss / max(1, n_batches)
        _wlog(wandb_run, {
            "awrq/epoch": epoch, "awrq/loss": avg,
            "awrq/w_mean": run_w_mean / max(1, n_batches),
            "awrq/w_std":  run_w_std  / max(1, n_batches),
            "awrq/w_max":  run_w_max  / max(1, n_batches),
            "awrq/A_mean": run_a_mean / max(1, n_batches),
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  AWR-Q epoch {epoch:3d}/{epochs} | loss={avg:.4f}  "
                  f"w: mean={run_w_mean/n_batches:.3f} std={run_w_std/n_batches:.3f} "
                  f"max={run_w_max/n_batches:.2f}  A_mean={run_a_mean/n_batches:+.3f}")


def train_awr_flex(
    policy,
    obs_all: torch.Tensor, act_all: torch.Tensor,
    adv_members: torch.Tensor,
    epochs: int, batch_size: int, lr: float, beta: float, device,
    weight_mode: str = "exp",
    weight_clip: float = 20.0,
    normalize_adv: bool = False,
    disagreement_lambda: float = 0.0,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Extraction-lab AWR: fine-tune `policy` from BC using PRECOMPUTED
    per-member advantages (Phase B' extraction experiments).

    adv_members: [K, N] tensor. K=1 for oracle advantage sources; K=ensemble
    size for learned V (precomputed since V is frozen during AWR anyway).
    Pessimism: A_pess = mean_k(A) - lambda * std_k(A)  (std=0 when K=1).

    weight_mode:
      "exp"    — w = exp(A_pess / beta), clipped            (paper pipeline)
      "rank"   — per-batch centered fractional rank r in [-0.5, 0.5],
                 w = exp(r / beta), clipped. Consumes ONLY the ordering of
                 advantages: robust to the BT null-space tail by construction
                 (cross-seed ranks transfer at rho 0.63-0.84; tails at
                 Jaccard 0.15 — see manuscript Sec 4.10).
      "binary" — w = 1[A_pess > 0]  (CRR-style indicator, Wang et al. 2020)
    """
    assert weight_mode in ("exp", "rank", "binary"), weight_mode
    K = adv_members.shape[0]
    A_mean_all = adv_members.mean(dim=0)
    A_std_all = adv_members.std(dim=0) if K > 1 else torch.zeros_like(A_mean_all)
    A_pess_all = A_mean_all - disagreement_lambda * A_std_all
    print(f"  [awr-flex] mode={weight_mode} K={K} lambda={disagreement_lambda} "
          f"A_pess: mean={A_pess_all.mean():+.4f} std={A_pess_all.std():.4f} "
          f"frac>0={(A_pess_all > 0).float().mean():.3f}", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ds = TensorDataset(obs_all, act_all, A_pess_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        run_loss = run_w_mean = run_w_std = run_w_max = run_a_mean = 0.0
        n_batches = 0
        for bo, ba, bA in loader:
            bo = bo.to(device); ba = ba.to(device); bA = bA.to(device)
            with torch.no_grad():
                if normalize_adv:
                    bA = (bA - bA.mean()) / (bA.std() + 1e-6)
                if weight_mode == "exp":
                    w = torch.exp(bA / beta).clamp(max=weight_clip)
                elif weight_mode == "rank":
                    n = bA.numel()
                    ranks = bA.argsort().argsort().float()
                    r = ranks / max(n - 1, 1) - 0.5          # [-0.5, 0.5]
                    w = torch.exp(r / beta).clamp(max=weight_clip)
                else:  # binary
                    w = (bA > 0).float()
            log_p = policy.get_log_prob(bo, ba)
            loss = -(w * log_p).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item()
            run_w_mean += w.mean().item(); run_w_std += w.std().item()
            run_w_max += w.max().item(); run_a_mean += bA.mean().item()
            n_batches += 1

        avg_loss = run_loss / max(1, n_batches)
        _wlog(wandb_run, {
            "awr/epoch": epoch, "awr/loss": avg_loss,
            "awr/w_mean": run_w_mean / max(1, n_batches),
            "awr/w_std": run_w_std / max(1, n_batches),
            "awr/w_max": run_w_max / max(1, n_batches),
            "awr/A_mean": run_a_mean / max(1, n_batches),
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  AWR-flex epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f}  "
                  f"w: mean={run_w_mean/n_batches:.3f} std={run_w_std/n_batches:.3f} "
                  f"max={run_w_max/n_batches:.2f}  A_mean={run_a_mean/n_batches:+.4f}",
                  flush=True)


def train_dynamics_ensemble(
    dyn, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device,
    val_frac: float = 0.05,
    log_every: int = 5,
    wandb_run=None,
) -> float:
    """Train a DynamicsEnsemble on one-step transitions with per-member MSE
    on state deltas. All members see the same data; diversity comes from
    random init (parity with VEnsemble's design). Returns held-out MSE of
    the ensemble mean so callers can gate on model quality.
    """
    n = len(obs)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    ds = TensorDataset(obs[tr_idx], act[tr_idx], next_obs[tr_idx])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(dyn.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        run_loss, n_batches = 0.0, 0
        for bo, ba, bno in loader:
            bo, ba, bno = bo.to(device), ba.to(device), bno.to(device)
            pred_all = dyn.forward_all(bo, ba)              # [K, B, obs]
            loss = ((pred_all - bno.unsqueeze(0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); n_batches += 1
        avg = run_loss / max(1, n_batches)
        _wlog(wandb_run, {"dyn/epoch": epoch, "dyn/train_mse": avg})
        if epoch % log_every == 0 or epoch == 1:
            print(f"  DYN epoch {epoch:3d}/{epochs} | mse={avg:.5f}", flush=True)

    with torch.no_grad():
        vo = obs[val_idx].to(device); va = act[val_idx].to(device)
        vno = next_obs[val_idx].to(device)
        val_mse = float(((dyn(vo, va) - vno) ** 2).mean().item())
        # per-dim baseline: predicting s' = s
        ident_mse = float(((vo - vno) ** 2).mean().item())
    print(f"  DYN val: ens-mean mse={val_mse:.5f} vs identity {ident_mse:.5f} "
          f"(ratio {val_mse / max(ident_mse, 1e-12):.3f})", flush=True)
    _wlog(wandb_run, {"dyn/val_mse": val_mse, "dyn/identity_mse": ident_mse})
    return val_mse


def train_cf_awr(
    policy, bc_ref, v_ens, dyn,
    obs_all: torch.Tensor, act_all: torch.Tensor,
    epochs: int, batch_size: int, lr: float, device,
    n_candidates: int = 8,
    cand_sigma_scale: float = 1.0,
    cf_beta: float = 0.1,
    lambda_v: float = 1.0,
    lambda_dyn: float = 1.0,
    perstate_norm: bool = False,
    log_every: int = 5,
    wandb_run=None,
) -> None:
    """Counterfactual AWR (Phase B'.3): imitate the best V-scored action at
    each state, where candidates come from a frozen BC reference policy and
    are scored through a one-step dynamics ensemble.

    perstate_norm: z-score A_pess ACROSS THE CANDIDATE SET at each state
    before the softmax. The v1 runs showed raw candidate-advantage spreads
    of ~0.01-0.03 against cf_beta=0.1, leaving the softmax near-uniform and
    the policy distilling BC (mechanism never engaged). Per-state
    normalization makes the temperature scale-free: cf_beta then selects
    how concentrated the weights are on the best candidate REGARDLESS of
    the advantage's absolute scale. The trade-off: it also amplifies noise
    at states where the true spread is negligible, so the (normalized)
    softmax always commits somewhere; whether that helps is exactly what
    the v2 experiment measures.

    For each state s with data action a_0 and BC-sampled candidates
    a_1..a_M:
        A[k, d, j] = V_k(f_d(s, a_j)) - V_k(s)      per V-member k, dyn-member d
        A_pess[j]  = mean_{k,d} A - lambda_v * std_k(mean_d A)
                                  - lambda_dyn * std_d(mean_k A)
        w[j]       = softmax_j(A_pess / cf_beta)
        L          = - sum_j w[j] * log pi(a_j | s)

    The data action is always candidate j=0, so when every counterfactual is
    OOD (high dynamics disagreement) or low-value, the softmax collapses onto
    the data action and the update degrades gracefully to BC. This is the
    structural difference from every AWR variant above: the policy can be
    steered toward actions the dataset never took at s, with the dual
    disagreement penalty as the support constraint.
    """
    for m in (bc_ref, v_ens, dyn):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    ds = TensorDataset(obs_all, act_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    M = n_candidates

    for epoch in range(1, epochs + 1):
        run_loss = run_wdata = run_cf_frac = run_apess_gap = run_spread = 0.0
        n_batches = 0
        for bo, ba in loader:
            bo, ba = bo.to(device), ba.to(device)
            B = bo.shape[0]
            with torch.no_grad():
                mean, std = bc_ref(bo)                       # [B, A], [A]
                eps = torch.randn(M, B, mean.shape[-1], device=device)
                cands = mean.unsqueeze(0) + cand_sigma_scale * std * eps
                cands = torch.cat([ba.unsqueeze(0), cands], dim=0)  # [M+1, B, A]

                obs_rep = bo.unsqueeze(0).expand(M + 1, B, -1).reshape((M + 1) * B, -1)
                act_rep = cands.reshape((M + 1) * B, -1)
                # [Kd, (M+1)*B, obs]
                pred = dyn.forward_all(obs_rep, act_rep)
                Kd = pred.shape[0]
                # V of predicted next states: [Kv, Kd*(M+1)*B]
                V_next = v_ens.forward_all(pred.reshape(Kd * (M + 1) * B, -1))
                Kv = V_next.shape[0]
                V_next = V_next.reshape(Kv, Kd, M + 1, B)
                V_s = v_ens.forward_all(bo).reshape(Kv, 1, 1, B)
                A = V_next - V_s                              # [Kv, Kd, M+1, B]
                A_mean = A.mean(dim=(0, 1))                   # [M+1, B]
                std_v = A.mean(dim=1).std(dim=0)              # [M+1, B]
                std_d = A.mean(dim=0).std(dim=0)              # [M+1, B]
                A_pess = A_mean - lambda_v * std_v - lambda_dyn * std_d
                spread = A_pess.std(dim=0)                    # [B] raw per-state spread
                if perstate_norm:
                    A_pess = (A_pess - A_pess.mean(dim=0, keepdim=True)) / \
                             (A_pess.std(dim=0, keepdim=True) + 1e-6)
                w = torch.softmax(A_pess / cf_beta, dim=0)    # [M+1, B]

            log_p = policy.get_log_prob(
                obs_rep, act_rep).reshape(M + 1, B)
            loss = -(w * log_p).sum(dim=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            run_loss += loss.item()
            run_wdata += w[0].mean().item()                  # weight on data action
            run_cf_frac += (w.argmax(dim=0) != 0).float().mean().item()
            run_apess_gap += (A_pess.max(dim=0).values - A_pess[0]).mean().item()
            run_spread += spread.mean().item()               # raw scale, pre-norm
            n_batches += 1

        nb = max(1, n_batches)
        _wlog(wandb_run, {
            "cfawr/epoch": epoch, "cfawr/loss": run_loss / nb,
            "cfawr/w_data": run_wdata / nb,
            "cfawr/cf_argmax_frac": run_cf_frac / nb,
            "cfawr/apess_gap": run_apess_gap / nb,
            "cfawr/raw_spread": run_spread / nb,
        })
        if epoch % log_every == 0 or epoch == 1:
            print(f"  CF-AWR epoch {epoch:3d}/{epochs} | loss={run_loss/nb:.4f}  "
                  f"w_data={run_wdata/nb:.3f}  cf_best={run_cf_frac/nb:.3f}  "
                  f"gap={run_apess_gap/nb:+.4f}  spread={run_spread/nb:.4f}",
                  flush=True)
