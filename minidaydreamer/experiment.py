"""Train only on the fixed corpus and RSSM transitions. Evaluate on fresh physics resets."""

from contextlib import ExitStack
import json

import numpy as np
import torch

from .data import Dataset
from .models import HORIZON, FlowPolicy, NoiseSAC, RSSM
from .stack import ACTION_DIM


def fit(module, steps, loss_fn):
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5)
        opt.step()
    module.eval().requires_grad_(False)


@torch.no_grad()
def posterior_states(wm, data, episodes):
    states = []
    for i in episodes:
        obs = data.obs[i]
        state = wm.initial(obs[:1])
        for j in range(data.lengths[i]):
            states.append(state)
            h = wm.advance(state, data.actions[i:i+1, j])
            state, _ = wm.filter(h, obs[j+1:j+2])
    return torch.cat(states)


@torch.no_grad()
def world_diagnostics(wm, data):
    pairs = np.flatnonzero(data.split == "holdout").reshape(-1, 2)
    real_gaps, predicted_gaps, pred_errors, static_errors = [], [], [], []
    for good, bad in pairs:
        if data.seeds[good] != data.seeds[bad]:
            raise ValueError("Held-out pairs must share the same starting scene")
        if not torch.allclose(data.obs[good, 0], data.obs[bad, 0], atol=.01):
            raise ValueError("Held-out pair starts differ by more than 1 mm or have different gripper states")
        # Branch where the recorded commands first differ. Condition on the common prefix.
        end = min(data.lengths[good], data.lengths[bad])
        differs = (data.actions[good, :end] - data.actions[bad, :end]).abs().amax(1) > 1e-5
        indices = torch.nonzero(differs)
        if not len(indices):
            continue
        start = int(indices[0, 0])
        width = min(3 * HORIZON, end - start)
        state = wm.initial(data.obs[good:good+1, 0])
        for t in range(start):
            h = wm.advance(state, data.actions[good:good+1, t])
            state, _ = wm.filter(h, data.obs[good:good+1, t+1])
        pred, truth = [], []
        for i in (good, bad):
            s, total = state, 0.
            for t in range(start, start + width):
                s = wm.imagine(s, data.actions[i:i+1, t])
                total += wm.reward(s).item()
            pred.append(total)
            truth.append(data.rewards[i, start:start+width].sum().item())
            actual = data.obs[i, start+width, 3:6]
            pred_errors.append((wm.decode(s)[0, 3:6] - actual).norm().item() * 100)
            static_errors.append((data.obs[i, start, 3:6] - actual).norm().item() * 100)
        if abs(truth[0] - truth[1]) > .01:
            real_gaps.append(truth[0] - truth[1])
            predicted_gaps.append(pred[0] - pred[1])
    return dict(informative_pairs=len(real_gaps),
                rank_accuracy=float(np.mean(np.multiply(real_gaps, predicted_gaps) > 0)) if real_gaps else None,
                red_position_error_mm=float(np.mean(pred_errors)),
                static_prediction_error_mm=float(np.mean(static_errors)))


@torch.no_grad()
def imagine_batch(wm, policy, sac, starts):
    state = starts[torch.randint(len(starts), (64,))]
    transitions = []
    for _ in range(2):
        shift, _ = sac.draw(state)
        noise = torch.randn(len(state), HORIZON, ACTION_DIM) + shift[:, None]
        actions = policy.sample(wm.decode(state), noise)
        successor = state
        discount = torch.ones_like(state[:, :1])
        reward = torch.zeros_like(discount)
        for action in actions.unbind(1):
            successor = wm.imagine(successor, action, sample=True)
            reward += discount * wm.reward(successor)
            discount *= .99 * (1 - wm.success(successor).sigmoid())
        transitions.append(torch.cat((state, shift, reward, discount, successor), -1))
        state = successor
    return torch.cat(transitions)


def train(args):
    data = Dataset(args.data, args.seed)
    demo = np.flatnonzero(data.split == "demo")
    wm_ids = np.flatnonzero(data.split != "holdout") if args.coverage == "varied" else demo
    # Fit the exact same BC policy first for both coverage conditions.
    policy = FlowPolicy()
    def bc_loss():
        obs, act, _, _ = data.sample(demo, 64, HORIZON)
        return policy.loss(obs[:, 0], act)
    fit(policy, args.bc_steps, bc_loss)
    wm = RSSM()
    fit(wm, args.model_steps, lambda: wm.loss(*data.sample(wm_ids, 32, 8)))
    metrics = world_diagnostics(wm, data)
    print(json.dumps(metrics), flush=True)
    starts = posterior_states(wm, data, wm_ids)
    sac = NoiseSAC()
    # Circular replay measured in 128-transition batches.
    blocks = min(128, args.rl_steps)
    replay = torch.empty((blocks, 128, 2 * starts.shape[1] + ACTION_DIM + 2))
    for i in range(args.rl_steps):
        replay[i % blocks] = imagine_batch(wm, policy, sac, starts)
        flat = replay[:min(i + 1, blocks)].flatten(0, 1)
        sample = flat[torch.randint(len(flat), (128,))]
        sac.update(*sample.split([starts.shape[1], ACTION_DIM, 1, 1, starts.shape[1]], dim=1))
    torch.save(dict(policy=policy.state_dict(), world=wm.state_dict(), actor=sac.actor.state_dict(),
                    eval_seed=int(data.seeds.max()) + 1, coverage=args.coverage, training_seed=args.seed,
                    demos=len(demo), world_episodes=len(wm_ids), backend=data.backend, diagnostics=metrics), args.out)


@torch.no_grad()
def evaluate(args, env):
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    pi, wm, sac = FlowPolicy(), RSSM(), NoiseSAC()
    pi.load_state_dict(ckpt["policy"])
    wm.load_state_dict(ckpt["world"])
    sac.actor.load_state_dict(ckpt["actor"])
    if ckpt["backend"] != env.backend:
        raise ValueError(f"Checkpoint requires the {ckpt['backend']} backend")
    results = {}
    for mode in ("base", "steered"):
        successes = []
        for i in range(args.episodes):
            seed = ckpt["eval_seed"] + i
            generator = torch.Generator().manual_seed(seed)
            obs = torch.from_numpy(env.reset(seed))[None]
            state = wm.initial(obs)
            with ExitStack() as recording:
                if i == 0 and args.video_dir:
                    import imageio.v2 as imageio
                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    path = args.video_dir / f"{mode}-{seed}.mp4"
                    if path.exists():
                        raise FileExistsError(path)
                    writer = recording.enter_context(imageio.get_writer(path, fps=round(1 / env.env.unwrapped.step_dt)))
                    env.frame_callback = lambda: writer.append_data(env.rgb())
                    recording.callback(setattr, env, "frame_callback", None)
                done = False
                while not done:
                    noise = torch.randn(1, HORIZON, ACTION_DIM, generator=generator)
                    if mode == "steered":
                        noise += sac.draw(state, deterministic=True)[0][:, None]
                    chunk = pi.sample(obs, noise)[0]
                    for action in chunk:
                        o, success, done = env.step(action.numpy())
                        obs = torch.from_numpy(o)[None]
                        state, _ = wm.filter(wm.advance(state, action[None]), obs)
                        if done:
                            break
            successes.append(int(success))
        results[mode] = successes
    delta = np.array(results["steered"]) - results["base"]
    return dict(episodes=args.episodes, first_seed=ckpt["eval_seed"],
                base_stacks=sum(results["base"]), steered_stacks=sum(results["steered"]),
                gained=int((delta > 0).sum()), lost=int((delta < 0).sum()),
                success_rate_difference=float(delta.mean()), world_validation=ckpt["diagnostics"])
