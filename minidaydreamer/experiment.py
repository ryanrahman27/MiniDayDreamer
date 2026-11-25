"""Train only on the fixed corpus and RSSM transitions. Evaluate on fresh physics resets."""

from contextlib import ExitStack
import hashlib
import json

import numpy as np
import torch

from .data import Dataset
from .models import HORIZON, FlowPolicy, NoiseSAC, make_policy, make_world
from .stack import ACTION_DIM, STEP, potential
from .state_reward import StateReward
from .transport import CarryStall, WINDOW, targetward_xy


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
def imagine_batch(wm, policy, sac, starts, chunks=2, state_reward=None):
    indices = torch.randint(len(starts), (128 // chunks,))
    state = starts[indices]
    dwell = state_reward.starts[indices] if state_reward is not None else None
    transitions = []
    for _ in range(chunks):
        shift, _ = sac.draw(state)
        noise = torch.randn(len(state), HORIZON, ACTION_DIM) + shift[:, None]
        actions = policy.sample(wm.decode(state), noise)
        successor = state
        discount = torch.ones_like(state[:, :1])
        reward = torch.zeros_like(discount)
        for action in actions.unbind(1):
            previous = successor
            successor = wm.imagine(successor, action, sample=True)
            if state_reward is None:
                tick_reward = wm.reward(successor)
            else:
                tick_reward, dwell = state_reward.step(wm.decode(previous), wm.decode(successor), dwell)
            reward += discount * tick_reward
            discount *= .99 * (1 - wm.success(successor).sigmoid())
        transitions.append(torch.cat((state, shift, reward, discount, successor), -1))
        state = successor
    return torch.cat(transitions)


def fit_base(data, demo, steps):
    policy = FlowPolicy()
    def bc_loss():
        obs, act, _, _ = data.sample(demo, 64, HORIZON)
        return policy.loss(obs[:, 0], act)
    fit(policy, steps, bc_loss)
    return policy


def calibrate(args, env):
    """Choose the earliest candidate that succeeds at least once on separate development scenes."""
    history = []
    for steps in args.bc_steps:
        data = Dataset(args.data, args.seed)
        if data.backend != env.backend:
            raise ValueError("Calibration must use the corpus backend")
        torch.manual_seed(args.seed)
        demo = np.flatnonzero(data.split == "demo")
        policy = fit_base(data, demo, steps)
        first_seed = int(data.seeds.max()) + 10001
        rows = []
        for seed in range(first_seed, first_seed + args.episodes):
            metrics, _ = policy_episode(env, policy, None, None, seed)
            rows.append({"seed": seed, **metrics})
        history.append({"steps": steps, "outcomes": rows})
        successes = sum(row["success"] for row in rows)
        print(f"Calibration: {steps} BC updates, {successes}/{len(rows)} stacks", flush=True)
        if successes:
            break
    torch.save(dict(policy=policy.state_dict(), bc_steps=steps, calibration=history,
                    corpus_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest()), args.out)


def train(args):
    data = Dataset(args.data, args.seed)
    demo = np.flatnonzero(data.split == "demo")
    direct = StateReward(data, demo) if args.reward_mode == "direct" else None
    wm_ids = np.flatnonzero(data.split != "holdout") if args.coverage == "varied" else demo
    if args.base_model:
        base = torch.load(args.base_model, map_location="cpu", weights_only=True)
        if base["corpus_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest():
            raise ValueError("Base policy was fitted on a different corpus")
        policy = FlowPolicy()
        policy.load_state_dict(base["policy"])
        policy.eval().requires_grad_(False)
    else:
        policy = fit_base(data, demo, args.bc_steps)
    wm = make_world(args.world_model)
    fit(wm, args.model_steps, lambda: wm.loss(*data.sample(wm_ids, 32, 8)))
    metrics = world_diagnostics(wm, data)
    print(json.dumps(metrics), flush=True)
    # Only the world-model fitting data varies between arms of this experiment.
    starts = posterior_states(wm, data, demo)
    pre_actor_rng = torch.get_rng_state()
    if args.match_short:
        ref = torch.load(args.match_short, map_location="cpu", weights_only=True)
        if (ref["coverage"] != args.coverage or ref["training_seed"] != args.seed
                or ref.get("imagination_chunks", 2) != 2):
            raise ValueError("Mismatched short-horizon reference")
        verify_weights(policy, ref["policy"], "frozen base")
        verify_weights(wm, ref["world"], "frozen world")
        reference_reward = StateReward(data, demo) if ref.get("reward_mode") == "direct" else None
        short = fit_steering(wm, policy, starts, args.rl_steps, 2, reference_reward)
        verify_weights(short.actor, ref["actor"], "six-step actor")
        print("Reproduced the base, RSSM and six-step actor weights exactly", flush=True)
        torch.set_rng_state(pre_actor_rng)
    sac = fit_steering(wm, policy, starts, args.rl_steps, args.imagination_chunks, direct)
    torch.save(dict(policy=policy.state_dict(), world=wm.state_dict(), actor=sac.actor.state_dict(),
                    eval_seed=int(data.seeds.max()) + 1, coverage=args.coverage, training_seed=args.seed,
                    demos=len(demo), world_episodes=len(wm_ids), imagination_start_episodes=len(demo),
                    backend=data.backend, diagnostics=metrics, world_model=args.world_model,
                    imagination_chunks=args.imagination_chunks, rl_steps=args.rl_steps,
                    reward_mode=args.reward_mode, direct_bonuses=direct.bonuses if direct else None,
                    imagined_ticks=args.rl_steps * 128 * HORIZON,
                    pre_actor_rng=pre_actor_rng, matched_short=bool(args.match_short)), args.out)


def verify_weights(module, weights, name):
    current = module.state_dict()
    if current.keys() != weights.keys() or any(not torch.equal(current[k], weights[k]) for k in weights):
        raise ValueError(f"Could not reproduce {name} weights")


def fit_steering(wm, policy, starts, rl_steps, chunks, state_reward=None):
    sac = NoiseSAC(wm.features)
    # Circular replay measured in 128-transition batches.
    blocks = min(128, rl_steps)
    replay = torch.empty((blocks, 128, 2 * starts.shape[1] + ACTION_DIM + 2))
    for i in range(rl_steps):
        replay[i % blocks] = imagine_batch(wm, policy, sac, starts, chunks, state_reward)
        flat = replay[:min(i + 1, blocks)].flatten(0, 1)
        sample = flat[torch.randint(len(flat), (128,))]
        sac.update(*sample.split([starts.shape[1], ACTION_DIM, 1, 1, starts.shape[1]], dim=1))
    return sac


def load_policies(path, env):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["backend"] != env.backend:
        raise ValueError(f"Checkpoint requires the {ckpt['backend']} backend")
    wm = make_world(ckpt.get("world_model", "absolute"))
    pi, sac = make_policy(ckpt.get("policy_kind", "original")), NoiseSAC(wm.features)
    pi.load_state_dict(ckpt["policy"])
    wm.load_state_dict(ckpt["world"])
    sac.actor.load_state_dict(ckpt["actor"])
    return ckpt, pi, wm, sac


@torch.no_grad()
def policy_episode(env, pi, wm, sac, seed, video=None, trace=None, release_gate=False, transport_pulse=None,
                   handoff_at_target=False):
    if handoff_at_target and sac is None:
        raise ValueError("Target handoff requires a steering actor")
    generator = torch.Generator().manual_seed(seed)
    first = env.reset(seed)
    obs = torch.from_numpy(first)[None]
    state = wm.initial(obs) if sac is not None else None
    lifted = False
    steering_off_step = -1
    steps = 0
    if transport_pulse not in (None, "xy", "height"):
        raise ValueError("Unknown transport intervention")
    detector = CarryStall(first) if transport_pulse else None
    trigger, pulse_steps = -1, 0
    target_z = None
    if trace:
        trace.parent.mkdir(parents=True, exist_ok=True)
        observations, commands, proposed_grip, proposals = [first.copy()], [], [], []
        apertures = [getattr(env, "aperture", np.nan)]
    with ExitStack() as recording:
        if video:
            import imageio.v2 as imageio
            video.parent.mkdir(parents=True, exist_ok=True)
            if video.exists():
                raise FileExistsError(video)
            writer = recording.enter_context(imageio.get_writer(video, fps=round(1 / env.env.unwrapped.step_dt)))
            env.frame_callback = lambda: writer.append_data(env.rgb())
            recording.callback(setattr, env, "frame_callback", None)
        done = False
        while not done:
            noise = torch.randn(1, HORIZON, ACTION_DIM, generator=generator)
            chunk_start = obs
            shifted = noise
            if sac is not None and steering_off_step < 0:
                shifted = noise + sac.draw(state, deterministic=True)[0][:, None]
            chunk = pi.sample(chunk_start, shifted)[0]
            for index in range(len(chunk)):
                if handoff_at_target and steering_off_step < 0:
                    before = obs[0].numpy()
                    if (before[9] > .5 and before[5] - first[5] > .2
                            and np.linalg.norm(before[3:5] - before[6:8]) < .12):
                        steering_off_step = steps
                        # Keep the chunk clock and its original Gaussian draw, even on a mid-chunk handoff.
                        chunk = pi.sample(chunk_start, noise)[0]
                action = chunk[index]
                command = action.numpy()
                if transport_pulse:
                    before = obs[0].numpy()
                    if trigger < 0 and detector.observe(before):
                        trigger = steps
                        target_z = before[5] + .35
                    if trigger >= 0 and steps == trigger + pulse_steps and pulse_steps < WINDOW:
                        if detector.carrying(before) and command[3] > 0 and np.linalg.norm(before[6:8]-before[3:5]) > .12:
                            command = command.copy()
                            if transport_pulse == "xy":
                                command[:2] = targetward_xy(before)
                            else:
                                command[2] = np.clip(.1 * (target_z-before[5]) / STEP, 0., .85)
                            pulse_steps += 1
                if trace:
                    proposed_grip.append(float(command[3]))
                    if transport_pulse:
                        proposals.append(action.numpy().copy())
                if release_gate and lifted and command[3] <= 0:
                    before = obs[0].numpy()
                    if before[9] > .5 and before[10] > .5:
                        delta_m = .1 * (before[3:6] - before[6:9])
                        if np.linalg.norm(delta_m[:2]) >= .012 or abs(delta_m[2] - env.block_height) >= .008:
                            command = command.copy()
                            command[3] = 1
                o, success, done = env.step(command)
                if trace:
                    observations.append(o.copy())
                    commands.append(command.copy())
                    apertures.append(getattr(env, "aperture", np.nan))
                obs = torch.from_numpy(o)[None]
                if sac is not None:
                    state, _ = wm.filter(wm.advance(state, action[None]), obs)
                steps += 1
                lifted |= bool(o[9] > .5 and o[5] - first[5] > .2)
                if done:
                    break
    if trace:
        extra = dict(pulse_trigger=trigger, pulse_steps=pulse_steps,
                     proposals=np.stack(proposals)) if transport_pulse else {}
        if handoff_at_target:
            extra["steering_off_step"] = steering_off_step
        with trace.open("xb") as f:
            np.savez_compressed(f, obs=np.stack(observations), actions=np.stack(commands),
                                proposed_grip=proposed_grip, aperture_m=apertures, success=success, seed=seed, **extra)
    # Shaping telescopes to the change in potential over an episode.
    return dict(success=int(success), steps=steps, lifted=lifted,
                episode_return=float(potential(o) - potential(first) + 2 * success),
                final_xy_error_mm=float(100 * np.linalg.norm(o[3:5] - o[6:8]))), first.copy()


def evaluate(args, env):
    ckpt, pi, wm, sac = load_policies(args.model, env)
    results = {}
    for mode in ("base", "steered"):
        successes = []
        for i in range(args.episodes):
            seed = ckpt["eval_seed"] + i
            video = args.video_dir / f"{mode}-{seed}.mp4" if i == 0 and args.video_dir else None
            metrics, _ = policy_episode(env, pi, wm, sac if mode == "steered" else None, seed, video)
            successes.append(metrics["success"])
        results[mode] = successes
    delta = np.array(results["steered"]) - results["base"]
    return dict(episodes=args.episodes, first_seed=ckpt["eval_seed"],
                base_stacks=sum(results["base"]), steered_stacks=sum(results["steered"]),
                gained=int((delta > 0).sum()), lost=int((delta < 0).sum()),
                success_rate_difference=float(delta.mean()), world_validation=ckpt["diagnostics"])


def compare(args, env):
    """Evaluate each arm on the same scene and base-noise sequence."""
    d, pi, wm_d, sac_d = load_policies(args.demos_model, env)
    v, pi_v, wm_v, sac_v = load_policies(args.varied_model, env)
    if (d["coverage"], v["coverage"]) != ("demos", "varied") or d["eval_seed"] != v["eval_seed"]:
        raise ValueError("Supply the demos and varied checkpoints from the same split")
    if not all(torch.equal(a, b) for a, b in zip(pi.parameters(), pi_v.parameters(), strict=True)):
        raise ValueError("The two checkpoints must contain identical base policy weights")
    arms = [("base", wm_d, None), ("demos", wm_d, sac_d), ("varied", wm_v, sac_v)]
    if args.original_models:
        for coverage, path in zip(("demos", "varied"), args.original_models, strict=True):
            ref, ref_pi, ref_wm, ref_sac = load_policies(path, env)
            if ref["coverage"] != coverage or ref["eval_seed"] != d["eval_seed"]:
                raise ValueError("Reference checkpoint coverage or split differs")
            if not all(torch.equal(a, b) for a, b in zip(pi.parameters(), ref_pi.parameters(), strict=True)):
                raise ValueError("Reference checkpoint contains a different base policy")
            arms.append(("original_" + coverage, ref_wm, ref_sac))
    if args.test_release_gate:
        arms.append(("gated_base", wm_d, None))
    if args.test_transport_pulse:
        arms.append(("pulse_base", wm_d, None))
    with args.out.open("x", encoding="utf-8") as output:
        for i in range(args.episodes):
            scene = args.scene_offset + i
            seed = d["eval_seed"] + scene
            row = {"seed": seed}
            first = None
            # Rotate ordering to balance any effect of simulator history.
            offset = scene % len(arms)
            for mode, wm, sac in arms[offset:] + arms[:offset]:
                video = args.video_dir / f"{mode}-{seed}.mp4" if i == 0 and args.video_dir else None
                trace = args.trace_dir / f"{mode}-{seed}.npz" if args.trace_dir else None
                metrics, start = policy_episode(env, pi, wm, sac, seed, video, trace,
                                                mode == "gated_base", "xy" if mode == "pulse_base" else None)
                if first is None:
                    first = start
                    row["start"] = start.tolist()
                elif not np.allclose(first, start, atol=.01, rtol=0):
                    raise ValueError(f"Scene {seed} did not reset to the same start within 1 mm: {first} vs {start}")
                row[mode] = metrics
            print(json.dumps(row), file=output, flush=True)
            print(f"Evaluated {i+1}/{args.episodes} scenes: seed={seed} "
                  f"base={row['base']['success']} demos={row['demos']['success']} varied={row['varied']['success']}",
                  flush=True)
