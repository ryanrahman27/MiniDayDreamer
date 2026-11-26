"""Evaluate a fixed camera encoder with frozen flow, RSSM and noise actors in matched scenes."""

import copy
import hashlib
import json
import os
import random
import tempfile

import numpy as np
import torch
from rgb_isaac import pixels, speeds, step
from score_multiseed import extract, summarize_trace

from minidaydreamer.models import ACTION_DIM, HORIZON, NoiseSAC
from minidaydreamer.vision import CameraEncoder, VisualFlowHead
from minidaydreamer.visual_rssm import VisualRSSM


def preserve_appearance(env, env_ids, **params):
    """A paired reset keeps the material and dome light sampled for this group."""


@torch.inference_mode()
def score(env, args):
    ck = args.run / "flow.pt"
    base = torch.load(ck, weights_only=True, map_location="cpu")
    encoder, pi = CameraEncoder().cuda().eval(), VisualFlowHead().eval()
    encoder.load_state_dict(base["encoder"])
    pi.load_state_dict(base["flow"])
    base_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    width = args.width
    scores = args.run / f"{args.split}-{args.label}"
    scores.mkdir(exist_ok=True)
    seeds = (101,) if args.split == "dev" else (101, 211, 307, 401, 503)
    models = {}
    study = args.failure_study
    wait_study = study in ("wait", "calibrated")
    release_study = study in ("release", "wait", "calibrated")
    if release_study:
        from rgb_release_control import CONFIGS, SEEDS, ReleaseControl, configure

        seeds = SEEDS
    if wait_study:
        from rgb_wait_control import LIMITS, WaitControl, configure_wait
    if study == "replay":
        if args.recorded_resets is None:
            raise ValueError("Replay requires the archived reset directory")
        conditions = ["rgb-base"]
    elif args.label.startswith("check"):
        conditions = ["rgb-base", "rgb-repeat"]
    else:
        from train_visual_dreams import ProgressHead

        world = torch.load(args.run / "world.pt", weights_only=True, map_location="cpu")
        wm = VisualRSSM().eval()
        wm.load_state_dict(world["rssm"])
        head = ProgressHead().eval()
        head.load_state_dict(world["progress"])
        assert world["base_sha256"] == base_sha
        conditions = ["rgb-base"]
        for seed in () if wait_study else seeds:
            arm = f"rgb-dsrl-{seed}"
            saved = torch.load(args.run / f"actor-{seed}.pt", weights_only=True)
            assert (
                saved["base_sha256"] == base_sha
                and saved["world_sha256"]
                == hashlib.sha256((args.run / "world.pt").read_bytes()).hexdigest()
            )
            models[arm] = NoiseSAC(wm.features)
            models[arm].actor.load_state_dict(saved["actor"])
            conditions.append(arm)
        if study in ("fresh", "release"):
            from rgb_failure_actor import load_target

            for seed in seeds:
                models[f"target-{seed}"] = load_target(args.run, wm, seed, base_sha)
            conditions = ["rgb-base", "rgb-incumbent", "rgb-target"]
        if wait_study:
            geometry, release_actors, wait_actors, conditions, selected = (
                configure_wait(
                    args.run, args.split, wm, calibrated=study == "calibrated"
                )
            )
        elif release_study:
            geometry, release_actors, conditions, selected = configure(
                args.run, args.split, wm
            )
    streams = (0,) if args.split == "dev" else (0, 1000003)
    plan = {
        "base_sha256": base_sha,
        "actors": conditions,
        "noise_offsets": streams,
        "width": width,
        "batches": args.batches,
        "inputs": ["table_rgb", "wrist_rgb"],
        "repeat_base_first_stream": study != "replay"
        and not args.label.startswith("check"),
        "initial_image": "shared RGB from the matched reset; live cameras after first action",
        "score": "final 8 ticks released, aligned and slow",
    }
    if study:
        plan.update(failure_study=study, actor_seeds=list(seeds), task_ticks=200)
        if release_study:
            plan.update(
                first_scene=810000 if args.split == "dev" else 910000,
                configs=CONFIGS,
                selected=selected,
                geometry_sha256=hashlib.sha256(
                    (args.run / "release-geometry.pt").read_bytes()
                ).hexdigest(),
                seed_assignment="scene offset modulo 5",
                repeat_base_first_stream=args.split == "test",
            )
            if wait_study:
                plan.update(
                    first_scene=1010000 if args.split == "dev" else 1110000,
                    limits=LIMITS,
                    placement_sha256=hashlib.sha256(
                        (args.run / "placement-view.pt").read_bytes()
                    ).hexdigest(),
                )
                if study == "calibrated":
                    plan.update(
                        first_scene=1210000 if args.split == "dev" else 1310000,
                        height_calibration=json.loads(
                            (args.run / "dev-height-calibration.json").read_text()
                        ),
                    )
        if study == "fresh":
            plan.update(
                first_scene=730000,
                seed_assignment="scene offset modulo 5; paired actors share seed",
            )
    manifest = scores / "protocol.json"
    plan = json.loads(json.dumps(plan))
    # Test shards share a directory. Publish the complete protocol before a
    # second process can open it, and reject resumes with different settings.
    if not manifest.exists():
        with tempfile.NamedTemporaryFile(dir=scores, delete=False, mode="w") as tmp:
            json.dump(plan, tmp, indent=2)
        try:
            os.link(tmp.name, manifest)
        except FileExistsError:
            pass
        finally:
            os.unlink(tmp.name)
    if json.loads(manifest.read_text()) != plan:
        raise ValueError(f"Protocol differs from {manifest}")
    raw = env.env.unwrapped
    visual_events = {
        name: copy.copy(raw.event_manager.get_term_cfg(name))
        for name in ("randomize_light", "randomize_table_visual_material")
    }
    stop = args.batches if args.stop_batch is None else args.stop_batch
    if not 0 <= args.first_batch < stop <= args.batches:
        raise ValueError("Invalid evaluation shard")
    for b in range(args.first_batch, stop):
        group = (520000 if args.split == "dev" else 620000) + b * width
        if study == "fresh":
            group = 730000 + b * width
        if release_study:
            group = plan["first_scene"] + b * width
        for name, cfg in visual_events.items():
            raw.event_manager.set_term_cfg(name, cfg)
        random.seed(group)
        np.random.seed(group)
        torch.manual_seed(group)
        env.env.reset(seed=group)
        for name, cfg in visual_events.items():
            locked = copy.copy(cfg)
            locked.func = preserve_appearance
            raw.event_manager.set_term_cfg(name, locked)
        physical = raw.scene.get_state(is_relative=True)
        bank = {
            kind: {
                asset: {key: value.clone() for key, value in fields.items()}
                for asset, fields in assets.items()
            }
            for kind, assets in physical.items()
        }
        reset_dir = args.recorded_resets or scores
        bank_file = reset_dir / f"bank-{group}.pt"
        reference = reset_dir / f"reference-{group}.npz"
        if args.recorded_resets is not None and not (bank_file.is_file() and reference.is_file()):
            raise FileNotFoundError(f"Missing archived reset for {group}")
        if bank_file.exists():
            bank = torch.load(bank_file, weights_only=True, map_location=raw.device)
        else:
            torch.save(bank, bank_file)
        ref_pose = ref_image = None
        if reference.exists():
            with np.load(reference) as saved:
                ref_pose = torch.from_numpy(saved["pose"])
                ref_image = torch.from_numpy(saved["rgb"]).to(raw.device)
        for off in streams:
            names = conditions.copy()
            if plan["repeat_base_first_stream"] and off == 0:
                names.append("rgb-repeat")
            rotate = (b + int(off > 0)) % len(names)
            for arm in names[rotate:] + names[:rotate]:
                path = scores / f"{group}-{off}-{arm}.json"
                # Always re-run reset when resuming; this also recreates the visual matching audit.
                obs, _ = raw.reset_to(bank, seed=group, is_relative=True)
                # RTX cameras lag the reset buffers until physics advances. Let the
                # same open, stationary command settle before the 200 scored ticks.
                neutral = torch.zeros(width, 4)
                neutral[:, 3] = -1
                for _ in range(6):
                    obs = step(env, neutral)
                # Reinstate the exact bank after flushing the camera pipeline.
                # The warm-up is outside the scored trajectory.
                obs, _ = raw.reset_to(bank, seed=group, is_relative=True)
                frame = pixels(obs)
                closed = torch.zeros(width, device=raw.device)
                truth, jaw = extract(env, obs, closed)
                if ref_pose is None:
                    ref_pose, ref_image = truth.clone(), frame.clone()
                    np.savez_compressed(
                        reference, pose=ref_pose.numpy(), rgb=ref_image.cpu().numpy()
                    )
                pose_delta = float((truth - ref_pose).abs().max())
                if pose_delta > 0.01:
                    raise RuntimeError(f"Mismatched physical reset {pose_delta}")
                pixel_diff = (
                    (frame.float() - ref_image.float())
                    .abs()
                    .mean((1, 2, 3))
                    .cpu()
                    .tolist()
                )
                # Identical stored pixels condition every arm at tick zero. This
                # guard catches a gross camera reset failure (observed MAE ~30)
                # while retaining native renderer variation in the audit.
                if max(pixel_diff) > 8:
                    np.savez_compressed(
                        scores / "visual-reset-diagnostic.npz",
                        before=ref_image.cpu().numpy(),
                        after=frame.cpu().numpy(),
                    )
                    raise RuntimeError(f"Unmatched initial images: {pixel_diff}")
                if path.exists():
                    continue
                initial = truth.clone()
                gens = [
                    torch.Generator().manual_seed(group + i + off) for i in range(width)
                ]
                actor = models.get(arm)
                lane_actors = arm in ("rgb-incumbent", "rgb-target")
                control = (
                    ReleaseControl(arm, width)
                    if release_study and arm.startswith(("release-", "fixed-"))
                    else None
                )
                waiting = (
                    WaitControl(arm, width)
                    if wait_study and arm.startswith(("wait-", "lower-"))
                    else None
                )
                if waiting:
                    control = waiting
                lane_seeds = [
                    seeds[(group + i - plan.get("first_scene", 730000)) % len(seeds)]
                    for i in range(width)
                ]
                seed_tensor = torch.tensor(lane_seeds)
                rgb = ref_image
                release_log = []
                placement_log = []
                condition = encoder(ref_image).cpu()
                latent = (
                    wm.initial(torch.cat((condition, head(condition)), 1))
                    if actor or lane_actors or control
                    else None
                )
                trace, jaws, vel = [truth.numpy()], [jaw], []
                commands = []
                feature_trace = (
                    [condition.numpy().astype(np.float16)]
                    if study == "fresh" or release_study
                    else []
                )
                for t in range(env.max_steps):
                    if t % HORIZON == 0:
                        noise = torch.cat(
                            [
                                torch.randn(1, HORIZON, ACTION_DIM, generator=g)
                                for g in gens
                            ]
                        )
                        if actor:
                            noise += actor.draw(latent, deterministic=True)[0][:, None]
                        if lane_actors:
                            prefix = "target" if arm == "rgb-target" else "rgb-dsrl"
                            for seed in seeds:
                                ids = [i for i, s in enumerate(lane_seeds) if s == seed]
                                noise[ids] += models[f"{prefix}-{seed}"].draw(
                                    latent[ids], deterministic=True
                                )[0][:, None]
                        if release_study:
                            if wait_study:
                                ready, coords, visible = (
                                    v.cpu() for v in geometry(rgb)
                                )
                                advantage, eligible = (
                                    torch.zeros(width),
                                    torch.zeros(width),
                                )
                            else:
                                ready = geometry(rgb).cpu()
                            if waiting:
                                chunk, started, shifted, advantage, eligible = (
                                    waiting.sample(
                                        pi,
                                        condition,
                                        noise,
                                        ready,
                                        coords,
                                        visible,
                                        wait_actors,
                                        seed_tensor,
                                    )
                                )
                            elif control:
                                chunk, started, shifted = control.sample(
                                    pi,
                                    condition,
                                    noise,
                                    ready,
                                    latent,
                                    release_actors,
                                    seed_tensor,
                                )
                            else:
                                chunk = pi.sample(condition, noise)
                                started, shifted = (
                                    torch.zeros(width),
                                    torch.zeros(width),
                                )
                            release_log.append(
                                torch.stack((ready, started, shifted), 1).numpy()
                            )
                            if wait_study:
                                placement_log.append(
                                    torch.column_stack(
                                        (coords, visible, advantage, eligible)
                                    ).numpy()
                                )
                        else:
                            chunk = pi.sample(condition, noise)
                    act = chunk[:, t % HORIZON]
                    if control:
                        control.observe(act)
                    if study:
                        commands.append(act.numpy().copy())
                    obs = step(env, act)
                    truth, jaw = extract(env, obs, act[:, 3].to(raw.device) > 0)
                    rgb = pixels(obs)
                    condition = encoder(rgb).cpu()
                    if study == "fresh" or release_study:
                        feature_trace.append(condition.numpy().astype(np.float16))
                    if actor or lane_actors or control:
                        latent, _ = wm.filter(
                            wm.advance(latent, act),
                            torch.cat((condition, head(condition)), 1),
                        )
                    trace.append(truth.numpy())
                    jaws.append(jaw)
                    vel.append(speeds(env))
                trace, jaws, vel = map(np.stack, (trace, jaws, vel))
                if study:
                    tmp_trace = path.with_suffix(".tmp.npz")
                    features = (
                        {"visual_features": np.stack(feature_trace)}
                        if feature_trace
                        else {}
                    )
                    np.savez_compressed(
                        tmp_trace,
                        obs=trace,
                        actions=np.stack(commands),
                        jaws=jaws,
                        speeds=vel,
                        **(
                            {"release_decisions": np.stack(release_log)}
                            if release_study
                            else {}
                        ),
                        **features,
                        **(
                            {"placement_decisions": np.stack(placement_log)}
                            if wait_study
                            else {}
                        ),
                    )
                    tmp_trace.replace(path.with_suffix(".npz"))
                rows = []
                for i in range(width):
                    fallen = np.flatnonzero(trace[1:, i, [5, 8]].min(axis=-1) < -0.5)
                    end = int(fallen[0]) + 1 if len(fallen) else env.max_steps
                    measured = trace[: end + 1, i]
                    score = summarize_trace(measured, jaws[: end + 1, i], vel[:end, i])
                    score.update(
                        scene=group + i,
                        batch_seed=group,
                        lane=i,
                        noise_seed=group + i + off,
                        arm=arm,
                        initial=initial[i].tolist(),
                        reset_rgb_mae=pixel_diff[i],
                        grasped=bool(measured[:, 9].max() > 0.5),
                        lifted=bool(
                            ((measured[:, 9] > 0.5) & (measured[:, 5] > 0.65)).any()
                        ),
                    )
                    rows.append(score)
                    if study == "fresh" or release_study:
                        score["actor_seed"] = lane_seeds[i]
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(rows))
                tmp.replace(path)
                print(
                    group,
                    off,
                    arm,
                    sum(r["success"] for r in rows),
                    "/",
                    width,
                    flush=True,
                )
