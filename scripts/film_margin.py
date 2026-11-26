"""Record a scored camera replay illustrating a base loss and a rescued stack."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from margin_study import controllers_for, dump, prior_name
from minidaydreamer.experiment import policy_episode
from score_multiseed import reset_lanes


def montage(folder, seed, name):
    label = "Release-margin steering" if name.startswith("margin") else "Tuned noise steering"
    files = [folder / f"{n}-{seed}.mp4" for n in ("base", name)]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    caption = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    dest = folder / "paired-stack.mp4"
    with (
        imageio.get_reader(files[0]) as left,
        imageio.get_reader(files[1]) as right,
        imageio.get_writer(dest, fps=20, codec="libx264", quality=8) as writer,
    ):
        total = min(left.count_frames(), right.count_frames())
        for i in range(total + 28):
            j = min(i, total - 1)
            a, b = left.get_data(j), right.get_data(j)
            h, w = a.shape[:2]
            canvas = Image.new("RGB", (2 * w, h + 80), (19, 27, 34))
            canvas.paste(Image.fromarray(a), (0, 52))
            canvas.paste(Image.fromarray(b), (w, 52))
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 11), "Frozen imitation policy", font=font, fill="white")
            draw.text((w + 18, 11), label, font=font, fill=(129, 231, 199))
            if i >= total - 1:
                draw.text((18, h + 55), "Finished outside the stack criterion", font=caption, fill="white")
                draw.text(
                    (w + 18, h + 55),
                    "Finished with a stable, released stack",
                    font=caption,
                    fill=(129, 231, 199),
                )
            else:
                draw.text(
                    (18, h + 55),
                    f"Isaac Lab  /  paired camera replay  /  scene {seed}",
                    font=caption,
                    fill="white",
                )
                draw.text(
                    (w + 18, h + 55),
                    f"Same scene and Gaussian draws   ·   {(j + 1) / 20:.1f} s",
                    font=caption,
                    fill="white",
                )
            writer.append_data(np.array(canvas))
    return total


def film(env, run, old, name):
    folder = run / ("video-margin" if name.startswith("margin-") else "video")
    folder.mkdir(exist_ok=True)
    pi, wm, models = controllers_for(run, old, [101])
    bank = torch.load(run / "scenes.pt", weights_only=True)
    rows = sorted(
        (
            json.loads(line)
            for f in (run / "test").glob("eval-*.jsonl")
            for line in f.read_text().splitlines()
        ),
        key=lambda r: r["seed"],
    )
    candidates = [r["seed"] for r in rows if not r["base"]["success"] and r[name]["success"]]
    attempts = []
    for seed in candidates[:8]:
        flags = []
        for arm in ("base", name):
            env.initial_bank = bank
            result, start = policy_episode(
                env, pi, wm if arm != "base" else None, models[arm], seed, video=folder / f"{arm}-{seed}.mp4"
            )
            np.testing.assert_allclose(start, bank[seed][0].numpy(), rtol=0, atol=0.01)
            flags.append(int(result["success"]))
        attempts.append(dict(seed=seed, base=flags[0], steered=flags[1]))
        if flags == [0, 1]:
            frames = montage(folder, seed, name)
            dump(
                folder / "caption.json",
                dict(
                    arm=name,
                    seed=seed,
                    frames=frames,
                    attempts=attempts,
                    selection="First paired primary base loss and steering win reproduced in the camera environment. Illustrative, excluded from study rates.",
                ),
            )
            return
    raise RuntimeError(f"No camera replay reproduced the selected pair: {attempts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("--arm", choices=(prior_name(101), "margin-101"), default=prior_name(101))
    a = p.parse_args()
    torch.set_num_threads(1)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True)
    try:
        from minidaydreamer.isaac import IsaacStack

        class CameraScene(IsaacStack):
            def reset(self, seed):
                super().reset(seed)
                return self.state(reset_lanes(self, self.initial_bank, [seed]))

        env = CameraScene(cameras=True, released_at_finish=True)
        try:
            film(env, a.run, a.old, a.arm)
        finally:
            env.close()
    finally:
        app.app.close()
