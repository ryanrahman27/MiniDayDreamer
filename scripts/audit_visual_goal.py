"""After freezing the checkpoints, compare the visual goal score with Isaac success on validation recordings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from train_visual_dreams import ProgressHead

from minidaydreamer.vision import CameraEncoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("recordings", type=Path)
    args = parser.parse_args()
    encoder, progress = CameraEncoder().cuda().eval(), ProgressHead().cuda().eval()
    encoder.load_state_dict(
        torch.load(args.run / "flow.pt", weights_only=True)["encoder"]
    )
    progress.load_state_dict(
        torch.load(args.run / "world.pt", weights_only=True)["progress"]
    )
    rows = []
    for rgb in sorted((args.run / "inputs").glob("valid-*.npz")):
        with np.load(rgb) as record:
            endpoint = torch.from_numpy(record["rgb"][-8:]).cuda()
        score = float(progress(encoder(endpoint)).mean())
        with np.load(args.recordings / rgb.name) as audit:
            rows.append(
                {
                    "episode": rgb.stem,
                    "isaac_success": int(audit["success"]),
                    "variant": str(audit["variant"]),
                    "visual_progress": score,
                }
            )
    yes = np.array([r["visual_progress"] for r in rows if r["isaac_success"]])
    no = np.array([r["visual_progress"] for r in rows if not r["isaac_success"]])
    diff = yes[:, None] - no
    out = {
        "rows": rows,
        "stable_mean": float(yes.mean()),
        "failed_mean": float(no.mean()),
        "pair_ranking_accuracy": float((diff > 0).mean() + 0.5 * (diff == 0).mean()),
    }
    (args.run / "goal-audit.json").write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
