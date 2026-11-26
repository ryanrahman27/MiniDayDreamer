# Reproduce the RGB stacking experiment

The result in the demo comes from the calibrated release controller. The frozen flow policy takes 96 × 96 RGB images from the table and wrist cameras. Every three simulator ticks, the controller can keep the sampled noise or shift its gripper coordinate toward opening. The controller learned that decision from 24-tick rollouts inside the visual RSSM.

Run the commands from the repository root. To collect the demonstrations yourself and run the training and evaluation sequence, [install Isaac](../README.md#install-isaac) and start with a new directory.

```bash
bash scripts/reproduce_rgb.sh runs/from-scratch
```

The runner collects 72 successful demonstrations, 72 varied rollouts and 16 validation sequences. Behavior cloning uses the 72 demonstrations. The visual RSSM trains on 24 of them and the 72 varied rollouts. The runner then trains the earlier noise controllers, fits the release actor, and calibrates the RSSM’s predicted height change on development scenes. Each stage writes logs under its own run directory. [Collector](../scripts/rgb_isaac.py) · [Base policy](../scripts/fit_rgb.py) · [World model](../scripts/train_visual_dreams.py) · [Actor training](../scripts/rgb_wait_actor.py)

## Check your run

The calibrated actor advances to the confirmation bank if it beats the original fixed shift on development stacks, intervenes at least 4 mm lower at the median, and stays below 12 mm median horizontal error. Check `selection.json` for the decision. A new collection can change which weights qualify. After a qualifying run, count the stacks from its generated trial table.

```bash
python scripts/check_rgb_result.py runs/from-scratch/isaac-rgb-wait-v15-calibrated
```

Our confirmation bank contained 160 held-out scenes with two noise draws each. The base stacked in 97/320 trials, the original fixed shift in 146/320, and the fixed shift with the same visual trigger in 150/320. The learned controller stacked in 155/320. It gained 67 stacks and lost 9 against the base. The paired 95% interval for the gain was +12.5 to +24.1 percentage points, resampling scenes with both noise draws together. The interval against the matched fixed shift was −2.5 to +5.6 points. The demo end card uses the 164 paired trials in which either the base or learned controller stacked. [Evaluator](../scripts/rgb_score.py) · [Stack criterion](../scripts/score_retreat.py) · [Paired comparison](../scripts/summarize_rgb_wait.py)

## Score existing trajectories on CPU

We keep the original camera recordings, weights and raw trajectories outside Git. The from-scratch run above generates its own data. If you already have the local experiment bundle, set `assets` to its directory. Use the same directory for the remaining examples.

```bash
assets="$PWD/runs/rgb-stacking-v1"
(cd "$assets" && sha256sum --check SHA256SUMS)
python -m pip install -e .
mkdir -p runs/records
tar -xzf "$assets/rgb-confirmation-traces.tar.gz" -C runs/records
python scripts/summarize_rgb_wait.py test runs/records
python scripts/check_rgb_result.py runs/records
```

The scorer checks each terminal stack against the last eight ticks of cube positions, jaw apertures and speeds and rebuilds the summary for all 1,760 rollouts, including controls and repeats.

## Replay the frozen policies in Isaac

Use Linux with an RTX GPU. The machine used here ran Ubuntu 24.04, Python 3.12.3, Isaac Lab 3.0.0, Isaac Sim 6.2.0.0, PyTorch 2.11.0+cu130 and NumPy 2.5.5. The GPU was an RTX PRO 6000 Blackwell Max-Q, with driver 610.57.04. The Isaac installation uses NVIDIA’s version-matched overrides. [Setup commands](../README.md#install-isaac)

Use the archived resets from the preceding step. Keep the new outputs in a separate directory. The evaluator resumes completed trials, so pointing it at the archived results would skip work.

```bash
mkdir -p runs/replay
tar -xzf "$assets/rgb-steering-checkpoints.tar.gz" -C runs/replay
python scripts/rgb_isaac.py score runs/replay --failure-study calibrated \
  --label wait --split test --width 16 --batches 10 \
  --recorded-resets runs/records/test-wait
python scripts/summarize_rgb_wait.py test runs/replay
```

For a quick replay, add `--first-batch 0 --stop-batch 1` to the Isaac command. That runs all controllers and both noise draws on the first 16 scenes. Run the summary after all ten batches finish. The reset banks and initial images are shared across controllers. Later camera frames and contacts can vary on a replay, including repeats of the frozen base. [Evaluation code](../scripts/rgb_score.py)

## Refit the steering actor on CPU

The checkpoint bundle includes the frozen flow, RSSM, placement estimator, posterior starts, and release actors. The saved calibration came from 21 six-tick windows in the development recordings. Extract the bundle into a new directory so the published weights stay available for comparison.

```bash
mkdir -p runs/refit
tar -xzf "$assets/rgb-steering-checkpoints.tar.gz" -C runs/refit
python scripts/rgb_wait_actor.py runs/refit --seed 101 \
  --calibration runs/refit/dev-height-calibration.json
```

Each fit uses 1,290,240 imagined action ticks. The five published seeds are 101, 211, 307, 401 and 503. The actor file and training diagnostics are written to `runs/refit/wait-101.pt` and `wait-101.json` for the example above.

## Train from the camera recordings

The archive contains 72 successful demonstrations, 72 varied rollouts and 16 validation sequences. Each episode contains only `rgb` and `actions`. Behavior cloning uses the 72 demos. The RSSM uses 24 of them and the 72 varied rollouts. The scorer reads simulator state to measure the outcome.

In the Isaac environment, run the following to replay the training sequence, including the earlier controllers used as baselines.

```bash
mkdir -p runs/rgb-reproduction/isaac-rgb-actions-v12
tar --zstd -xf "$assets/rgb-actions.tar.zst" -C runs/rgb-reproduction/isaac-rgb-actions-v12
bash scripts/reproduce_rgb.sh runs/rgb-reproduction
```

## What we verified

The CPU scorer reconstructed the published summary from all 1,760 confirmation rollouts. Refitting the five actor seeds on the original host reproduced every saved actor weight exactly. We also replayed one 16-scene batch in Isaac with the packaged controllers and checked the terminal stack labels against all 176 new trajectories. The full base-policy training and ten-batch simulator evaluation were part of the original experiment.
