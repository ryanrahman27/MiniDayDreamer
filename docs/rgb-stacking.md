# Reproduce the RGB stacking experiment

The result in the demo comes from the calibrated release controller. The frozen flow policy takes 96 × 96 RGB images from the table and wrist cameras. Every three simulator ticks, the controller can keep the sampled noise or shift its gripper coordinate toward opening. The controller learned that decision from 24-tick rollouts inside the visual RSSM.

Run the commands below from the repository root. The checkpoints, camera data and trajectories are attached to the [RGB stacking release](https://github.com/ryanrahman27/MiniDayDreamer/releases/tag/rgb-stacking-v1).

## Check the reported result

This reads the checked-in CSV with Python’s standard library.

```bash
python scripts/check_rgb_result.py
```

The base stacked in 97/320 trials, the fixed shift in 146/320, the fixed shift with the same visual trigger in 150/320, and the learned controller in 155/320. The learned controller gained 67 stacks and lost 9 against the base. The paired 95% interval for the gain is +12.5 to +24.1 percentage points, resampling 160 scenes with both noise draws together. The five-stack advantage over the matched fixed shift has an interval that includes zero. The video's end card uses the 164 paired trials in which either the base or learned controller stacked. [Per-trial data](../results/isaac-rgb-wait-v15/calibrated/test-outcomes.csv) · [Paired analysis](../results/isaac-rgb-wait-v15/calibrated/summary.json)

To recompute the scores from recorded cube positions, jaw apertures and speeds, install the project and unpack the confirmation trajectories. This part runs on CPU. The scorer checks each terminal stack against the last eight ticks and rebuilds the summary for all 1,760 rollouts, including controls and repeats.

```bash
python -m pip install -e .
mkdir -p runs/downloads runs/records
gh release download rgb-stacking-v1 -p rgb-confirmation-traces.tar.gz -D runs/downloads
(cd runs/downloads && sha256sum --check --ignore-missing ../../results/isaac-rgb-wait-v15/SHA256SUMS)
tar -xzf runs/downloads/rgb-confirmation-traces.tar.gz -C runs/records
python scripts/summarize_rgb_wait.py test runs/records
```

## Replay the frozen policies in Isaac

Use Linux with an RTX GPU. The machine used here ran Ubuntu 24.04, Python 3.12.3, Isaac Lab 3.0.0, Isaac Sim 6.2.0.0, PyTorch 2.11.0+cu130 and NumPy 2.5.5. The GPU was an RTX PRO 6000 Blackwell Max-Q, with driver 610.57.04. The Isaac installation uses NVIDIA’s version-matched overrides. [Setup commands](../README.md#install-isaac)

Use the archived resets from the preceding step. Keep the new outputs in a separate directory. The evaluator resumes completed trials, so pointing it at the archived results would skip work.

```bash
mkdir -p runs/replay
gh release download rgb-stacking-v1 -p rgb-steering-checkpoints.tar.gz -D runs/downloads
(cd runs/downloads && sha256sum --check --ignore-missing ../../results/isaac-rgb-wait-v15/SHA256SUMS)
tar -xzf runs/downloads/rgb-steering-checkpoints.tar.gz -C runs/replay
python scripts/rgb_isaac.py score runs/replay --failure-study calibrated \
  --label wait --split test --width 16 --batches 10 \
  --recorded-resets runs/records/test-wait
python scripts/summarize_rgb_wait.py test runs/replay
```

For a quick replay, add `--first-batch 0 --stop-batch 1` to the Isaac command. That runs all controllers and both noise draws on the first 16 scenes. Run the summary after all ten batches finish. The reset banks and initial images are shared across controllers. Later camera frames and contacts can vary on a replay, including repeats of the frozen base. [Evaluation protocol](../results/isaac-rgb-wait-v15/calibrated/protocol.json)

## Refit the steering actor on CPU

The checkpoint bundle includes the frozen flow, RSSM, placement estimator, posterior starts, and release actors. The saved calibration came from 21 six-tick windows in the development recordings. Extract the bundle into a new directory so the published weights stay available for comparison.

```bash
mkdir -p runs/refit
tar -xzf runs/downloads/rgb-steering-checkpoints.tar.gz -C runs/refit
python scripts/rgb_wait_actor.py runs/refit --seed 101 \
  --calibration runs/refit/dev-height-calibration.json
```

Each fit uses 1,290,240 imagined action ticks. The five published seeds are 101, 211, 307, 401 and 503. The actor file and training diagnostics are written to `runs/refit/wait-101.pt` and `wait-101.json` for the example above.

## Train from the camera recordings

The archive contains 72 successful demonstrations, 72 varied rollouts and 16 validation sequences. Each episode contains only `rgb` and `actions`. Behavior cloning uses the 72 demos. The RSSM uses 24 of them and the 72 varied rollouts. The scorer reads simulator state to measure the outcome.

In the Isaac environment, run the following to replay the training sequence, including the earlier controllers used as baselines.

```bash
mkdir -p runs/downloads runs/rgb-reproduction/isaac-rgb-actions-v12
gh release download rgb-stacking-v1 -p rgb-actions.tar.zst -D runs/downloads
(cd runs/downloads && sha256sum --check --ignore-missing ../../results/isaac-rgb-wait-v15/SHA256SUMS)
tar --zstd -xf runs/downloads/rgb-actions.tar.zst -C runs/rgb-reproduction/isaac-rgb-actions-v12
bash scripts/reproduce_rgb.sh runs/rgb-reproduction
```

To collect new demonstrations in Isaac as well, run `bash scripts/reproduce_rgb.sh runs/from-scratch` with a new directory. The runner collects data, fits the RGB policy and RSSM, trains the noise controllers, fits the height correction on development rollouts, and evaluates the calibrated actor if it passes the original development criterion. Each stage writes logs under its own run directory. A new collection or GPU build can change which checkpoint qualifies. [Selection rule](../results/isaac-rgb-wait-v15/calibrated/selection.json)
