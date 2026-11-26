# MiniDayDreamer

Our frozen RGB policy could pick up a cube and carry it over the target, then lift it away again without opening the gripper. We trained a small controller inside a recurrent state-space model (RSSM) to decide when to shift the policy’s input noise toward release.

On 160 held-out Isaac scenes with two noise draws each, the frozen policy stacked **97/320** times (30.3%). With learned steering it stacked **155/320** times (48.4%). A fixed shift using the same visual trigger stacked **150/320** times (46.9%). The paired 95% interval was +12.5 to +24.1 points against the base and −2.5 to +5.6 against the matched fixed shift.

[Read the write-up](https://ryanrahman.org/preprints/minidaydreamer.html) · [Reproduce the RGB experiment](docs/rgb-stacking.md) · [Steering objective](scripts/rgb_wait_actor.py)

On an Isaac machine, run `bash scripts/reproduce_rgb.sh runs/from-scratch` to collect the demos, fit the models and run the paired evaluation.

## Earlier state-input experiments

<details>
<summary>Results from the state-input baselines and steering experiments</summary>

On 128 fresh Isaac scenes and two action-noise streams, release-margin steering averaged 78.7% released stacks, versus 59.8% for the frozen base, 75.5% for the previous tuned actor and 77.7% for the same correction trained with the old reward. The 18.9-point gain over the base held across all five steering seeds (95% interval +10.9 to +27.0). The matched reward difference was +0.9 points (−0.9 to +2.9). Off-center failures persisted while failures to open became less frequent. [Paired release measurements and fresh-seed comparison](results/isaac-release-margin-v11/README.md)

Across two action-noise streams on 100 fresh physical scenes, five newly trained DSRL actors averaged 74% released stacks versus 54% for the frozen base and 69.5% for the previous steering checkpoint. The gain over the base was 20.0 percentage points (95% interval +10.7 to +29.2). On 32 additional single-environment scenes the three scores were 87.5%, 59.4% and 90.6%, respectively. [Multi-seed tuning, from-scratch controls and paired validation](results/isaac-multiseed-v10/README.md)

On 100 fresh matched Isaac scenes, a retreat-reward DSRL actor trained in the frozen varied RSSM produced 81 released terminal stacks, versus 51 for the previous targeted actor and 63 for the frozen base. Requiring the policy to pull its own gripper 50 mm clear for eight ticks gave 73, 44 and 61 stacks, respectively. A separate withdrawal test found that the extra terminal placements persisted when the hand moved away. [Stack stability, gripper clearance and paired results](results/isaac-retreat-v9/README.md)

Across 300 archived frozen-base runs, the two most frequent failures were never opening after lift and first opening off-center. We trained a DSRL actor to descend, center and release using rewards from the frozen RSSM. On 100 fresh paired Isaac scenes it stacked 56 times, versus 55 with the original reward on identical training starts, 47 with the earlier DSRL actor and 63 for the frozen base. [Failure modes, reward design and paired outcomes](results/isaac-failure-dsrl-v8/README.md)

The steered cube kept climbing as its successful base partner leveled off or descended, typically 31–32 mm from the target in two archived failure cohorts. On 100 fresh scenes, switching steering off at 80 mm yielded 48 stacks, versus 45 at 12 mm and 53 for the frozen base. [Climb timing, trajectory plot and paired early-handoff results](results/isaac-approach-handoff-v7/README.md)

In 16 of 24 archived losses, direct steering kept holding the cube. On 100 new scenes, switching steering off when the cube first crossed over blue stacked 40/100 times, versus 36/100 with always-on steering and 51/100 for the frozen base. [Paired height and release measurements, and fresh handoff results](results/isaac-target-handoff-v6/README.md)

A frozen baseline selected at 15/32 development stacks completed 60/100 fresh Isaac scenes without steering. Noise-space steering in the varied RSSM completed 51/100 with learned reward and 48/100 with direct state reward. [Development selection and paired steering comparison](results/isaac-midbase-v5/README.md)

Computing imagination rewards from predicted cube and gripper states gave 4/50 stacks with the original base and varied RSSM, compared with 1/50 using the learned reward and 2/50 from imitation alone. A separate 72-demo base stacked on 48/50 of the same scenes before steering and 47/50 with direct-reward steering. [Paired reward comparison, forecast checks and stronger baseline](results/isaac-state-reward-v4/README.md)

Extending imagination from six to 24 ticks yielded 0/50 stacks at both depths with demos-only data and 4/50 versus 3/50 with varied data. A paired Isaac intervention traced a common carry stall to weak target-directed commands. [Horizon comparison and carry-stall measurements](results/isaac-horizon-24ticks-v3/README.md)

The repaired RSSM beat a stationary cube forecast on 12 independent scenes. On 50 fresh stacking scenes, each repaired steering actor completed two stacks and the base completed one. A gripper intervention tested whether delaying release helped. [World-model repair and gripper-timing results](results/isaac-residual-24-v2/README.md)

On 50 held-out Isaac scenes, the frozen policy stacked 2/50 times, demos-only RSSM steering stacked 1/50, and varied-rollout RSSM steering stacked 2/50. All three lifted the cube on every scene. [Experiment results and paired outcomes](results/isaac-stack-24-v1/README.md)

The Franka visuomotor stacking scene randomizes cube placements, lighting and materials. The green cube is a distractor. The current multi-seed study scores eight ticks of released, stable stacking at the end of the episode. The historical pilot commands below use the earlier early-stopping rule, which also required the tool to move 50 mm clear.

</details>

## Install Isaac

For the RTX PRO 6000 Blackwell Max-Q rental, use Python 3.12, Isaac Lab 3.0.0 and Isaac Sim 6.2. On the Ubuntu 24.04 Vast PyTorch image, install the graphics libraries with `apt-get install libxt6t64 libglu1-mesa`. Activate a Python 3.12 environment, then install NVIDIA's tested package combination. [Isaac Lab installation](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)

```bash
uv pip install -r requirements/isaac-rgb.txt \
  --overrides https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt \
  --extra-index-url https://pypi.nvidia.com --torch-backend cu130 --index-strategy unsafe-best-match
```

Install the package in the active Isaac environment with `uv pip install -e .`. The [RGB reproduction guide](docs/rgb-stacking.md) covers the camera experiment.

## Run the state-input baseline

From the repository root, run the following commands.

```bash
python -m pip install -e ".[video]"
python -m minidaydreamer preview --seed 0 --out runs/scene-0.mp4
python -m minidaydreamer preview --seed 1 --out runs/scene-1.mp4
run=runs/isaac-full-24
python -m minidaydreamer collect --demos 24 --extra 72 --holdout 12 --out "$run/corpus.npz"
python -m minidaydreamer calibrate --data "$run/corpus.npz" --out "$run/base.pt"
for arm in demos varied; do
  python -m minidaydreamer train --data "$run/corpus.npz" --base-model "$run/base.pt" --coverage "$arm" --out "$run/$arm.pt"
done
python -m minidaydreamer compare --demos-model "$run/demos.pt" --varied-model "$run/varied.pt" --episodes 50 --out "$run/eval-0.jsonl"
python scripts/summarize_stack.py "$run"
```

The collector keeps successful demonstrations for behavior cloning and includes scripted misses, bad placements, early drops and attempted successful placements in the extra trajectories. Each world-model validation scene has two recorded attempts from the same reset seed. Calibration evaluates BC checkpoints in order of 3,000, 6,000 and 12,000 updates. It selects the first that stacks at least once on eight development scenes, or the last if none succeeds. These scenes are disjoint from training and final evaluation.

The comparison uses fresh scene seeds, paired Gaussian noise, and one JSON record per scene for all three policies. For parallel runs, set `--scene-offset` and `--episodes` to disjoint intervals and write each shard to `eval-<offset>.jsonl`. Add `--video-dir` to record the first seed of that invocation. State-only runs skip the two camera sensors. The task, physics and reset events are shared across runs.

## Training and interpretation

The frozen base policy generates three Cartesian and gripper commands at a time with conditional flow matching. Both coverage conditions load the selected base checkpoint, and the comparison checks that their base weights match. The default residual RSSM carries the measured positions and latest displacement through each filter update. It learns action-conditioned position changes, grasp state, reward and success with one-step and six-step open-loop losses. The commanded gripper state follows the issued action. Use `train --world-model absolute` to reproduce the original absolute-position decoder. Previously saved checkpoints load with their original architecture.

A soft actor-critic learner selects a bounded four-dimensional shift to fresh per-step Gaussian noise. Imagined rollouts start from demonstration histories. The default depth is two three-tick chunks. The training option `--imagination-chunks` changes the depth while keeping 128 generated chunk transitions per update. The base policy consumes 11 state values measured in the simulator during evaluation and predicted by the RSSM during imagination. The noise actor consumes the RSSM state in both settings.

The validation report ranks pairs of held-out action sequences by their short-horizon return and compares red-cube position error with a stationary-cube prediction. Check the informative-pair count when interpreting ranking accuracy. The evaluation report gives completed stacks and counts both gains and losses on matched scenes. A larger imagined reward alone cannot establish improved stacking.

## CPU smoke test

A small MuJoCo backend exercises the pipeline without Isaac. It uses an abstract grasp constraint and rigid-body release. Its checkpoints and scores are backend-specific. The offline trainer runs on CPU and does not load either simulator.

```bash
python -m pip install -e ".[smoke]"
python -m minidaydreamer collect --backend mujoco --demos 16 --extra 32 --holdout 6 --out runs/smoke.npz
python -m minidaydreamer train --data runs/smoke.npz --bc-steps 2000 --model-steps 700 --rl-steps 400 --out runs/smoke.pt
python -m minidaydreamer eval --backend mujoco --model runs/smoke.pt --episodes 32
```

The scripted controller completed and released stable stacks on seeds 0 and 1 on the RTX rental. The table camera recorded 640 × 480 video at 20 fps.
