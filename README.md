# MiniDayDreamer

Can a weak imitation policy learn to stack a red cube on a blue cube by changing its input noise entirely inside a learned world model? The first experiment uses a simulated Franka arm in Isaac Lab. The RGB camera records what happened. The policy and recurrent state-space model (RSSM) initially use simulator state, so their success measures control under different cube poses. Testing robustness to camera appearance requires a visual encoder in a later experiment.

## Isaac pilot

A frozen baseline selected at 15/32 development stacks completed 60/100 fresh Isaac scenes without steering. Noise-space steering in the varied RSSM completed 51/100 with learned reward and 48/100 with direct state reward. [Development selection and paired steering comparison](results/isaac-midbase-v5/README.md)

Computing imagination rewards from predicted cube and gripper states gave 4/50 stacks with the original base and varied RSSM, compared with 1/50 using the learned reward and 2/50 from imitation alone. A separate 72-demo base stacked on 48/50 of the same scenes before steering and 47/50 with direct-reward steering. [Paired reward comparison, forecast checks and stronger baseline](results/isaac-state-reward-v4/README.md)

Extending imagination from six to 24 ticks yielded 0/50 stacks at both depths with demos-only data and 4/50 versus 3/50 with varied data. A paired Isaac intervention traced a common carry stall to weak target-directed commands. [Horizon comparison and carry-stall measurements](results/isaac-horizon-24ticks-v3/README.md)

The repaired RSSM beat a stationary cube forecast on 12 independent scenes. On 50 fresh stacking scenes, each repaired steering actor completed two stacks and the base completed one. A gripper intervention tested whether delaying release helped. [World-model repair and gripper-timing results](results/isaac-residual-24-v2/README.md)

On 50 held-out Isaac scenes, the frozen policy stacked 2/50 times, demos-only RSSM steering stacked 1/50, and varied-rollout RSSM steering stacked 2/50. All three lifted the cube on every scene. [Experiment results and paired outcomes](results/isaac-stack-24-v1/README.md)

The Franka visuomotor stacking scene randomizes cube placements, lighting and materials. The green cube is a distractor. The experiment ends successfully when the red cube stays centered above blue, both cubes move slowly, and the gripper is open and clear for eight control ticks.

For the RTX PRO 6000 Blackwell Max-Q rental, use Python 3.12, Isaac Lab 3.0.0 and Isaac Sim 6.2. On the Ubuntu 24.04 Vast PyTorch image, install the graphics libraries with `apt-get install libxt6t64 libglu1-mesa`. Activate a Python 3.12 environment, then install NVIDIA's tested package combination. [Isaac Lab installation](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)

```bash
uv pip install 'isaaclab[isaacsim]==3.0.0' \
  --overrides https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt \
  --extra-index-url https://pypi.nvidia.com --torch-backend cu130 --index-strategy unsafe-best-match
```

With the Isaac environment activated and MiniDayDreamer as the current directory:

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
