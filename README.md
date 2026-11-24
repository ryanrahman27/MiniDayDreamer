# MiniDayDreamer

Can a weak imitation policy learn to stack a red cube on a blue cube by changing its input noise entirely inside a learned world model? The first experiment uses a simulated Franka arm in Isaac Lab. The RGB camera records what happened. The policy and recurrent state-space model (RSSM) initially use simulator state, so their success measures control under different cube poses. Testing robustness to camera appearance requires a visual encoder in a later experiment.

## Isaac pilot

Use Isaac Lab **v2.3.2**, Isaac Sim **5.1.0** and its Python **3.11** environment. The Franka visuomotor stacking scene supplies randomized cube positions and orientations, lighting and surface materials. The green cube is a distractor. The experiment ends successfully when the red cube stays centered above blue, both cubes move slowly, and the gripper is open and clear for eight control ticks. See the [pinned installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html), [scene configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py) and [GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

With the Isaac environment activated and MiniDayDreamer as the current directory:

```bash
python -m pip install -e ".[video]"
python -m minidaydreamer preview --seed 0 --out runs/scene-0.mp4
python -m minidaydreamer preview --seed 1 --out runs/scene-1.mp4
python -m minidaydreamer collect --demos 24 --extra 72 --holdout 12 --out runs/isaac.npz
python -m minidaydreamer train --data runs/isaac.npz --coverage demos --out runs/demos.pt
python -m minidaydreamer train --data runs/isaac.npz --coverage varied --out runs/varied.pt
python -m minidaydreamer eval --model runs/demos.pt --episodes 50 --video-dir runs/demo-videos
python -m minidaydreamer eval --model runs/varied.pt --episodes 50 --video-dir runs/varied-videos
```

The preview records the scripted controller and prints whether it achieved a stable release. Check this on the target host before collecting the corpus. The collector keeps successful demonstrations for behavior cloning and includes scripted misses, bad placements, early drops and successful placements in the extra trajectories. Each held-out scene has two recorded attempts from the same initial condition. Evaluation uses fresh scene seeds, pairs policy noise seeds, and records the **first** scene for each policy irrespective of its outcome.

## Training and interpretation

The frozen base policy generates three Cartesian commands at a time with conditional flow matching. Both coverage conditions fit the base policy from the same successful episodes and random seed. The RSSM learns action-conditioned dynamics, reward and success from recorded sequences. A soft actor-critic learner selects a bounded four-dimensional shift to fresh per-step Gaussian noise. Its replay buffer contains only two-chunk imagined rollouts initialized from filtered training histories. The policy consumes 11 state values measured in the simulator during evaluation and decoded from the RSSM during imagination. The noise actor consumes the RSSM state in both settings.

The validation report ranks pairs of held-out action sequences by their short-horizon return and compares red-cube position error with a stationary-cube prediction. Check the informative-pair count when interpreting ranking accuracy. The evaluation report gives completed stacks and counts both gains and losses on matched scenes. A larger imagined reward alone cannot establish improved stacking.

## CPU smoke test

A small MuJoCo backend exercises the pipeline without Isaac. It uses an abstract grasp constraint and rigid-body release. Its checkpoints and scores are backend-specific. The offline trainer runs on CPU and does not load either simulator.

```bash
python -m pip install -e ".[smoke]"
python -m minidaydreamer collect --backend mujoco --demos 16 --extra 32 --holdout 6 --out runs/smoke.npz
python -m minidaydreamer train --data runs/smoke.npz --bc-steps 2000 --model-steps 700 --rl-steps 400 --out runs/smoke.pt
python -m minidaydreamer eval --backend mujoco --model runs/smoke.pt --episodes 32
```

The Isaac runtime, camera capture and physical grasp controller still require validation on an RTX host. The current experiment measures performance in simulation.
