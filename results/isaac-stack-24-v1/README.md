# Isaac cube stacking with 24 demonstrations

Noise-space RL did not improve held-out stacking in this run. The frozen imitation policy finished 2 of 50 stacks. Steering with the demos-only RSSM finished 1, and steering with the varied-rollout RSSM finished 2. All three policies grasped and lifted the red cube more than 2 cm on every scene.

| Policy | Stable stacks | Success rate | Successful scene seeds | Median final horizontal separation |
| --- | ---: | ---: | --- | ---: |
| Frozen imitation policy | 2/50 | 4% | 127, 149 | 68.8 mm |
| Noise steering, demos-only RSSM | 1/50 | 2% | 154 | 57.4 mm |
| Noise steering, varied-rollout RSSM | 2/50 | 4% | 127, 149 | 75.2 mm |

The demos-only policy gained scene 154 and lost both base-policy successes. The varied-rollout policy matched the base outcome on every scene. The low success counts and single training seed limit what this experiment can establish about other training runs. [Paired outcomes for all 150 trials](paired-outcomes.csv) · [Full metrics](summary.json)

## What the models learned from

- **Base policy.** A conditional flow policy learned three-step action chunks from 24 successful scripted demonstrations in Isaac (seeds 0–23, 3,764 transitions). Checkpoints at 3,000, 6,000 and 12,000 updates scored 0/8 stable stacks on separate calibration scenes (seeds 10108–10115). The prespecified fallback selected 12,000 updates. That checkpoint lifted the cube on 8/8 calibration scenes.
- **World models.** The demos-only RSSM used those same 24 demonstrations. The varied-rollout RSSM also used 72 episodes (seeds 24–95), with 18 successful stacks, 18 missed grasps, 18 off-center placements and 18 early drops. This comparison changes both the number and the mix of trajectories.
- **Steering.** Both conditions used identical frozen base weights, 1,800 RSSM training updates and 1,500 soft actor-critic updates with training seed 0. The steering actor shifted the noise entering the base policy. All RL transitions came from six-step imagined rollouts initialized from the same 24 demonstration histories.
- **Scoring.** The three policies used the same 50 held-out scene seeds (108–157) and matched Gaussian noise, with a rotating execution order. A stack required the red cube centered on blue, a clear open gripper and low cube velocities for eight consecutive ticks. The policies used simulator state. RGB recorded the first test scene.

## Held-out world-model accuracy

Forecasts used 24 diagnostic trajectories from 12 additional scenes (seeds 96–107). The table gives mean Euclidean error in the red cube's predicted 3D position after six recorded actions. A stationary forecast keeps the cube at its last observed position.

| Forecast | All 707 windows | 272 windows with at least 5 mm of cube motion |
| --- | ---: | ---: |
| Stationary cube | 9.9 mm | 25.5 mm |
| Demos-only RSSM | 64.6 mm | 83.5 mm |
| Varied-rollout RSSM | 48.4 mm | 65.9 mm |

The extra rollouts reduced forecast error on both sets of windows. The stationary forecast still had the lowest error even when the cube moved. Only one of 12 paired action sequences had a short-horizon return difference above the diagnostic threshold of 0.01, leaving too little evidence to judge reward ranking.

## Replay reproducibility

After scoring, we replayed scene 127 in a fresh process because both the base and the varied policy had stacked there. On two repeats, one with cameras enabled and one with cameras disabled, all three policies failed to finish the stack. The recorded starting positions matched the scored trial exactly. The cause of the outcome difference is unresolved. These two repeats are reported separately from the 50-scene test. [Original and repeat outcomes](scene-127-repeats.json)

The collection, training, and evaluation code used for this run is at commit `38bdbf6`. The corpus checksum and all calibration outcomes are in the full metrics. [Run instructions](../../README.md)
