# Cube stacking from table and wrist RGB

The frozen RGB flow policy stacked on **84 of 256 trials (32.8%)**. Five actors trained in the visual RSSM stacked on **423 of 1,280 trials (33.0%)**. The paired difference was **+0.23 percentage points**. The 95% bootstrap interval over physical scenes and steering seeds was **−4.0 to +4.6 points**. This experiment did not reproduce the 18.9-point steering gain observed with simulator-state inputs. The two experiments also differ in learned weights, reward and evaluation scenes. [Previous state-input experiment](../isaac-release-margin-v11/README.md)

| Controller | Stable stacks | Success | Paired change |
| --- | ---: | ---: | ---: |
| Frozen RGB flow policy | 84 / 256 | 32.8% | — |
| Steering seed 101 | 83 / 256 | 32.4% | −0.4 points |
| Steering seed 211 | 85 / 256 | 33.2% | +0.4 points |
| Steering seed 307 | 82 / 256 | 32.0% | −0.8 points |
| Steering seed 401 | 86 / 256 | 33.6% | +0.8 points |
| Steering seed 503 | 87 / 256 | 34.0% | +1.2 points |

The 128 fresh physical scenes each had two action-noise streams. Every actor shared the same frozen flow policy, initial camera images, physical reset and Gaussian draws as its unsteered comparison. Over the 1,280 matched comparisons, steering rescued 111 base failures and lost 108 base successes. The gain was +0.94 points on the first stream and −0.47 on the second. A bootstrap that keeps each of the eight lighting groups together and also resamples actor seeds gives a 95% interval of −3.0 to +3.5 points. [Results](summary.json) · [Paired rollout records](paired-outcomes.csv)

## Where stacking failed

The base grasped red on 239 of 256 trials and lifted it on 211. Of its 172 failed trials, 127 got as far as lifting. The steered actors also lifted on 1,072 of 1,280 trials and finished 423 stacks. These observations localize most failures to the period after pickup. They do not identify a specific release error.

Repeating the unsteered policy on the first noise stream produced 48/128 stacks versus 42/128 on the original pass. The two passes agreed on 108/128 outcomes. The scorer restored cube and robot state, locked the sampled materials and lighting, and used the same stored image for the first action chunk. Subsequent images came live from Isaac. Native reset images varied by up to 3.4 intensity levels on average per scene on the 0–255 scale. Differences on an individual paired rollout can therefore include replay variation. [Evaluation protocol](protocol.json)

## Training from RGB and actions

The complete model-training corpus contains just two arrays per episode, synchronized RGB and commands. Isaac state was used by the scripted demonstrator, to identify successful expert demonstrations, and to score physical outcomes. The flow policy, visual dynamics and steering learners used no pose labels, grasp-state targets or state-trained checkpoints.

- **Base policy.** A shared convolutional encoder reads 96 × 96 table and wrist RGB and learns 16 conditioning features through the flow-matching loss. The original 160-wide velocity network and eight-step flow solver generate three four-dimensional commands at a time. The encoder and velocity network learned jointly from 72 successful demonstrations. Update 7,500 was selected by action error on four separate expert sequences. Replacing each sequence’s images with another scene at the same tick raised normalized action MAE from 0.107 to 0.227. [Training code](../../scripts/fit_rgb.py) · [Image check](image-check.json)
- **Visual RSSM.** A new recurrent model learned from 24 demos and 72 varied rollouts. It predicts the frozen policy’s camera features and a visual estimate of demonstration progress. The progress estimator learned its targets from ordering and gripper-command transitions in expert demonstrations. On 16 validation recordings, 12-action open-loop visual feature MSE was 0.196, compared with 0.473 for retaining the initial features. Progress MAE was 0.071 versus 0.116. These are feature-space measurements. [Dynamics and steering training](../../scripts/train_visual_dreams.py) · [World-model diagnostics](world.json)
- **Steering.** Each seed received 1,500 SAC updates and 576,000 imagined action ticks. The reward used changes in predicted visual progress and a small bonus for high progress. The base and world-model parameters stayed frozen. In a separate post-training check, mean final visual progress was 0.98 on four stable validation stacks and 0.14 on twelve failed validation rollouts. The physical held-out score required eight released, aligned, slow ticks at episode end. [Reward audit](goal-audit.json)

## Reproduction

Run `bash scripts/run_rgb.sh runs/isaac-rgb-actions-v12` in the project’s Isaac Lab environment. The script collects camera data when needed, fits the RGB policy and RSSM, trains the five actors, and scores the paired held-out trials. The saved run contains the frame-and-action archive `rgb-actions.tar.zst`, the frozen `flow.pt` and `world.pt`, the five actor checkpoints, and `paired-records.tar.gz` with the physical banks and reset images. Extracting the frame-and-action archive into the run directory supplies the original `inputs/` corpus. [Experiment runner](../../scripts/run_rgb.sh) · [Policy checkpoint selection](bc.json)
