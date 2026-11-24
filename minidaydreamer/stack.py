"""Headless two-cube MuJoCo task with an abstract Cartesian gripper."""

import numpy as np


SIZE = 0.04  # cube edge in metres
STEP = 0.025
STATE_DIM = 11
ACTION_DIM = 4
MAX_STEPS = 65
XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" iterations="60"/>
  <worldbody>
    <geom type="plane" size=".4 .4 .01" rgba=".5 .5 .5 1"/>
    <body name="tool" mocap="true" pos="0 0 .15">
      <geom type="sphere" size=".008" contype="0" conaffinity="0" rgba="1 1 1 .5"/>
    </body>
    <body name="red" pos="-.07 0 .02"><freejoint/>
      <geom type="box" size=".02 .02 .02" mass=".06" rgba="1 .1 .1 1"/>
    </body>
    <body name="blue" pos=".07 0 .02"><freejoint/>
      <geom type="box" size=".02 .02 .02" mass=".06" rgba=".1 .2 1 1"/>
    </body>
  </worldbody>
  <equality><weld name="grasp" body1="tool" body2="red" active="false"
                  relpose="0 0 0 1 0 0 0" solref=".005 1"/></equality>
</mujoco>
"""


class CubeStack:
    """State is tool/red/blue xyz (in 10 cm units), attached flag and closed flag."""

    backend = "mujoco"
    max_steps = MAX_STEPS
    block_height = SIZE
    grip_delay = 1

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

    def observe(self):
        d = self.data
        positions = np.concatenate((d.mocap_pos[0], d.xpos[2], d.xpos[3])) / 0.1
        return np.r_[positions, float(d.eq_active[0]), float(self.closed)].astype(np.float32)

    def reset(self, seed):
        mujoco = self.mj
        d = self.data
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, d)
        for start, side in ((0, -1), (7, 1)):
            d.qpos[start:start + 3] = [side * rng.uniform(.045, .105), rng.uniform(-.065, .065), SIZE / 2]
        d.mocap_pos[0] = [rng.uniform(-.04, .04), rng.uniform(-.04, .04), .15]
        self.closed = False
        self.steps = self.stable = 0
        mujoco.mj_forward(self.model, d)
        return self.observe()

    def step(self, action):
        mujoco = self.mj
        d = self.data
        a = np.clip(action, -1, 1)
        next_tool = np.clip(d.mocap_pos[0] + STEP * a[:3], [-.19, -.19, .019], [.19, .19, .21])
        closed = bool(a[3] > 0)
        if not closed:
            d.eq_active[0] = 0
        elif not self.closed and np.linalg.norm(d.mocap_pos[0] - d.xpos[2]) < .014:
            d.eq_active[0] = 1
        self.closed = closed
        # Interpolate the mocap target to give the equality solver a smooth trajectory.
        start = d.mocap_pos[0].copy()
        for fraction in range(1, 21):
            d.mocap_pos[0] = start + (next_tool - start) * (fraction / 20)
            mujoco.mj_step(self.model, d)
        mujoco.mj_forward(self.model, d)
        red, blue = d.xpos[2:4]
        contacts = {tuple(sorted(c.geom)) for c in d.contact}
        stacked = (not closed and np.linalg.norm(next_tool - red) > .045
                   and (2, 3) in contacts and (0, 3) in contacts
                   and np.linalg.norm(red[:2] - blue[:2]) < .012
                   and abs(red[2] - blue[2] - SIZE) < .006
                   and np.max(np.abs(d.qvel.reshape(2, 6)[:, :3])) < .025)
        self.stable = self.stable + 1 if stacked else 0
        self.steps += 1
        success = self.stable >= 8
        return self.observe(), success, success or self.steps >= MAX_STEPS


def scripted_episode(env, seed, variant="demo"):
    """Fixed offline collector. Misses and bad releases are deliberate interventions."""
    rng = np.random.default_rng(seed)
    obs = env.reset(seed)
    states, actions, rewards, successes = [obs], [], [], []
    red, blue = obs[3:6] * .1, obs[6:9] * .1
    grasp = red.copy()
    place = blue + [0, 0, env.block_height + .002]
    if variant == "miss":
        grasp[0] += .027
    elif variant == "offset":
        offset_rng = np.random.default_rng(seed + 30000)
        place[:2] += offset_rng.choice([-1, 1]) * offset_rng.uniform(.020, .043, 2)
    elif variant == "drop":
        place += [0, .045, .055]
    elif variant != "demo":
        raise ValueError(variant)
    waypoints = [(grasp + [0, 0, .065], -1), (grasp, -1), (grasp, 1),
                 (grasp + [0, 0, .095], 1), (place + [0, 0, .06], 1),
                 (place, 1), (place, -1), (place + [0, 0, .08], -1)]
    phase = dwell = 0
    for _ in range(env.max_steps):
        goal, grip = waypoints[phase]
        delta = (goal - obs[:3] * .1) / STEP
        xyz = np.clip(delta, -1, 1) * rng.uniform(.70, .98)
        action = np.r_[xyz, grip].astype(np.float32)
        next_obs, success, done = env.step(action)
        reward = float(potential(next_obs) - potential(obs) + 2 * success)
        states.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        successes.append(success)
        obs = next_obs
        if np.linalg.norm(delta) < .22 and phase < len(waypoints) - 1:
            dwell += 1
            if dwell >= (env.grip_delay if phase in (2, 6) else 1):
                phase += 1
                dwell = 0
        if done:
            break
    return (np.stack(states), np.stack(actions), np.array(rewards, np.float32),
            np.array(successes, np.float32))


def potential(obs):
    """Bounded shaping for approaching, lifting and aligning. Success is measured separately."""
    tool, red, blue = obs[:3], obs[3:6], obs[6:9]
    lift = np.clip((red[2] - .2) / .6, 0, 1)
    goal_dist = np.linalg.norm(red - (blue + [0, 0, .4]))
    return (.2 * np.exp(-np.linalg.norm(tool - red)) + .25 * obs[9]
            + .2 * lift + .5 * np.exp(-2 * goal_dist))
