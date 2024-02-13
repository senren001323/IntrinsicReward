import math
import numpy as np
import pybullet as p
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary


class CasualEnv(BaseMultiagentAviary):
    """

    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        """
        Parameters:
         -target:
             Fixed target point
         -actions:
             Initial actions, would dynamically change in each step
         -EPISODE_LEN_SEC:
             Length of whole episode
         -INIT_XYZS:
             Init positions of drones
        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act)
        self.target = np.array([0.0, -2.0, 0.3])
        self.num_drones = num_drones
        self.actions = {i: np.zeros((1, 4)) for i in range(num_drones)}
        self.EPISODE_LEN_SEC = 2
        self.INIT_XYZS = np.array([[0.0, 0.0, 0.1125],
                                   [0.0, -0.5, 0.1125]])

    def reset(self):
        super().reset()
        self.addObstacle()
        self.target = np.array([0.0, -2.0, 0.3])
        return self._computeObs()

    def step(self, actions):
        """Extend from super class
            Especially get actions for agent B's obs
        """
        super().step(actions)
        self.actions = actions
        return self._computeObs(), self._computeReward(), self._computeDone(), self._computeInfo()
    
    def addObstacle(self):
        """Add obstacles to the environment.
        Add the small cube as target.
        """
        obstacle_id = p.loadURDF("assets/cube_small.urdf",
                         self.target,
                         p.getQuaternionFromEuler([0, 0, 0]),
                         physicsClientId=self.CLIENT,
                         useFixedBase=True)
        p.setCollisionFilterGroupMask(obstacle_id, -1, 
                                      collisionFilterGroup=0, 
                                      collisionFilterMask=0)


    def _computeObs(self):
        """Extend from super class
               Assign different observations for agent A and B 
        """
        obs = super()._computeObs()
        obs[1] = np.hstack((obs[1], self.actions[0].squeeze(0)))
        return obs

    def _observationSpace(self):
        """
        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
         -Agent 0: Box() os shape(12+3,), 3 is the target point
         -Agent 1: Box() os shape(12+4,), 4 is agent0's action
        """
        return spaces.Dict({0: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1]),
                                          high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1]),
                                          dtype=np.float32),
                            1: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,-1]),
                                          high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1,1]),
                                          dtype=np.float32)})

    def _computeReward(self):
        """Compute rewards
        reward 1: The distance between agent A and target, dense reward
        reward 2: The sparse reward for agent B
        
        Returns:
         -rewards: dict[int, float]
              The reward value for each drones
        """
        reward_0 = 0
        drone0_pos = self._getDroneStateVector(0)[:3]
        distance0 = np.linalg.norm(drone0_pos - self.target)
        reward_0 += 0.01 * 1/(distance0+1)
        if distance0 <= 0.8:
            reward_0 += 0.02 * 1/(distance0+1)
        if distance0 <= 0.4:
            reward_0 += 0.03 * 1/(distance0+1)
        if distance0 <= 0.1:
            reward_0 += 0.1 * 1/(distance0+1)

        reward_1 = 0
        drone1_pos = self._getDroneStateVector(0)[:3]
        distance1 = np.linalg.norm(drone1_pos - self.target)
        if distance1 < 0.1:
            reward_1 += 10

        rewards = {0: reward_0, 1: reward_1}
        return rewards

    def _computeDone(self):
        """
        1.Reached Target
        2.Timelimit: 1 second with 240 steps, 480 steps in 2 seconds
        3.Out of bound: Done when drone cross the setting border
        Return:
         -dict[int | "__all__", bool]
        """    
        done_outbound = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            if abs(drone_pos[0]) > 2 or abs(drone_pos[1]) > 2 or drone_pos[2] > 2:
                done_outbound[i] = True
            else:
                done_outbound[i] = False
                
        done_reached = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[:3]
            if np.linalg.norm(drone_pos - self.target) < 0.2:
                done_reached[i] = True
            else:
                done_reached[i] = False
        
        done_time = True if self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC else False
        
        done = {i: done_reached[i] or done_outbound[i] or done_time for i in range(self.NUM_DRONES)}
        done["__all__"] = np.any(done_reached) or done_time or np.all(done_outbound)
        return done
        
    def _computeInfo(self):
        """Unused.
        Returns:
        dict[int, dict[]]
            Dictionary of empty dictionaries.
        """
        return {i: {} for i in range(self.NUM_DRONES)}

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters:
         -state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        Returns:
         -ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 1 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
                
    def _clipAndNormalizeStateWarning(self, state, 
                                      clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        """
        Print a warning if values in a state vector is out of the clipping range.
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in _clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))





