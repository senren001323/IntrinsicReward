import numpy as np
from gym import spaces
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

class Explore(BaseSingleAgentAviary):
    """Single agent RL problem: Reach a random target
        In sparse reward
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
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
         -min_distance: 
             Minimum distance between generated point and initial drone's xyz
         -target:
             Randomly generated target point
        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        self.blocks = np.array([[1.5, 1.5, 0.5],
                                 [-1.5, 1.5, 0.5],
                                 [1.5, -1.5, 0.5],
                                 [-1.5, -1.5, 0.5]])
        self.blocks_ids = np.zeros(len(self.blocks), dtype=bool)
        self.EPISODE_LEN_SEC = 10

    def reset(self):
        super().reset()
        self.addObstacle()
        return self._computeObs()
    
    def addObstacle(self):
        """Add obstacles to the environment.
        Add the cube as target.
        """
        for block in self.blocks:
            p.loadURDF("../assets/cube_small.urdf",
                       block,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT,
                       useFixedBase=True)
    
    def _computeReward(self):
        """Computes the current reward value
            Sparse reward --> if reached:0, else:-1
        Return:
         -reward: float
        """
        reward = 0
        drone_pos = self._getDroneStateVector(0)[:3]
        for i, block in enumerate(self.blocks):
            if not self.blocks_ids[i] and np.linalg.norm(drone_pos - block) <= 0.5:
                reward += 2
                self.blocks_ids[i] = True
            else:
                reward += 0
        if abs(drone_pos[0]) >= 3 or abs(drone_pos[1]) >= 3 or drone_pos[2] >= 2:
            reward += -1
            
        return reward

    def _computeDone(self):
        """1.Reached Target: Done when drone reached target
           2.Timelimit: 1 second with 240 steps, 1200 steps in 5 seconds
           3.Out of bound: Done when drone cross the setting border
        Return:
         -done: bool
        """
        drone_pos = self._getDroneStateVector(0)[:3]
        if abs(drone_pos[0]) >= 3 or abs(drone_pos[1]) >= 3 or drone_pos[2] >= 2:
            done_outbound = True
        else: done_outbound = False

        done_reached = np.zeros(self.NUM_DRONES, dtype=bool)
        '''
        if any(np.linalg.norm(drone_pos - self.blocks, axis=1) <= 0.5):
            done_reached = True
        else:
            done_reached = False
        '''
        done_time = True if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        
        done = done_time or done_outbound
        return done
    
    def _computeInfo(self):
        """Unused.
        Return
         -dict[str, int] Dummy value.
        """
        return {"answer": 41}
        
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

