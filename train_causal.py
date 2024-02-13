from train_function.train import PPOTrainCausal
from custom_envs.CausalImitationA import CasualEnv


env = CasualEnv()

policy = PPOTrainCausal(env)
agents, return_list = policy.learn(num_episodes=1000)

env_test = CasualEnv(gui=True,
                    record=True,
                    )
agents.agents[0].set_mode('test')
agents.agents[1].set_mode('test')

obs = env_test.reset()
for i in range(200*env_test.SIM_FREQ):
    action = agents.take_actions(obs)
    obs, reward, done, info = env_test.step(action)
    done = done["__all__"]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    if done:
        obs = env_test.reset()
env_test.close()

agents.save_model(1)