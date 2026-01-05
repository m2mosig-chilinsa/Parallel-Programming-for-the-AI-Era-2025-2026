import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
ray.init(ignore_reinit_error=True)
config = (
  PPOConfig()
  .environment("CartPole-v1")
  .env_runners(num_env_runners=1)
  .framework("torch")
  .training(lr=3e-4, gamma = 0.99)
)

algo = config.build_algo()

start = time.perf_counter()
for i in range(10):
  result = algo.train()
  reward = result["env_runners"]["episode_return_mean"]
  print(f'Iteration {i} reward = {reward}')
end = time.perf_counter()
print(f'=======Training time: {end - start:.2f}=========')
ray.shutdown()
