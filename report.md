# Project Report: Distributed Reinforcement Learning on Grid'5000

## 1. Introduction
The objective of this project was to deploy a distributed Reinforcement Learning (RL) workload on the **Grid'5000** cluster. We utilized **Ray RLLib**, an industry-standard library for distributed RL, to train an agent to solve control problems provided by **Gymnasium**.

The primary goal was to demonstrate how adding computational resources (scaling from 1 node to 2 nodes) impacts training throughput and time-to-solution.

## 2. Infrastructure Setup
The experiments were conducted on the **Grenoble** site (specifically the `dahu` clusters). We used the `oar` scheduler to reserve resources and deployed a custom Python environment containing `Ray`, `PyTorch`, and `Gymnasium`.
To access the grid500 cluster we use ssh. The resourcees were reserved using ```oarsub -I -l nodes=2,walltime=01:00:00``` command
### 2.1 Cluster Architecture
We manually orchestrated a Ray cluster over two physical nodes using SSH tunneling.

*   **Node 1 (Head Node):** Responsible for the Global Driver, the Policy Optimizer (Learner), and managing the object store.
*   **Node 2 (Worker Node):** Dedicated to running parallel environment simulations (EnvRunners) to collect data.

### 2.2 Deployment Commands
To link the physical machines into a single logical cluster, the following commands were executed:

**On the Head Node:**
```bash
ray start --head --port=6379
```

**On the Worker Node:**
```bash
ray start --address='<HEAD_NODE_IP>:6379'
```

```bash
(venv) achilins@dahu-6:~$ ray status
======== Autoscaler status: 2026-01-05 16:14:02.304327 ========
Node status
---------------------------------------------------------------
Active:
 1 node_5c45432329db5af499d4447496d115aca915fd1f6bc5b53a08e0331a
 1 node_c90c5804a9e0130c46a4c8a9d6a7d9e4eddfb3f4e716d86110c1d19f
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Total Usage:
 0.0/64.0 CPU
 0B/260.30GiB memory
 0B/111.56GiB object_store_memory

From request_resources:
 (none)
Pending Demands:
 (no resource demands)
```

---

## 3. Methodology & Problems Solved

We selected the **Proximal Policy Optimization (PPO)** algorithm, which is robust and supports parallel data collection. We tested the infrastructure on **CartPole-v1**, a classic control problem where the agent balances a pole.

---

## 4. Performance Analysis

To evaluate performance, we compared the **Total Time** required to complete a fixed number of training iterations.

*   **Config 1 (Single Node):** 1 Physical Node, 1 CPU, `num_env_runners=1`.
*   **Config 2 (Single Node):** 1 Physical Node, 64 CPUs, `num_env_runners=64`.
*   **Config 2 (Distributed):** 2 Physical Nodes, 128 CPUs, `num_env_runners=128`.

### 4.1 Results Table

| Metric | 1 Node (Baseline) | 1 Node | 2 Nodes (Distributed) |
| :--- | :--- | :--- | :--- |
| **Number of Workers** | 1 | 64 | 128 | 2x |
| **Training Time (10 Iters)** | ~150s | ~120s | ~100s |
| **Mean Reward (Final)** | ~290.00 | ~250.00 | ~270.00 |

### 4.2 Observations
By increasing the number of workers, the time decreased slightly. However, when we further increased the number of workers and distributed the work between 2 nodes, the time didn't change as much as we expected. The speed is not linear due to Network Overhead. In distributed RL, the Head node must serialize neural network weights and send them to workers over the network, and workers must send collected trajectories back. For lightweight environments like CartPole, this communication cost is significant.

```bash
Iteration 1 reward = 43.09
Iteration 2 reward = 72.59
Iteration 3 reward = 102.51
Iteration 4 reward = 135.33
Iteration 5 reward = 166.64
Iteration 6 reward = 200.11
Iteration 7 reward = 229.01
Iteration 8 reward = 258.17
Iteration 9 reward = 289.82
=======Training time: 150.22=========
```

We were able utilise all CPUs
```bash
(venv) achilins@dahu-6:~$ watch -n 1 "ray status"

Every 1.0s: ray status                                                                                                              dahu-6.grenoble.grid5000.fr: Mon Jan  5 15:23:00 2026

======== Autoscaler status: 2026-01-05 15:22:58.764560 ========
Node status
---------------------------------------------------------------
Active:
 1 node_c08247fb9f60c29723f5a9f40fae7ef385bbfd3b3bb9c919e8fe1b2b
 1 node_d2297439e47d5cdecdb8b5aa2e26cf6403ca3333e38916036d291633
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Total Usage:
 128.0/128.0 CPU
 0B/251.03GiB memory
 9.77KiB/107.58GiB object_store_memory

From request_resources:
 (none)
Pending Demands:
 (no resource demands)
```

---

## 5. Implementation Code

Single-node implementation:
```python
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
```
Distributed implementation:
```python
import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
ray.init(address="auto")
config = (
  PPOConfig()
  .environment("CartPole-v1")
  .env_runners(num_env_runners=128)
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
```


## 6. Conclusion
This project successfully demonstrated the deployment of a distributed AI workload on Grid'5000. By orchestrating a Ray cluster across two nodes, we were able to parallelize the data collection phase of Reinforcement Learning.

Key takeaways:
1.  **Ray RLLib** abstracts the complexity of distributed computing; the same code runs on 1 node or 100 nodes.
2.  **Scaling behavior:** Distributed computing provides speedups, but it requires tuning the number of workers (`env_runners`) to match the available CPU cores. Also, the speedup is not always what we expect.
3.  **Environment Management:** Setting up dependencies (like SWIG/Box2D) on restricted HPC environments requires careful package management or alternative environment selection (Acrobot).
