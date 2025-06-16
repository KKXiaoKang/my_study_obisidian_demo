* 定义：所有RL算法的通用接口

```python
class BaseAlgorithm(ABC):

"""

The base of RL algorithms

  

:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)

:param env: The environment to learn from

(if registered in Gym, can be str. Can be None for loading trained models)

:param learning_rate: learning rate for the optimizer,

it can be a function of the current progress remaining (from 1 to 0)

:param policy_kwargs: Additional arguments to be passed to the policy on creation

:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average

the reported success rate, mean episode length, and mean reward over

:param tensorboard_log: the log location for tensorboard (if None, no logging)

:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for

debug messages

:param device: Device on which the code should run.

By default, it will try to use a Cuda compatible device and fallback to cpu

if it is not possible.

:param support_multi_env: Whether the algorithm supports training

with multiple environments (as in A2C)

:param monitor_wrapper: When creating an environment, whether to wrap it

or not in a Monitor wrapper.

:param seed: Seed for the pseudo random generators

:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)

instead of action noise exploration (default: False)

:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE

Default: -1 (only sample at the beginning of the rollout)

:param supported_action_spaces: The action spaces supported by the algorithm.

"""
```

#### 基础参数Parameters解析
* **policy** `[str, type[BasePolicy]] `
    * 基础策略模型，比如 MlpPolicy CnnPolicy
*  **env**  `[GymEnv, str, None]`
    * 学习所用的环境，如果已注册在Gym当中，可以是str。也可以是加载训练好的模型而设为None
*  **learning_rate** 