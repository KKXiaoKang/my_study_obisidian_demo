*  基础离线策略的实现类，比如（`SAC`）和 （`TD3`）
*  继承于`BaseAlgorithm`
```python
class OffPolicyAlgorithm(BaseAlgorithm):

"""

The base for Off-Policy algorithms (ex: SAC/TD3)

  

:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)

:param env: The environment to learn from

(if registered in Gym, can be str. Can be None for loading trained models)

:param learning_rate: learning rate for the optimizer,

it can be a function of the current progress remaining (from 1 to 0)

:param buffer_size: size of the replay buffer

:param learning_starts: how many steps of the model to collect transitions for before learning starts

:param batch_size: Minibatch size for each gradient update

:param tau: the soft update coefficient ("Polyak update", between 0 and 1)

:param gamma: the discount factor

:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit

like ``(5, "step")`` or ``(2, "episode")``.

:param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)

Set to ``-1`` means to do as many gradient steps as steps done in the environment

during the rollout.

:param action_noise: the action noise type (None by default), this can help

for hard exploration problem. Cf common.noise for the different action noise type.

:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).

If ``None``, it will be automatically selected.

:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.

:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer

at a cost of more complexity.

See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195

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

:param use_sde: Whether to use State Dependent Exploration (SDE)

instead of action noise exploration (default: False)

:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE

Default: -1 (only sample at the beginning of the rollout)

:param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling

during the warm up phase (before learning starts)

:param sde_support: Whether the model support gSDE or not

:param supported_action_spaces: The action spaces supported by the algorithm.

"""
```
#### 基础参数Parameters解析
##### 父类继承
* ***policy** `[str, type[BasePolicy]] `
    * 基础策略模型，比如 MlpPolicy CnnPolicy
* ***env**  `[GymEnv, str, None]`
    * 学习所用的环境，如果已注册在Gym当中，可以是str。也可以是加载训练好的模型而设为None
* ***learning_rate** `[float |Callable[float]]`
    *  优化器的学习率，可以是当前剩余进度（从 1 到 0）的函数

##### 本类特有
* ***buffer_size** `[int]`
    *   回放缓冲区的大小