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
* ***policy** `[str, type[BasePolicy]] `
    * 基础策略模型，比如 MlpPolicy CnnPolicy
* ***env**  `[GymEnv, str, None]`
    * 学习所用的环境，如果已注册在Gym当中，可以是str。也可以是加载训练好的模型而设为None
* ***learning_rate** `[float |Callable[float]]`
    *  优化器的学习率，可以是当前剩余进度（从 1 到 0）的函数
* ***policy_kwargs** `[dict [str,Any] | None]`
    *   策略参数 (dict[str, Any] | None) – 创建策略时传递的额外参数
* ***stats_window_size**  `[int]`
    * 状态窗口大小 (int) – 推理日志的窗口大小，指定要平均报告的成功率、平均回合长度和平均奖励的回合数量
* ***tensorboard_log**   `(str | None)`
    *  tensorboard 日志 (str | None) – tensorboard 的日志位置（如果为 None，则不进行日志记录）
* ***verbose**  `(int)`
    *  verbose (int) – 日志级别：0 表示不输出，1 表示输出信息（如设备或包装器使用情况），2 表示输出调试信息
* ***device** `[device | int]`
    *  device (device | str) – 代码应运行的设备。默认情况下，会尝试使用兼容 Cuda 的设备，如果不可行则回退到 CPU
* ***support_multi_env** `[bool]`
    *  – 算法是否支持使用多个环境进行训练（比如`A2C`）
* ***monitor_wrapper** `[bool]`
    * 在创建环境时，是否用`Monitor`包装器进行包装
* ***seed** `[int | None]`
    * 随机数生成的种子
* ***use_sde** `[bool]`
    *  是否使用广义状态依赖探索（gSDE）而不是动作噪声探索（默认：False）
* ***sde_sample_freq** `[int]`
    *  使用 gSDE 时，每 n 步采样一个新的噪声矩阵。默认：-1（仅在展开开始时采样）
*  ***supported_action_spaces** `[tuple[type[Space],...] | None ]`
    *  