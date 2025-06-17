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
* ***stats_window_size**  `[int]`
    * 状态窗口大小 (int) – 推理日志的窗口大小，指定要平均报告的成功率、平均回合长度和平均奖励的回合数量
* ***policy_kwargs** `[dict[str, Any] | None]`
    * 创建策略时候额外传递的参数

##### 本类特有
* ***buffer_size** `[int]`
    * 回放缓冲区的大小
*  ***learning_starts** `[int]`
    * 模型开始学习之前，模型收集转换的步数
* ***batch_size** `[int]`
    *  每次梯度更新的小批量大小
* ***tau** `[float]`
    * 软更新系数，介于$0$ 和 $1$ 之间
* ***gamma** `[float]`
    * 折扣因子，用于Q值计算`MSE`计算损失 
* ***train_freq** `[int | tuple [int, str]]`
	* 每执行`train_freq`步更新模型，或者传递一个频率和单位的元组， 如`(5, "step")` 或 `(2, "episode")`
*  ***gradient_steps** `[int]`
    * 每次`rollout`后执行多少梯度步，设置为-1表示执行的梯度步数与rollout过程中环境执行的步数相同



> 经验说明：一般每次`rollout`当中指定`N_step`为16，然后执行一次`gradient_steps=1`

*  ***action_noise** `[ActionNoise | None]`
    * 动作噪声类型，这有助于解决困难的探索问题 。 参考`common.noise` 不同的动作噪声
*   ***replay_buffer_class** `[dict[str, Any | None]]`
    * 使用的回放缓冲区类。如果为None 则会自动选择
* ***optimize_memory_usage** `[bool]`
    * 启用更复杂的内存高效回放缓冲区变体
* ***use_sde_at_warmup** `[bool]`
    * 是否在预热阶段使用 gSDE 而不是均匀采样（学习开始前） 
* ***sde_support** `[bool]` 
    * 模型是否支持 gSDE 


#### 子函数说明
* ***collect_rollouts** `(env, callback, train_freq, action_noise, learning_starts, replay_buffer, log_interval)`
     * 收集经验并且存储到`replay_buffer`当中
     * `env`
         * 训练环境 
    * `callback`
         * 在每一步（以及策略回放的开始和结束）时将被调用的回调函数
    * `train_freq`
         * 训练频率，通过执行当前策略的回放来收集多少经验
    *  `action_noise`
         *  动作噪声（ActionNoise 或 None）—— 用于探索的动作噪声，确定性策略（例如 TD3）所必需。这也可以与 SAC 中的随机策略一起使用
    * `replay_buffer`
    * `log_interval`

*  ***dump_logs**
    * 写入日志到Tensorboard日志

* ***learn** `(total_timesteps, callback=None, log_interval=4, tb_log_name="run", reset_num_timesteps=True, progress_bar=False)`
     * `total_timesteps`
         * int 
         * 用于训练的总样本数（环境步数），模型能力的下限
    *  `callback`
         *  每一步都会调用的回调函数，传入算法的状态
    * `log_interval`
         * 对于在线算法(PPO, A2C)  这是记录前的训练迭代次数
         * 对于离线算法（TD3 , SAC） 这是记录前的回合数
    * `tb_log_name`
        *  TensorBoard 记录的运行名称
    * `reset_num_timesteps` 
        * 是否重置当前时间步数（用于日志记录）
    * `progress_bar` (bool) 
        * 是否使用rich和tqdm显示进度条
    * `self` 
        * 直接加载已训练的模型 

* ***load_replay_buffer** 
    * 从pickle文件加载回放缓冲区 
    *  `path` (str)
        *  回放缓冲区的路径 
    * `truncate_last_traj` (bool)
        * 当使用`HerReplayBuffer`进行在线采样的时候，如果设置为`True`， 我们假设回放缓冲区中的最后一个轨迹已经完成（并截断它）。如果设置为`False`，我们假设我们继续相同的轨迹（相同的回合）

* ***save_replay_buffer**  
    * 将重放缓冲区保存为pickle文件
    * `path`
        * 重放缓冲区保存的文件路径 


* ***train** `[gradient_steps, batch_size]`
    * 对重放缓冲区进行采样并且执行更新（梯度下降和更新目标网络）
    * `gradient_steps` (int)
        * 梯度步数
    * `batch_size` (int)
        * 批次大小    
	