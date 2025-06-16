* 定义：所有RL算法的通用接口

> _class_stable_baselines3.common.base_class.BaseAlgorithm(_policy_, _env_, _learning_rate_, _policy_kwargs=None_, _stats_window_size=100_, _tensorboard_log=None_, _verbose=0_, _device='auto'_, _support_multi_env=False_, _monitor_wrapper=True_, _seed=None_, _use_sde=False_, _sde_sample_freq=-1_, _supported_action_spaces=None_) 

#### 基础参数Parameters解析
* **policy** `[str, type[BasePolicy]] `
    * 基础策略模型，比如 MlpPolicy CnnPolicy
*  **env**  `[GymEnv, str, None]`
    * 学习所用的环境，如果已注册在Gym当中，可以是str。也可以是加载训练好的模型而设为None
*  **learning_rate** 