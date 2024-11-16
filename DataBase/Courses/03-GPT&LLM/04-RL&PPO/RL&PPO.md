# Reinforcement Learning

## Policy Gradient

* Actor, Environment, Reward

* Policy of Actor 

  * Policy $\pi$ 是一个网络，具有参数 $\theta$
  * 输入环境的观测，输出动作

* Trajectory 序列 $\tau={s_1,a_1,s_2,a_2,...,s_T,a_T}, s表示state, a表示action$

* policy具有参数 $\theta$ 时得到序列 $\tau$ 的概率

  * $p_\theta(\tau)=p(s_1)p_\theta(a_1|s_1)p(s_2|s_1,a_1)p_\theta(a_2|s_2)p(s_3|s_2,a_2)...$
  * $p_\theta(\tau)=p(s_1)\prod_{t=1}^Tp_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$

* Reward 

  * 单次奖励 $R(\tau)=\sum_{t=1}^Tr_t$
  * 期望奖励 $\bar{R_\theta}=\sum_{\tau}R(\tau)p_{\theta}(\tau)=E_{\tau\sim p_\theta(\tau)}[R(\tau)]$ 

* Policy Gradient

  $$
  \begin{aligned}
  
  \theta &\leftarrow \theta+\eta\triangledown\bar{R_\theta}
  
  \\
  \triangledown\bar{R_\theta}
  &=\sum_\tau R(\tau)\triangledown p_\theta(\tau)
  
  \\
  &=\sum_\tau R(\tau)p_\theta(\tau)\frac{\triangledown p_\theta(\tau)}{p_\theta(\tau)}
  
  \\
  &=\sum_\tau R(\tau)p_\theta(\tau)\triangledown log p_\theta(\tau)
  
  \\
  &=E_{\tau\sim p_\theta(\tau)}[R(\tau)\triangledown log p_\theta(\tau)]
  
  \\
  & \approx \frac1N \sum_{n=1}^N R(\tau^n) \triangledown log p_\theta(\tau^n)
  
  \\
  &= \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n} R(\tau^n)\triangledown log p_\theta(a_t^n|s_t^n)
  
  \end{aligned}
  $$

* 增加 baseline
  $$
  \begin{aligned}
  \triangledown\bar{R_\theta} 
  &\approx \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n} (R(\tau^n)-b) \triangledown log p_\theta(a_t^n|s_t^n)
  
  \\
  b &\approx E[R(\tau)] 
  \end{aligned}
  $$

* 分配合适的分数
  $$
  \begin{aligned}
  \triangledown\bar{R_\theta} 
  &\approx \frac1N \sum_{n=1}^N \sum_{t=1}^{T_n} (R(\tau^n)-b) \triangledown log p_\theta(a_t^n|s_t^n)
  
  \\
  
  R(\tau^n) 
  &\rightarrow \sum_{t'=t}^{T_n} r_{t'}^n 
  \rightarrow \sum_{t'=t}^{T_n} \gamma^{t'-t} \times r_{t'}^{n}
  \quad
  (\gamma < 1)
  
  \\
  A^\theta(s_t,a_t)
  &=\sum_{t'=t}^{T_n} \gamma^{t'-t} \times r_{t'}^{n} - b
  \end{aligned}
  $$

  * 不再取最终 Reward，取当前时刻到最后的 Reward 之和
  * 增加递减因子。随时间跨度变大，Reward 贡献变小
  * 将 $R(\tau^n)-b$ 及其变种用 $A^\theta(s_t,a_t)$ 代替，含义为：在 $s_t$ 状态采取 $a_t$ 行动的利益

## On-policy --> Off-policy 

* on-policy：agent 行为策略与目标策略相同

* off-policy：agent 行为策略与目标策略不同

* Importance Sampling
  $$
  \begin{aligned}
  \triangledown\bar{R_\theta}
  &=E_{\tau\sim p_\theta(\tau)}[R(\tau)\triangledown log p_\theta(\tau)]
  
  \\
  E_{\tau\sim p}[f(x)]
  &\approx \frac1N \sum_{i=1}^N f(x^i)
  
  \\
  &=\int f(x)p(x)dx = \int f(x)\frac{p(x)}{q(x)}q(x)dx
  
  \\
  &=E_{x \sim q}[f(x)\frac{p(x)}{q(x)}]
  
  \\
  VAR_{x\sim p}[f(x)] 
  &\quad 不一定等于 \quad
  VAR_{x\sim q}[f(x)\frac{p(x)}{q(x)}]
  
  \\
  期望相同，方差&不同，应避免 p(x) 与 q(x) 分布差异过大
  \end{aligned}
  $$

* off-policy 公式
  $$
  \begin{aligned}
  \triangledown\bar{R_\theta}
  &=E_{\tau\sim p_{\theta'}(\tau)}
  \bigg[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau)\triangledown log p_\theta(\tau)\bigg]
  \end{aligned}
  $$

* 允许从另外的分布采样

* 梯度更新公式
  $$
  \begin{aligned}
  梯度更新
  &=E_{(s_t,a_t)\sim \pi_\theta}[A^\theta(s_t,a_t)\triangledown log p_\theta(a_t^n|s_t^n)]
  
  \\
  &= E_{(s_t,a_t)\sim \pi_{\theta'}}[\frac{p_\theta(s_t,a_t)}{p_{\theta'}(s_t,a_t)}A^{\theta'}(s_t,a_t)\triangledown log p_\theta(a_t^n|s_t^n)]
  
  \\
  &= E_{(s_t,a_t)\sim \pi_{\theta'}}[
  \frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}
  \frac{p_\theta(s_t)}{p_{\theta'}(s_t)}
  A^{\theta'}(s_t,a_t)\triangledown log p_\theta(a_t^n|s_t^n)]
  
  \\
  &\approx E_{(s_t,a_t)\sim \pi_{\theta'}}[\frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}
  A^{\theta'}(s_t,a_t)\triangledown log p_\theta(a_t^n|s_t^n)]
  
  \\
  J^{\theta'}(\theta)
  &= E_{(s_t,a_t)\sim \pi_{\theta'}}[
  \frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}
  A^{\theta'}(s_t,a_t)]
  \end{aligned}
  $$
  

## Add Constraint

* PPO/TRPO，Proximal Policy Optimization，近端策略优化

* PPO 公式
  $$
  \begin{aligned}
  &J_{PPO}^{\theta'}(\theta)=J^{\theta'}(\theta)-\beta KL(\theta,\theta')
  \\
  &KL(\theta,\theta')为 \theta 和 \theta' 输出的行为概率的分布的KL散度
  \end{aligned}
  $$
  
* TRPO 公式

  
  $$
  \begin{aligned}
  J_{TRPO}^{\theta'}(\theta)=E_{(s_t,a_t)\sim \pi_{\theta'}}
  \bigg[
  \frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)}
  A^{\theta'}(s_t,a_t)
  \bigg]
  \quad (~KL(\theta,\theta')<\delta~)
  \end{aligned}
  $$

* PPO 算法

  * 初始化 $\theta^0$
  * 迭代
    * 用 $\theta^k$ 与环境互动，得到 ${s_t,a_t}$ 并计算 $A^{\theta^k}(s_t,a_t)$
    * 用 $\theta$ 计算梯度并更新优化

$$
\begin{aligned}
    J^{\theta^k}(\theta)
    &\approx \sum_{(s_t,a_t)}
    \frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}
    A^{\theta^k}(s_t,a_t)

  \\
  J_{PPO}^{\theta^k} 
  &= J^{\theta^k}(\theta)-\beta KL(\theta, \theta^k)

  

  \end{aligned}
$$

  * 适应性 KL 惩罚项

    * 如果 $KL(\theta,\theta^k)>KL_{max}$ ，增大 $\beta$

    * 如果 $KL(\theta,\theta^k)<KL_{min}$ ，减小 $\beta$

* PPO2 算法
  $$
  \begin{aligned}
  
  J_{PPO2}^{\theta^k}(\theta)
  \approx \sum_{(s_t,a_t)} min \Bigg(&
  \frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}
  A^{\theta^k}(s_t,a_t),
  
  \\&
  clip\Big(
  \frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)},1-\epsilon,1+\epsilon
  \Big)
  A^{\theta^k}(s_t,a_t)
  \Bigg)
  
  \end{aligned}
  $$

* 限制分布变化