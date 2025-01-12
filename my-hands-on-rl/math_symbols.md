## 数学符号

Note：notation上，大写的字母为stochastic random variable，代表某个分布。小写的字母为具体的值。

### 马尔可夫过程

1. 状态
    $$
    S
    $$
2. 状态转移矩阵
    $$
    P_{ss'} = P(s_{t+1} = s' | S_t = s)
    $$

### 马尔可夫奖励过程

1. 奖励函数
    $$
    r(s) = \mathbb{E}[r| S_t = s]
    $$
2. 回报
    $$
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-1} R_T = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
    $$
3. 价值函数
    $$
    V(s) = \mathbb{E}[G_t | S_t = s]
    \\ = \mathbb{E}[R_{t} + \gamma V(S_{t+1})|S_t=s]
    $$

### 马尔可夫决策过程

1. 动作
    $$
    A
    $$
2. 策略
    $$
    \pi(a|s) = P(A_t = a | S_t = s)
    $$
3. 状态价值函数
    $$
    V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]
    $$
4. 动作价值函数
    $$
    Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]
    $$
5. 贝尔曼期望方程
    $$
    V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s, a)
    $$
    $$
    Q^\pi(s, a) = \mathbb{E}[R_{t} + \gamma Q^\pi(S_{t+1}, A_{t+1})|S_t=s, A_t=a] = r(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V^\pi(s')
    $$
6. 状态访问分布
    $$
    \nu^\pi(s) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)
    $$
7. 占用度量
    $$
    \rho ^\pi(s,a) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t P_t^\pi(s)\pi(a|s)
    $$
8. 最优状态价值函数
    $$
    V^*(s) = \max_{\pi} V^\pi(s)
    $$
    $$
    V^*(s) = \max_{a\in A} Q^*(s,a)
    $$
9. 最优动作价值函数
    $$
    Q^*(s, a) = \max_{\pi} Q^\pi(s,a)=r(s,a)+\gamma\sum_{s' \in S}P(s'|s,a)V^*(s')
    $$
10. 贝尔曼最优方程
    $$
    V^*(s) = \max_{a \in A} \{r(s,a)+\gamma \sum_{s'\in S}p(s'|s,a)V^*(s')\}
    $$
    $$
    Q^*(s, a) = r(s, a) + \gamma \sum_{s' \in S} p(s'|s,a) \max_{a'\in A}Q^*(s',a')
    $$