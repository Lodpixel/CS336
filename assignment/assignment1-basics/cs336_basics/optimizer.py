import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: torch.nn.Parameter,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
    
    def step(self):
        # param_group 是 optimizer 基类的成员，是一个 list[dict]。
        # 包含了所有基类传的所有 params 以及 defaults 里的键值对。
        # params 里是我们需要训练的东西，defaults 里面的东西是超参数配置 j
        #  self.param_groups            # list
        #  └── group                   # dict
        #      ├── "params"            # list
        #      │   ├── p1              # torch.nn.Parameter（张量）
        #      │   ├── p2              # torch.nn.Parameter
        #      │   └── ...
        #      ├── "lr"                # float
        #      └── "betas"             # tuple
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad == None:
                    continue
                # state 指的是每个参数保存下来的惯性量，每一轮会更新
                state = self.state.get(p, {"m": 0, "v": 0, "t": 0})
                # 数值初始化
                g = p.grad
                beta_1 = group["betas"][0]
                beta_2 = group["betas"][1]
                a = group["lr"]
                # t = t + 1，每迭代一轮，t 随之增加
                state["t"] += 1
                t = state["t"]
                # 执行 optimize 的流程
                a_t = a * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= a * p.data * group["weight_decay"]
                state["m"] = state["m"] * beta_1 + (1 - beta_1) * g
                state["v"] = state["v"] * beta_2 + (1 - beta_2) * (g ** 2)
                p.data -= a_t * state["m"] / torch.sqrt(state["v"] + group["eps"])
                # 第一轮运行的时候，state 是零初始化的，所以说此时更新
                # state 不会被反馈到 self.state[p] 里，需要做更新。
                if t == 1:
                    self.state[p] = state
                
# 用于实现根据轮次改变的学习率，it 指当前的轮次数
def lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters
):
    if (it < warmup_iters):
        return max_learning_rate * it / warmup_iters
    if (it > cosine_cycle_iters):
        return min_learning_rate
    ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + (max_learning_rate - min_learning_rate) / 2 * (1 + math.cos(math.pi * ratio))

# 进行 l2 范数计算，如果小于 max 则不动，大于则整体放缩
def gradient_clipping(
    parameter,
    l2_max
):
    # eps 为默认值
    eps = 1e-6
    sum = 0
    for p in parameter:
        if p.grad is not None:
            p_sqr = p.grad ** 2
            sum += p_sqr.sum()
    sum = math.sqrt(sum)
    if (sum > l2_max):
        for p in parameter:
            if p.grad is not None:
                p.grad *= (l2_max / (sum + eps))
    return