# EMPA-ASA.py
import math
import random
import time
import numpy as np
from utils import check_constraints


def adaptive_temperature_update(temperature, accepted_ratio):
    """
    基于自适应因子调整温度，接受率决定自适应因子大小。

    :param temperature: float, 当前温度
    :param accepted_ratio: float, 当前接受率
    :return: float, 更新后的温度
    """
    if accepted_ratio < 0.2:
        adaptive_factor = 0.98  # 接受率较低时，缓慢降温
    elif accepted_ratio < 0.5:
        adaptive_factor = 0.95  # 中等接受率，适度降温
    else:
        adaptive_factor = 0.90  # 接受率高，加速降温

    temperature *= adaptive_factor
    return temperature


def apply_action_to_assignments(assignments, action, parameters):
    """
    根据动作更新任务分配方案，确保任务分配在私有云和公有云范围内。
    """
    new_assignments = assignments[:]
    for i in range(len(assignments)):
        target_host = action
        if target_host < parameters["M_priv"]:  # 私有云
            new_assignments[i] = random.randint(0, parameters["M_priv"] - 1)
        else:  # 公有云
            new_assignments[i] = random.randint(parameters["M_priv"], parameters["M_priv"] + parameters["M_pub"] - 1)
    return new_assignments

def levy_distribution(beta, size=1):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    return u / abs(v) ** (1 / beta)


def total_objective(cost, user_experience, lambda_weight=0.5):
    """
    计算综合目标：权重较高的目标（成本或用户体验）会在优化中占主导地位。
    :param cost: 当前解的成本
    :param user_experience: 当前解的用户体验
    :param lambda_weight: 权重，控制成本和用户体验的相对重要性
    :return: 综合目标值
    """
    return lambda_weight * cost - (1 - lambda_weight) * user_experience


def empa_generate_new_solution(assignments, alpha, beta, parameters, A_priv, A_pub):
    """
    使用增强海洋捕食者算法 (EMPA) 的邻域搜索来生成新的任务分配解。
    """
    new_assignments = assignments[:]
    for i in range(len(assignments)):
        levy_flight = levy_distribution(beta)
        new_task = int(assignments[i] + alpha * levy_flight) % (parameters["M_priv"] + parameters["M_pub"])

        # 超载检查并调整
        if new_task < parameters["M_priv"]:  # 私有云任务
            if A_priv[new_task] >= parameters["cpu_capacity_priv"] * 8:  # 超载检查
                new_task = parameters["M_priv"] + random.randint(0, parameters["M_pub"] - 1)
        else:  # 公有云任务
            pub_task = new_task - parameters["M_priv"]
            if A_pub[pub_task] >= parameters["cpu_capacity_pub"] * 8:  # 超载检查
                new_task = random.randint(0, parameters["M_priv"] - 1)

        new_assignments[i] = new_task
    return new_assignments



class MDP:
    def __init__(self, states, actions, discount_factor=0.9, learning_rate=0.1):
        self.states = states
        self.actions = actions
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.q_table = np.zeros((len(states), len(actions)))

    def update_q_value(self, state, action, reward, next_state):
        # 确保 state 和 next_state 都在 q_table 的范围内
        state = state % len(self.q_table)
        next_state = next_state % len(self.q_table)

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def choose_action(self, state):
        # 确保 state 在 q_table 的范围内
        state = state % len(self.q_table)  # 限制 state 在 q_table 的行数范围内
        if random.uniform(0, 1) < 0.1:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])


def calculate_state(assignments):
    """
    根据当前任务分配（assignments）来计算状态。
    状态表示任务在私有云和公有云的分配情况，以及每个任务的资源使用情况。

    :param assignments: 当前任务分配的列表，表示每个任务被分配到的云控制器
    :return: 计算得到的状态值（通常是一个整数，代表当前任务分配的哈希值）
    """
    # 我们可以将任务的分配情况作为状态的一部分
    # 例如，任务分配是一个包含私有云和公有云控制器索引的列表
    state = tuple(assignments)  # 使用任务分配情况作为状态元组

    # 你也可以将其他因素（如资源使用情况）作为状态的一部分，基于当前的任务分配计算
    # 例如：计算私有云和公有云的资源使用情况（CPU、内存等）

    # 在此示例中，我们使用哈希值将状态映射到状态空间
    state_hash = hash(state) % (2 ** 16)  # 假设状态空间为 65536（可以根据需要调整）

    return state_hash

class ASA_EMPA:
    def __init__(self, parameters, costs, storage_demand):
        self.parameters = parameters
        self.costs = costs
        self.storage_demand = storage_demand
        self.mdp = MDP(
            range(parameters["M_priv"] + parameters["M_pub"]),
            range(parameters["M_priv"] + parameters["M_pub"]),
            parameters["discount_factor"],
            parameters["learning_rate"]
        )
    def create_mdp(self):
        # 创建并返回 MDP 对象
        states = range(self.parameters["M_priv"] + self.parameters["M_pub"])
        actions = range(self.parameters["M_priv"] + self.parameters["M_pub"])
        return MDP(states, actions, self.parameters["discount_factor"], self.parameters["learning_rate"])

    def calculate_mmc_performance(self, assignments, lambda_t,s, p, bandwidth_demand):
        """
        使用 M/M/c 模型根据优化后的任务分配和资源使用情况计算每个 QoS 指标：
        - 响应时间 (response_time)
        - 吞吐量 (throughput)
        - 带宽利用率 (bandwidth_utilization)
        - 延迟 (latency)
        - 抖动 (jitter)

        返回值是一个包含这些 QoS 指标的字典。
        """

        # 计算任务的到达率 (arrival rate) 和服务速率 (service rate)
        arrival_rate = sum(lambda_t)  # 总到达率
        service_rate = sum(p)  # 总服务速率
        c = len(set(assignments))  # 服务器数量，假设每个任务分配一个独立的服务器
        total_bandwidth = sum(bandwidth_demand)  # 总带宽需求
        # 计算系统负载
        rho = arrival_rate / (c * service_rate)
        # 计算响应时间 W (包含排队时间 Wq)
        if rho < 1:
            response_time = (1 / service_rate) * (1 / (1 - rho))  # 响应时间 W_q
            response_time += 1 / service_rate  # 总响应时间 W
        else:
            response_time = float('inf')  # 系统过载，响应时间趋于无穷大

        # 吞吐量 (Throughput) - 已有公式
        throughput = min(arrival_rate, service_rate * c)

        # 带宽利用率 (Bandwidth Utilization)
        bandwidth_utilization = total_bandwidth / (
                    arrival_rate * service_rate) if arrival_rate * service_rate > 0 else 0

        # 延迟 (Latency) - 基于任务分配和服务速率
        # 使用任务分配中的最大任务服务时间和均匀负载情况计算延迟
        latency = (sum(s) / len(s)) / service_rate if service_rate > 0 else float('inf')

        # 抖动 (Jitter) - 使用响应时间的标准差
        jitter = latency * 0.1

        # 返回计算的 QoS 指标字典
        return {
            "response_time": response_time,
            "throughput": throughput,
            "bandwidth_utilization": bandwidth_utilization,
            "latency": latency,
            "jitter": jitter
        }

    def optimize(self, lambda_t, s, p, A_priv, A_pub, bandwidth_demand, max_iterations=1000, initial_temp=100):
        """
        优化任务分配，通过结合 MDP、EMPA、自适应因子和 M/M/c 模型优化多个性能参数。
        """
        start_time = time.time()
        timeout = self.parameters["timeout"]

        # 根据当前负载动态生成任务分配
        num_tasks = len(lambda_t)
        assignments = [
            random.randint(0, self.parameters["M_priv"] + self.parameters["M_pub"] - 1)
            for _ in range(num_tasks)
        ]

        best_assignments, best_cost = assignments[:], float('inf')
        temperature = initial_temp
        accepted_solutions = 0

        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                break

            # 1. **MDP选择资源分配方案**
            state = calculate_state(assignments)
            action = self.mdp.choose_action(state)
            new_assignments = apply_action_to_assignments(assignments, action, self.parameters)

            # 修正任务分配，确保范围正确并避免超载
            for i, task in enumerate(new_assignments):
                if task < self.parameters["M_priv"]:
                    if A_priv[task] >= self.parameters["cpu_capacity_priv"] * 8:  # 超载修正
                        pub_task = random.randint(0, self.parameters["M_pub"] - 1)
                        new_assignments[i] = self.parameters["M_priv"] + pub_task
                else:
                    pub_task = task - self.parameters["M_priv"]
                    if A_pub[pub_task] >= self.parameters["cpu_capacity_pub"] * 8:  # 超载修正
                        priv_task = random.randint(0, self.parameters["M_priv"] - 1)
                        new_assignments[i] = priv_task

                        # 计算新解的成本
            new_cost, resource_usage = self.calculate_cost(new_assignments, lambda_t, s, p, A_priv, A_pub,
                                                                   bandwidth_demand)
            # 使用M/M/c模型计算优化后的QoS指标
            qos_metrics = self.calculate_mmc_performance(new_assignments, lambda_t,s, p, bandwidth_demand)

            # 3. **EMPA生成新解的优化**
            new_assignments = empa_generate_new_solution(
                new_assignments, alpha=0.7, beta=1.5, parameters=self.parameters, A_priv=A_priv, A_pub=A_pub
            )





            # 检查QoS和用户体验约束
            if check_constraints(
                    self, resource_usage["cpu_usage_priv"], resource_usage["cpu_usage_pub"],
                    resource_usage["mem_usage_priv"], resource_usage["mem_usage_pub"],
                    self.parameters["user_experience_min"]
            ):
                # 更新Q值
                reward = -new_cost["total"]
                next_state = calculate_state(new_assignments)
                self.mdp.update_q_value(state, action, reward, next_state)

                # 更新最优解
                if new_cost["total"] < best_cost:
                    best_assignments, best_cost = new_assignments[:], new_cost["total"]
                    accepted_solutions += 1
            # 4. **加入自适应因子优化新解的更新**
            accepted_ratio = accepted_solutions / (iteration + 1)
            temperature = adaptive_temperature_update(temperature, accepted_ratio)

        return best_assignments, best_cost, qos_metrics

    def calculate_cost(self, assignments, lambda_t, s, p, A_priv, A_pub, bandwidth_demand):
        """
        计算资源使用情况和总成本，包括计算资源（CPU、内存）、存储成本和通信成本。
        """
        # 确定任务数量为所有输入数组的最小长度
        num_tasks = min(len(assignments), len(lambda_t), len(s), len(p), len(bandwidth_demand))

        total_cost = {
            "cpu_cost": 0,
            "mem_cost": 0,
            "net_cost": 0,
            "storage_cost": 0,
            "bandwidth_cost": 0,
            "power_cost": 0,
            "total": 0
        }

        # 初始化资源使用情况
        cpu_usage_priv = [0] * self.parameters["M_priv"]
        cpu_usage_pub = [0] * self.parameters["M_pub"]
        mem_usage_priv = [0] * self.parameters["M_priv"]
        mem_usage_pub = [0] * self.parameters["M_pub"]
        net_usage_priv = [0] * self.parameters["M_priv"]
        net_usage_pub = [0] * self.parameters["M_pub"]
        storage_usage_priv = [0] * self.parameters["M_priv"]
        storage_usage_pub = [0] * self.parameters["M_pub"]

        # 修正 assignments，确保任务分配在主机容量范围内
        for i in range(num_tasks):
            task = assignments[i]
            if task < self.parameters["M_priv"]:
                # 确保分配到私有云的任务在合法范围内
                if task < 0 or task >= self.parameters["M_priv"]:
                    assignments[i] = random.randint(0, self.parameters["M_priv"] - 1)
            else:
                # 确保分配到公有云的任务在合法范围内
                pub_task = task - self.parameters["M_priv"]
                if pub_task < 0 or pub_task >= self.parameters["M_pub"]:
                    assignments[i] = random.randint(self.parameters["M_priv"],
                                                    self.parameters["M_priv"] + self.parameters["M_pub"] - 1)

        # 遍历任务，计算资源使用情况和相应的成本
        for i in range(num_tasks):
            cpu_cores = p[i] * lambda_t[i]
            cpu_units = (cpu_cores + 7) // 8  # 核数转换为颗数

            task = assignments[i]

            if task < self.parameters["M_priv"]:
                # 私有云
                if 0 <= task < len(cpu_usage_priv):
                    if cpu_usage_priv[task] + cpu_cores <= self.parameters["cpu_capacity_priv"] * 8:
                        cpu_usage_priv[task] += cpu_cores
                        mem_usage_priv[task] += s[i] * lambda_t[i]
                        net_usage_priv[task] += bandwidth_demand[i]
                        storage_usage_priv[task] += self.storage_demand[i] * lambda_t[i]

                        # 成本计算
                        total_cost["cpu_cost"] += cpu_units * self.costs["private"]["cpu"]
                        total_cost["mem_cost"] += s[i] * self.costs["private"]["mem"] * lambda_t[i]
                        total_cost["storage_cost"] += self.storage_demand[i] * self.costs["private"]["storage"] * \
                                                      lambda_t[i]
                        total_cost["bandwidth_cost"] += bandwidth_demand[i] * self.costs["private"]["bandwidth"]
                        total_cost["net_cost"] += bandwidth_demand[i] * self.costs["private"]["net"]
                        total_cost["power_cost"] += self.parameters["max_completion_time"] * self.costs["private"][
                            "power"]

                        A_priv[task] += 1
                    else:
                        pub_task = random.randint(0, self.parameters["M_pub"] - 1)
                        assignments[i] = self.parameters["M_priv"] + pub_task
                else:
                    raise IndexError(f"Task {i} assigned to invalid private cloud host {task}.")
            else:
                # 公有云
                pub_task = task - self.parameters["M_priv"]
                if 0 <= pub_task < len(cpu_usage_pub):
                    if cpu_usage_pub[pub_task] + cpu_cores <= self.parameters["cpu_capacity_pub"] * 8:
                        cpu_usage_pub[pub_task] += cpu_cores
                        mem_usage_pub[pub_task] += s[i] * lambda_t[i]
                        net_usage_pub[pub_task] += bandwidth_demand[i]
                        storage_usage_pub[pub_task] += self.storage_demand[i] * lambda_t[i]

                        # 成本计算
                        total_cost["cpu_cost"] += cpu_units * self.costs["public"]["cpu"]
                        total_cost["mem_cost"] += s[i] * self.costs["public"]["mem"] * lambda_t[i]
                        total_cost["storage_cost"] += self.storage_demand[i] * self.costs["public"]["storage"] * \
                                                      lambda_t[i]
                        total_cost["bandwidth_cost"] += bandwidth_demand[i] * self.costs["public"]["bandwidth"]
                        total_cost["net_cost"] += bandwidth_demand[i] * self.costs["public"]["net"]
                        total_cost["power_cost"] += self.parameters["max_completion_time"] * self.costs["public"][
                            "power"]

                        A_pub[pub_task] += 1
                    else:
                        priv_task = random.randint(0, self.parameters["M_priv"] - 1)
                        assignments[i] = priv_task
                else:
                    raise IndexError(f"Task {i} assigned to invalid public cloud host {pub_task}.")

        # 计算总成本
        total_cost["total"] = (
                total_cost["cpu_cost"] +
                total_cost["mem_cost"] +
                total_cost["storage_cost"] +
                total_cost["bandwidth_cost"] +
                total_cost["power_cost"]
        )

        # 返回计算后的总成本和资源使用情况
        resource_usage = {
            "cpu_usage_priv": cpu_usage_priv,
            "cpu_usage_pub": cpu_usage_pub,
            "mem_usage_priv": mem_usage_priv,
            "mem_usage_pub": mem_usage_pub,
            "net_usage_priv": net_usage_priv,
            "net_usage_pub": net_usage_pub,
            "storage_usage_priv": storage_usage_priv,
            "storage_usage_pub": storage_usage_pub
        }

        return total_cost, resource_usage








