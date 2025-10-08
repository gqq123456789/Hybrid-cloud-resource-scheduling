import random
from utils import check_constraints


def calculate_simplified_response_time(arrival_rate, service_rate, num_servers):
    """
    简化的响应时间计算（不使用排队论模型，直接基于到达率和服务速率计算）。
    适用于不使用排队论模型的算法。
    """
    # 简单假设：系统的响应时间与任务的到达率成正比
    # service_rate: 服务速率，arrival_rate: 任务到达速率，num_servers: 服务器数量

    # 如果到达率大于服务速率，返回一个较大的响应时间（系统可能过载）
    if arrival_rate > service_rate:
        return float('inf')  # 系统过载

    # 基于服务速率和到达率的简单比值来计算响应时间
    return arrival_rate / (num_servers * service_rate)


def calculate_simplified_throughput(arrival_rate, num_servers):
    """
    简化的吞吐量计算（不使用排队论模型）。
    适用于不使用排队论模型的算法。
    """
    # 吞吐量简化模型：吞吐量 = 服务器数 * 每个服务器的服务速率
    return min(arrival_rate, num_servers * 100)  # 假设每个服务器的最大吞吐量为100


def calculate_simplified_bandwidth_and_latency(arrival_rate):
    """
    简化的带宽和延迟计算（假设带宽和延迟与系统负载直接相关）。
    """
    # 假设带宽需求和延迟是线性相关的
    bandwidth = arrival_rate * 0.1  # 简化带宽需求（单位Mbps）
    latency = 10 + arrival_rate * 0.01  # 简化延迟（单位ms）

    # 假设延迟上限为100ms
    latency = min(latency, 100)

    return bandwidth, latency


class GeneticAlgorithm:
    def __init__(self, parameters, costs, storage_demand):
        """
        初始化遗传算法的参数和资源成本。
        """
        self.parameters = parameters
        self.costs = costs
        self.storage_demand = storage_demand

    def calculate_actual_qos_metrics(self, assignments, lambda_t, s, p, bandwidth_demand):
        """
        根据优化后的任务分配和资源使用情况计算每个 QoS 指标：
        - 响应时间 (response_time)
        - 吞吐量 (throughput)
        - 带宽利用率 (bandwidth_utilization)
        - 延迟 (latency)
        - 抖动 (jitter)
        """
        num_tasks = len(assignments)

        # 计算任务的到达率和服务速率
        arrival_rate = sum(lambda_t)
        service_rate = sum(p)

        # 计算总带宽需求
        total_bandwidth = sum(bandwidth_demand)

        # 响应时间
        response_time = arrival_rate / service_rate if service_rate > 0 else float('inf')

        # 吞吐量
        throughput = min(arrival_rate, service_rate)

        # 计算带宽利用率
        bandwidth_utilization = total_bandwidth / self.parameters.get("total_bandwidth", total_bandwidth)

        # 延迟
        latency = max(s) / service_rate if service_rate > 0 else float('inf')

        # 抖动
        jitter = latency * 0.1

        # 返回每个实际计算的 QoS 指标
        return {
            "response_time": response_time,
            "throughput": throughput,
            "bandwidth_utilization": bandwidth_utilization,
            "latency": latency,
            "jitter": jitter
        }

    def optimize(self, lambda_t, s, p, A_priv, A_pub, bandwidth_demand, population_size=100, generations=1000,
                 mutation_rate=0.1):
        """
        使用遗传算法优化任务分配，并动态计算每个 QoS 指标。
        """
        # 动态生成初始种群
        population = [
            [random.randint(0, self.parameters["M_priv"] + self.parameters["M_pub"] - 1) for _ in range(len(lambda_t))]
            for _ in range(population_size)
        ]

        best_solution = None
        best_cost = float('inf')
        best_qos_metrics = None  # 初始化最佳解的 QoS 指标

        for generation in range(generations):
            fitness = []
            for individual in population:
                # 修正 individual，确保任务分配在合法范围内
                for i, task in enumerate(individual):
                    if task < self.parameters["M_priv"]:
                        if A_priv[task] >= self.parameters["cpu_capacity_priv"] * 8:
                            pub_task = random.randint(0, self.parameters["M_pub"] - 1)
                            individual[i] = self.parameters["M_priv"] + pub_task
                    else:
                        pub_task = task - self.parameters["M_priv"]
                        if A_pub[pub_task] >= self.parameters["cpu_capacity_pub"] * 8:
                            priv_task = random.randint(0, self.parameters["M_priv"] - 1)
                            individual[i] = priv_task

                # 计算成本和资源使用情况
                cost, resource_usage = self.calculate_cost(
                    individual, lambda_t, s, p, A_priv, A_pub, bandwidth_demand
                )

                # 检查约束
                if not check_constraints(
                        self, resource_usage["cpu_usage_priv"], resource_usage["cpu_usage_pub"],
                        resource_usage["mem_usage_priv"], resource_usage["mem_usage_pub"],
                        self.parameters["user_experience_min"]
                ):
                    fitness.append((individual, float('inf'), None))  # 不满足约束，跳过
                    continue

                # 根据实际运行状态计算 QoS 指标
                qos_metrics = self.calculate_actual_qos_metrics(
                    individual, lambda_t, s, p,bandwidth_demand
                )

                # 计算适应度：总成本 + QoS 指标（加权组合，权重可调整）
                fitness_value = cost["total"] + qos_metrics["response_time"] * 0.5 - qos_metrics["throughput"] * 0.3

                # 将个体、适应度值和 QoS 指标存入 fitness 列表
                fitness.append((individual, fitness_value, qos_metrics))

                # 更新最佳解
                if fitness_value < best_cost:
                    best_solution = individual
                    best_cost = fitness_value
                    best_qos_metrics = qos_metrics

            # 选择最优个体作为父母
            fitness.sort(key=lambda x: x[1])  # 按适应度值升序排序
            selected_parents = [x[0] for x in fitness[:population_size // 2]]

            # 生成新种群，通过交叉和变异
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                crossover_point = random.randint(1, len(lambda_t) - 1)
                offspring = parent1[:crossover_point] + parent2[crossover_point:]
                # 变异
                if random.random() < mutation_rate:
                    mutation_point = random.randint(0, len(lambda_t) - 1)
                    offspring[mutation_point] = random.randint(
                        0, self.parameters["M_priv"] + self.parameters["M_pub"] - 1
                    )
                new_population.append(offspring)

            population = new_population

        return best_solution, best_cost, best_qos_metrics

    def dynamic_cloud_selection(self, lambda_t_priv, lambda_t_pub):
        """
        动态选择最优的云环境（私有或公有）。
        """
        response_time_priv = calculate_simplified_response_time(lambda_t_priv, self.parameters["service_rate_priv"],
                                                                  self.parameters["M_priv"])
        response_time_pub = calculate_simplified_response_time(lambda_t_pub, self.parameters["service_rate_pub"],
                                                                 self.parameters["M_pub"])
        return "private" if response_time_priv < response_time_pub else "public"

    def create_next_generation(self, fitness, population_size, mutation_rate):
        """
        生成下一代种群。
        """
        fitness_sorted = sorted(fitness, key=lambda x: x[1])
        parents = [individual for individual, _ in fitness_sorted[:population_size // 2]]

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            next_generation.extend([child1, child2])

        # 执行变异
        for individual in next_generation:
            if random.uniform(0, 1) < mutation_rate:
                mutation_point = random.randint(0, len(individual) - 1)
                individual[mutation_point] = random.randint(
                    0, self.parameters["M_priv"] + self.parameters["M_pub"] - 1
                )

        return next_generation[:population_size]
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



