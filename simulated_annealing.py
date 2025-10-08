import math
import random
import time
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
def calculate_qos_metrics(arrival_rate, service_rate, num_servers, max_bandwidth, max_latency, max_jitter):
    if isinstance(arrival_rate, list):
        arrival_rate = sum(arrival_rate)
    if isinstance(service_rate, list):
        service_rate = sum(service_rate)

    if arrival_rate <= 0 or service_rate <= 0:
        return {
            "response_time": float("inf"),
            "latency": float("inf"),
            "jitter": float("inf"),
            "throughput": 0.0,
            "bandwidth_utilization": 0.0,
        }

    response_time = arrival_rate / (num_servers * service_rate)
    latency = min(max_latency, 1 / (service_rate - arrival_rate) * 1000 if service_rate > arrival_rate else float("inf"))
    jitter = min(max_jitter, latency / 10)
    throughput = min(arrival_rate, max_bandwidth / 10)
    bandwidth_utilization = min(arrival_rate * 10, max_bandwidth) / max_bandwidth

    return {
        "response_time": response_time,
        "latency": latency,
        "jitter": jitter,
        "throughput": throughput,
        "bandwidth_utilization": bandwidth_utilization,
    }



class SimulatedAnnealing:
    def __init__(self, parameters, costs, storage_demand):
        """
        初始化模拟退火算法的参数、成本和存储需求。
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
        return {
            "response_time": response_time,
            "throughput": throughput,
            "bandwidth_utilization": bandwidth_utilization,
            "latency": latency,
            "jitter": jitter
        }

    def optimize(self, lambda_t, s, p, A_priv, A_pub, bandwidth_demand, max_iterations=1000, initial_temp=100):
        """
        使用模拟退火算法进行优化，找到最低成本的任务分配方案，并动态计算优化后的每个QoS指标（私有云和公有云合并）。
        """
        start_time = time.time()
        timeout = self.parameters["timeout"]

        # 初始化解和温度
        current_assignments = self.generate_dynamic_assignments(len(lambda_t))

        # 计算当前解的成本和资源使用情况
        current_cost, resource_usage = self.calculate_cost(
            current_assignments, lambda_t, s, p, A_priv, A_pub, bandwidth_demand
        )

        # 根据优化后的任务分配和资源使用情况计算当前解的 QoS 指标
        qos_metrics = self.calculate_actual_qos_metrics(
            current_assignments, lambda_t, s, p, bandwidth_demand
        )

        # 初始化最佳解和QoS指标
        best_assignments = current_assignments[:]
        best_cost = current_cost["total"]
        best_qos_metrics = qos_metrics
        temperature = float(initial_temp)

        for iteration in range(max_iterations):
            if time.time() - start_time > timeout:
                print("Timeout reached during iterations. Returning best found solution.")
                break

            # 生成新解（通过模拟退火的方式生成邻域解）
            new_assignments = self.generate_dynamic_assignments(len(lambda_t))
            # 检查约束条件
            if not check_constraints(
                    self, resource_usage["cpu_usage_priv"], resource_usage["cpu_usage_pub"],
                    resource_usage["mem_usage_priv"], resource_usage["mem_usage_pub"],
                    self.parameters["user_experience_min"]
            ):
                print("Constraint failed. Skipping this solution.")
                continue  # 如果不满足约束条件，则跳过这次迭代
            # 计算新解的成本和资源使用情况
            new_cost, resource_usage = self.calculate_cost(
                new_assignments, lambda_t, s, p, A_priv, A_pub, bandwidth_demand
            )

            # 根据优化后的任务分配和资源使用情况计算新解的 QoS 指标
            new_qos_metrics = self.calculate_actual_qos_metrics(
                new_assignments, lambda_t, s, p, bandwidth_demand
            )

            # 判断是否接受新解（通过Metropolis准则）
            cost_diff = new_cost["total"] - current_cost["total"]
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_assignments = new_assignments
                current_cost = new_cost
                qos_metrics = new_qos_metrics

                # 更新最佳解
                if current_cost["total"] < best_cost:
                    best_assignments = current_assignments
                    best_cost = current_cost["total"]
                    best_qos_metrics = qos_metrics

            # 降低温度
            temperature *= 0.99

        # 返回最佳解及对应的QoS指标
        return best_assignments, best_cost, best_qos_metrics

    def generate_dynamic_assignments(self, num_tasks):
        """
        根据当前负载动态生成任务分配方案。
        """
        assignments = []
        for _ in range(num_tasks):
            # 随机分配任务到私有云或公有云的主机
            if random.random() < 0.5:  # 50% 概率分配到私有云
                task = random.randint(0, self.parameters["M_priv"] - 1)
            else:  # 50% 概率分配到公有云
                task = random.randint(self.parameters["M_priv"],
                                      self.parameters["M_priv"] + self.parameters["M_pub"] - 1)
            assignments.append(task)
        return assignments

    def calculate_cost(self, assignments, lambda_t, s, p, A_priv, A_pub, bandwidth_demand):
        """
        计算资源使用情况和总成本，包括计算资源（CPU、内存）、存储成本和通信成本。

        assignments: 任务的分配情况（每个任务分配给哪个云）
        lambda_t: 每个任务的到达率
        s: 每个任务需要的内存（GB）
        p: 每个任务需要的CPU（核心数）
        A_priv: 私有云的任务分配情况
        A_pub: 公有云的任务分配情况
        bandwidth_demand: 带宽需求
        """
        total_cost = {
            "cpu_cost": 0,
            "mem_cost": 0,
            "net_cost": 0,
            "storage_cost": 0,
            "bandwidth_cost": 0,
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

        # 遍历任务，计算每个任务的资源使用情况和相应的成本
        for i, task in enumerate(assignments):
            cpu_cores = p[i] * lambda_t[i]  # 核数需求
            cpu_units = (cpu_cores + 7) // 8  # 转换为颗数（每 8 核为 1 颗，向上取整）

            if task < self.parameters["M_priv"]:
                # 任务分配给私有云
                if task < len(cpu_usage_priv):  # 确保索引在私有云范围内
                    cpu_usage_priv[task] += cpu_cores  # 记录私有云 CPU 使用（核数）
                    mem_usage_priv[task] += s[i] * lambda_t[i]  # 计算内存使用
                    total_cost["cpu_cost"] += cpu_units * self.costs["private"]["cpu"]  # 私有云按颗数计算CPU成本
                    total_cost["mem_cost"] += s[i] * self.costs["private"]["mem"] * lambda_t[i]  # 私有云内存成本
                    total_cost["storage_cost"] += self.storage_demand[i] * self.costs["private"]["storage"] * lambda_t[
                        i]  # 私有云存储成本
                    total_cost["bandwidth_cost"] += bandwidth_demand[i] * self.costs["private"]["bandwidth"]  # 私有云带宽成本
                    A_priv[task] += 1
                else:
                    print(f"Error: Task {i} assigned to invalid private cloud host {task}.")
            else:
                # 任务分配给公有云
                pub_task = task - self.parameters["M_priv"]
                if pub_task < len(cpu_usage_pub):  # 确保索引在公有云范围内
                    cpu_usage_pub[pub_task] += cpu_cores  # 记录公有云 CPU 使用（核数）
                    mem_usage_pub[pub_task] += s[i] * lambda_t[i]  # 计算内存使用
                    total_cost["cpu_cost"] += cpu_units * self.costs["public"]["cpu"]  # 公有云按颗数计算CPU成本
                    total_cost["mem_cost"] += s[i] * self.costs["public"]["mem"] * lambda_t[i]  # 公有云内存成本
                    total_cost["storage_cost"] += self.storage_demand[i] * self.costs["public"]["storage"] * lambda_t[
                        i]  # 公有云存储成本
                    total_cost["bandwidth_cost"] += bandwidth_demand[i] * self.costs["public"]["bandwidth"]  # 公有云带宽成本
                    A_pub[pub_task] += 1
                else:
                    print(f"Error: Task {i} assigned to invalid public cloud host {task}.")

        # 计算总成本
        total_cost["total"] = total_cost["cpu_cost"] + total_cost["mem_cost"] + total_cost["storage_cost"] + total_cost[
            "bandwidth_cost"]

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






