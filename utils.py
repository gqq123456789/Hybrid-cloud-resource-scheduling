# utils.py
import random


def calculate_qos_parameters( parameters):
    """
    计算 QoS 参数并返回相应的值。该函数直接控制各个 QoS 参数。

    assignments: 任务的分配情况（每个任务分配到的云资源）
    N: 任务总数
    parameters: 包含最大/最小QoS限制的字典（例如：最大响应时间、最大存储成本等）
    """
    # 模拟计算各个 QoS 参数

    # 计算响应时间（假设响应时间与任务数量成反比）
    response_time = random.uniform(0, parameters["max_response_time"])  # 最大响应时间限制

    # 计算存储成本（假设存储成本与任务分配的云资源有关）
    storage_cost = random.uniform(0, parameters["max_storage_cost"])  # 最大存储成本限制

    # 网络延迟（假设延迟与任务分配和带宽有关）
    latency = random.uniform(0, parameters["max_latency"])  # 最大延迟限制（毫秒）

    # 网络带宽（假设带宽与任务分配的云资源相关）
    bandwidth = random.uniform(0, parameters["max_bandwidth"])  # 最大带宽限制（GB）

    # 网络抖动（假设抖动与带宽需求、任务数量相关）
    jitter = random.uniform(0, parameters["max_jitter"])  # 最大抖动限制（毫秒）

    # 丢包率（假设丢包率与网络带宽和任务数量相关）
    packet_loss = random.uniform(0, parameters["max_packet_loss"])  # 最大丢包率限制（百分比）

    # 计算吞吐量，
    throughput = random.uniform(parameters["min_throughput"], parameters["max_throughput"])

    # 任务完成时间（假设任务完成时间与任务数量成正比）
    completion_time = random.uniform(0, parameters["max_completion_time"])  # 最大完成时间限制（秒）

    # 迭代次数（假设迭代次数与算法优化过程中的尝试次数相关）
    iterations = random.randint(1, parameters["max_iterations"])  # 最大迭代次数

    # 超时时间（假设超时时间控制任务的最大执行时间）
    timeout = random.uniform(0, parameters["timeout"])  # 超时时间限制（秒）

    # 用户体验：根据上述所有QoS参数计算，较小的QoS值代表较好的用户体验
    user_experience = 1 / (
            1.0 * response_time + 0.5 * storage_cost + 2.0 * latency + 1.0 * bandwidth +
            0.5 * jitter + 0.8 * packet_loss + 1.0 / throughput + 1.5 * completion_time
    )

    # 返回各项QoS参数和用户体验
    return response_time, storage_cost, latency, bandwidth, jitter, packet_loss, throughput, completion_time, iterations, timeout, user_experience


def check_constraints(self, cpu_usage_priv, cpu_usage_pub, mem_usage_priv, mem_usage_pub,user_experience_min):
        """
        检查所有约束条件，确保资源使用不超过限制，并满足QoS需求。
        """

        # 计算QoS参数并获取用户体验值
        response_time, storage_cost, latency, bandwidth, jitter, packet_loss, throughput, completion_time, iterations, timeout, user_experience = calculate_qos_parameters(
            self.parameters)

        # 检查用户体验值是否满足最低阈值
        if user_experience < user_experience_min:
            print(f"User experience ({user_experience}) is below the threshold.")
            return False
        # **CPU和内存资源约束**: 检查私有云和公有云的CPU和内存使用是否超过其最大容量。
        for i in range(self.parameters["M_priv"]):
            if cpu_usage_priv[i] > self.parameters["cpu_capacity_priv"]:
                print(f"Constraint failed: Private cloud CPU overused at host {i}.")
                return False
            if mem_usage_priv[i] > self.parameters["mem_capacity_priv"]:
                print(f"Constraint failed: Private cloud Memory overused at host {i}.")
                return False

        for j in range(self.parameters["M_pub"]):
            if cpu_usage_pub[j] > self.parameters["cpu_capacity_pub"]:
                print(f"Constraint failed: Public cloud CPU overused at host {j}.")
                return False
            if mem_usage_pub[j] > self.parameters["mem_capacity_pub"]:
                print(f"Constraint failed: Public cloud Memory overused at host {j}.")
                return False

        # **存储需求约束**: 检查存储成本是否超过最大允许的存储成本。
        if storage_cost > self.parameters["max_storage_cost"]:
            print(
                f"Constraint failed: Storage cost {storage_cost} exceeds max storage cost {self.parameters['max_storage_cost']}")
            return False

        # **响应时间约束**: 确保任务的响应时间不超过最大响应时间
        if response_time > self.parameters["max_response_time"]:
            print(
                f"Constraint failed: Response time {response_time} exceeds max response time {self.parameters['max_response_time']}")
            return False

        # **网络延迟约束**: 使用简化的延迟计算方法（不基于排队论）
        if latency > self.parameters["max_latency"]:
            print(f"Constraint failed: Latency {latency} exceeds max latency {self.parameters['max_latency']}")
            return False

        # **带宽约束**: 确保带宽需求不超过最大带宽
        if bandwidth > self.parameters["max_bandwidth"]:
            print(f"Constraint failed: Bandwidth {bandwidth} exceeds max bandwidth {self.parameters['max_bandwidth']}")
            return False

        # **网络抖动约束**: 确保网络抖动不超过给定的最大抖动
        if jitter > self.parameters["max_jitter"]:
            print(f"Constraint failed: Jitter {jitter} exceeds max jitter {self.parameters['max_jitter']}")
            return False

        # **丢包率约束**: 确保丢包率不超过最大丢包率
        if packet_loss > self.parameters["max_packet_loss"]:
            print(
                f"Constraint failed: Packet loss {packet_loss} exceeds max packet loss {self.parameters['max_packet_loss']}")
            return False

        # **吞吐量约束**: 确保吞吐量不低于最小吞吐量且不超过最大吞吐量
        if throughput < self.parameters["min_throughput"] or throughput > self.parameters["max_throughput"]:
            print(
                f"Constraint failed: Throughput {throughput} is out of range. "
                f"Expected between {self.parameters['min_throughput']} and {self.parameters['max_throughput']}.")
            return False

        # **任务完成时间约束**: 确保任务完成时间不超过最大完成时间
        if completion_time > self.parameters["max_completion_time"]:
            print(
                f"Constraint failed: Completion time {completion_time} exceeds max completion time {self.parameters['max_completion_time']}")
            return False

        # **超时约束**: 确保任务没有超时
        if timeout > self.parameters["timeout"]:
            print(f"Constraint failed: Timeout {timeout} exceeds max timeout {self.parameters['timeout']}")
            return False

        return True

def generate_initial_solution(parameters):
    """
    根据私有云和公有云的主机数量生成初始任务分配方案。
    """
    assignments = []
    num_tasks = parameters["N"]
    num_priv_hosts = parameters["M_priv"]
    num_pub_hosts = parameters["M_pub"]

    for _ in range(num_tasks):
        # 随机分配任务到私有云或公有云
        if random.uniform(0, 1) < 0.5:
            # 分配给私有云
            task = random.randint(0, num_priv_hosts - 1)
        else:
            # 分配给公有云
            task = random.randint(num_priv_hosts, num_priv_hosts + num_pub_hosts - 1)

        assignments.append(task)

    return assignments






















