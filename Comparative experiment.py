from asa_empa import ASA_EMPA
from genetic_algorithm import GeneticAlgorithm
from particle_swarm_optimization import ParticleSwarmOptimization
from simulated_annealing import SimulatedAnnealing
from traditional_eamp import TraditionalEAMP
import random
import pandas as pd
from asa_empa import MDP

# 设置 pandas 显示不省略列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # 防止行被截断
def display_results(experiment_results):
    try:
        # 提取低、中、高流量下的成本和QoS指标
        def extract_metrics(traffic_type):
            cost = [
                experiment_results[traffic_type].get(alg, {}).get("Cost", [0])[0]
                for alg in experiment_results[traffic_type]
            ]
            qos = [
                experiment_results[traffic_type].get(alg, {}).get("QoS", [{}])[0]
                for alg in experiment_results[traffic_type]
            ]
            return cost, qos

        cost_low_traffic, qos_low_traffic = extract_metrics("Low Traffic")
        cost_medium_traffic, qos_medium_traffic = extract_metrics("Medium Traffic")
        cost_high_traffic, qos_high_traffic = extract_metrics("High Traffic")

        # 构建表格数据
        def build_dataframe(algorithms, cost, qos):
            return pd.DataFrame({
                "Algorithm": algorithms,
                "Cost": cost,
                "Latency (ms)": [item.get("latency", "N/A") if isinstance(item, dict) else "N/A" for item in qos],
                "Throughput (tasks/s)": [item.get("throughput", "N/A") if isinstance(item, dict) else "N/A" for item in
                                         qos],
                "Jitter (ms)": [item.get("jitter", "N/A") if isinstance(item, dict) else "N/A" for item in qos],
                "Bandwidth Utilization (%)": [
                    item.get("bandwidth_utilization", "N/A") * 100 if isinstance(item, dict) else "N/A" for item in
                    qos],
                "Response Time (s)": [item.get("response_time", "N/A") if isinstance(item, dict) else "N/A" for item in
                                      qos]
            })

        algorithms = list(experiment_results["Low Traffic"].keys())

        low_traffic_df = build_dataframe(algorithms, cost_low_traffic, qos_low_traffic)
        medium_traffic_df = build_dataframe(algorithms, cost_medium_traffic, qos_medium_traffic)
        high_traffic_df = build_dataframe(algorithms, cost_high_traffic, qos_high_traffic)

        # 打印或返回表格
        print("Low Traffic Results:")
        print(low_traffic_df)
        print("\nMedium Traffic Results:")
        print(medium_traffic_df)
        print("\nHigh Traffic Results:")
        print(high_traffic_df)

    except Exception as e:
        print(f"Error in display_results: {e}")




class ExperimentMain:
    def __init__(self, parameters , costs, storage_demand):
        self.parameters = parameters
        self.costs = costs
        self.storage_demand = storage_demand
        self.algorithms = {
            "ASA-EAMP": ASA_EMPA(self.parameters, self.costs, self.storage_demand),
            "Genetic Algorithm": GeneticAlgorithm(self.parameters, self.costs, self.storage_demand),
            "Particle Swarm Optimization": ParticleSwarmOptimization(self.parameters, self.costs, self.storage_demand),
            "Simulated Annealing": SimulatedAnnealing(self.parameters, self.costs, self.storage_demand),
            "Traditional EAMP": TraditionalEAMP(self.parameters, self.costs, self.storage_demand)
        }

    def create_mdp(self):
        # 创建并返回 MDP 对象
        states = range(self.parameters["M_priv"] + self.parameters["M_pub"])
        actions = range(self.parameters["M_priv"] + self.parameters["M_pub"])
        return MDP(states, actions, self.parameters["discount_factor"], self.parameters["learning_rate"])

    def run_experiments(self):
        """
        根据不同的负载条件运行实验（低、中、高负载）。
        traffic_conditions: 定义的负载条件，包括用户数量、带宽需求范围、延迟、抖动和响应时间。
        """
        # 初始化三个字典以记录不同负载条件下的结果
        experiment_results = {"Low Traffic": {}, "Medium Traffic": {}, "High Traffic": {}}

        # 定义负载条件
        traffic_conditions = {
            "Low Traffic": {
                "users": int(self.parameters["N"] * 0.3),
                "bandwidth_range": (1000, 2000),
                "latency_range": (0, 50),
                "jitter_range": (0, 5),
                "response_time_range": (0, 0.3),
            },
            "Medium Traffic": {
                "users": int(self.parameters["N"] * 0.7),
                "bandwidth_range": (2000, 5000),
                "latency_range": (50, 100),
                "jitter_range": (5, 20),
                "response_time_range": (0.3, 0.5),
            },
            "High Traffic": {
                "users": self.parameters["N"],
                "bandwidth_range": (5000, 10000),
                "latency_range": (100, 300),
                "jitter_range": (20, 150),
                "response_time_range": (0.5, 10),
            },
        }

        for name, algorithm in self.algorithms.items():
            print(f"\nRunning {name}...")

            # 遍历负载条件
            for load_type, condition in traffic_conditions.items():
                # 从条件中获取用户数和各个范围
                num_users = condition["users"]
                bandwidth_min, bandwidth_max = condition["bandwidth_range"]
                latency_min, latency_max = condition["latency_range"]
                jitter_min, jitter_max = condition["jitter_range"]
                response_time_min, response_time_max = condition["response_time_range"]

                # 动态生成任务参数
                lambda_t = [random.uniform(1, 5) for _ in range(num_users)]  # 每个任务的到达率
                bandwidth_demand = [random.uniform(bandwidth_min, bandwidth_max) for _ in range(num_users)]  # 带宽需求
                latency = [random.uniform(latency_min, latency_max) for _ in range(num_users)]  # 延迟
                jitter = [random.uniform(jitter_min, jitter_max) for _ in range(num_users)]  # 抖动
                response_time = [random.uniform(response_time_min, response_time_max) for _ in range(num_users)]  # 响应时间
                s = [random.uniform(1, 3) for _ in range(num_users)]  # 每个任务的内存需求（GB）
                p = [random.uniform(1, 3) for _ in range(num_users)]  # 每个任务的CPU需求（核数）
                A_priv = [0] * self.parameters["M_priv"]  # 初始化私有云的任务分配情况
                A_pub = [0] * self.parameters["M_pub"]  # 初始化公有云的任务分配情况

                # 使用算法优化
                best_assignments, best_cost, qos_metrics = algorithm.optimize(
                    lambda_t, s, p, A_priv, A_pub, bandwidth_demand
                )
                # 打印 QoS 指标
                print(f"QoS Metrics for {name} ({load_type}): {qos_metrics}")

                # 确保负载类型存在于结果字典中
                if load_type not in experiment_results:
                    experiment_results[load_type] = {}
                if name not in experiment_results[load_type]:
                    experiment_results[load_type][name] = {"Cost": [], "QoS": []}

                # 存储结果
                experiment_results[load_type][name]["Cost"].append(best_cost)
                experiment_results[load_type][name]["QoS"].append(qos_metrics)
                print(f"Updated {load_type} -> {name}: {experiment_results[load_type][name]}")

        return experiment_results


if __name__ == "__main__":
    # 设置实验参数和成本
    parameters = {
        "N": 100,  # 系统中的用户数量，应该与 lambda_t 的长度一致
        "M_priv": 90,  # 私有云资源池中的控制器数量
        "M_pub": 100,  # 公有云资源池中的控制器数量
        # 将每个控制器的CPU容量单位从核数改为颗（1颗 = 8核），根据硬件条件
        "cpu_capacity_priv": 250, # 私有云每个控制器的最大CPU容量（单位：颗，即 1 颗cpu 等于8核）
        "cpu_capacity_pub": 500,  # 公有云每个控制器的最大CPU容量（单位：颗，即 1 颗cpu 等于8核）
        "mem_capacity_priv": 250,  # 私有云每个控制器的最大内存容量（单位：GB）
        "mem_capacity_pub": 500,  # 公有云每个控制器的最大内存容量（单位：GB）
        "service_rate_priv": 10,  # 私有云控制器的服务速率（单位：请求/秒）
        "service_rate_pub": 5,  # 公有云控制器的服务速率（单位：请求/秒）
        "max_response_time": 10,  # 最大响应时间（单位：秒）
        "max_storage_cost": 10000,  # 最大存储成本（单位：美元）
        "max_latency": 300,  # 最大延迟（单位：毫秒）
        "max_bandwidth": 1000,  # 最大带宽（单位：GB，假设网络带宽为 10 Gbps，约合 10000 GB）
        "max_jitter": 150,  # 最大抖动（单位：毫秒）
        "max_packet_loss": 0.3,  # 最大丢包率（单位：百分比）
        "min_throughput": 3,  # 最小吞吐量（单位：GB/s）
        "max_throughput": 10,  # 最大吞吐量（单位：GB/s）
        "omega": 0.5,  # 调节因子（可能用于平衡不同因素的权重）
        "discount_factor": 0.9,  # 折扣因子（用于强化学习中的长期奖励计算）
        "learning_rate": 0.1,  # 学习率（用于强化学习中的更新步骤）
        "max_completion_time": 60,  # 最大完成时间（单位：秒）
        "max_iterations": 100,  # 最大迭代次数（在优化过程中控制搜索的深度）
        "timeout": 30,  # 超时时间（单位：秒，用于控制任务的最大执行时间）
        "user_experience_min": 0.00001  # 用户体验最低阈值
    }

    # 修改后的 costs 参数（单位：美元）
    costs = {
        "public": {
            "cpu": 0.25,  # 公有云每颗CPU的成本（单位：美元，假设公有云的资源成本较高）
            "mem": 0.04,  # 公有云每GB内存的成本（单位：美元）
            "net": 0.015,  # 公有云每GB网络流量的成本（单位：美元）
            "storage": 0.06,  # 公有云每GB存储的成本（单位：美元）
            "bandwidth": 0.05,  # 公有云带宽成本（单位：美元/GB）
            "power": 0.02  # 公有云每小时电力成本（单位：美元）
        },
        "private": {
            "cpu": 0.12,  # 私有云每颗CPU的成本（单位：美元，私有云较便宜）
            "mem": 0.02,  # 私有云每GB内存的成本（单位：美元）
            "net": 0.005,  # 私有云每GB网络流量的成本（单位：美元）
            "storage": 0.03,  # 私有云每GB存储的成本（单位：美元）
            "bandwidth": 0.01,  # 私有云带宽成本（单位：美元/GB）
            "power": 0.01  # 私有云每小时电力成本（单位：美元）
        }
    }


    # 为每个用户生成随机的存储需求（单位：GB）
    storage_demand = [random.randint(100, 200) for _ in range(parameters["N"])]
    # 初始化实验类，传入参数、成本和存储需求
    experiment = ExperimentMain(parameters, costs, storage_demand)
    # 设置流量条件：低流量和高流量
    traffic_conditions = ["Low Traffic", "Medium Traffic", "High Traffic"]
    # 设置用户体验阈值
    user_experience_threshold = 0.6
    # 运行实验并返回结果
    results = experiment.run_experiments()
    # 调用 display_results 函数并传入 results 参数来显示实验结果
    display_results(results)
