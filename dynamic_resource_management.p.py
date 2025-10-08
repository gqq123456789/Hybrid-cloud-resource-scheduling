import pulp
import pandas as pd

# 1. 创建线性规划问题
model = pulp.LpProblem("Dynamic_Resource_Management", pulp.LpMinimize)

# 2. 假设一些参数
N = 10  # 请求数量
M_priv = 5  # 私有云控制器数量
M_pub = 5   # 公有云控制器数量

# 3. 初始化参数
lambda_t = [1] * N  # 请求到达率
A_priv = [5] * M_priv  # 私有云控制器的处理速率
A_pub = [5] * M_pub  # 公有云控制器的处理速率
s = [10] * N  # 请求存储需求
p = [0.1] * N  # 存储成本
nu_r = 1000  # 数据包传输速率
delta = 5  # 最大响应时间
gamma = 0.5  # 权重因子
S_max = 100  # 最大允许存储成本

# 4. 决策变量
x = pulp.LpVariable.dicts("x", (range(N), range(M_priv + M_pub)), cat='Binary')

# 5. 目标函数：最小化总通信及存储成本
D_priv = pulp.lpSum(2 * nu_r * (lambda_t[i] / (1 + j + 1)) * x[i][j] for i in range(N) for j in range(M_priv))
D_pub = pulp.lpSum(2 * nu_r * (lambda_t[i] / (1 + k + 1)) * x[i][k + M_priv] for i in range(N) for k in range(M_pub))
C_storage = pulp.lpSum(s[i] * p[i] * pulp.lpSum(x[i][j] for j in range(M_priv + M_pub)) for i in range(N))

model += gamma * D_priv + (1 - gamma) * D_pub + C_storage, "Total_Cost"

# 6. 约束条件
for i in range(N):
    model += pulp.lpSum(x[i][j] for j in range(M_priv + M_pub)) == 1, f"Connection_Constraint_{i}"

for j in range(M_priv):
    model += pulp.lpSum(lambda_t[i] * x[i][j] for i in range(N)) <= A_priv[j], f"Load_Constraint_Priv_{j}"

for k in range(M_pub):
    model += pulp.lpSum(lambda_t[i] * x[i][k + M_priv] for i in range(N)) <= A_pub[k], f"Load_Constraint_Pub_{k}"

model += (D_priv + D_pub) / N <= delta, "Response_Time_Constraint"
model += C_storage <= S_max, "Storage_Cost_Constraint"

# 7. 求解问题
model.solve()

# 8. 输出结果为表格
results = []
for i in range(N):
    for j in range(M_priv):
        if pulp.value(x[i][j]) > 0.5:
            results.append({"Request": i, "Controller": f"Private Controller {j}", "Type": "Private"})
    for k in range(M_pub):
        if pulp.value(x[i][k + M_priv]) > 0.5:
            results.append({"Request": i, "Controller": f"Public Controller {k}", "Type": "Public"})

df_results = pd.DataFrame(results)
print("\nConnection Results:")
print(df_results)

# 9. 成本计算
total_communication_cost = pulp.value(D_priv + D_pub)
total_storage_cost = pulp.value(C_storage)
total_cost = total_communication_cost + total_storage_cost

print(f"\nTotal Communication Cost: {total_communication_cost}")
print(f"Total Storage Cost: {total_storage_cost}")
print(f"Total Cost: {total_cost}")

# 10. 资源利用率
utilization_priv = [pulp.value(pulp.lpSum(lambda_t[i] * x[i][j] for i in range(N))) / A_priv[j] for j in range(M_priv)]
utilization_pub = [pulp.value(pulp.lpSum(lambda_t[i] * x[i][k + M_priv] for i in range(N))) / A_pub[k] for k in range(M_pub)]

print("\nResource Utilization:")
for j in range(M_priv):
    print(f"Private Controller {j}: {utilization_priv[j] * 100:.2f}%")
for k in range(M_pub):
    print(f"Public Controller {k}: {utilization_pub[k] * 100:.2f}%")

# 11. 时间/空间复杂度（简化示例）
time_complexity = "O(N * (M_priv + M_pub))"
space_complexity = "O(N * (M_priv + M_pub))"

print(f"\nTime Complexity: {time_complexity}")
print(f"Space Complexity: {space_complexity}")
