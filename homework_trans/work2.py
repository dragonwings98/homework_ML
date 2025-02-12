import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from workM import run_app

problem_instances = []
# 设置字体
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def analyze_vrp_results(problem_instances):

    
    df = pd.DataFrame(problem_instances, columns=["Problem Type", "Num Nodes", "Solution Quality", "Solve Time"])
    
    corr_matrix = df[["Num Nodes", "Solution Quality", "Solve Time"]].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation heat map")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Num Nodes", y="Solve Time", hue="Problem Type", style="Problem Type")
    plt.xlabel("Problem size (number of nodes)")
    plt.ylabel("calculation time (s)")
    plt.title("Problem dimension vs computation time")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Problem Type", y="Solve Time")
    plt.xlabel("Problem type")
    plt.ylabel("Calculation time (s)")
    plt.title("Calculation time distributions for different problem types")
    plt.show()

    return df




def process_vrp_files(root_folder):
    results = []
    
    for folder in ["A", "M", "P"]:
        folder_path = os.path.join(root_folder, folder)
        if not os.path.exists(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".vrp"):
                vrp_file = os.path.join(folder_path, file_name)
                sol_file = vrp_file.replace(".vrp", ".sol")
                print(file_name)
                solution_quality, solve_time, num_nodes = run_app(vrp_file, sol_file)
                problem_instances.append([folder, num_nodes, solution_quality, solve_time])

    return results


# **运行代码**
root_folder = r"homework_trans"
process_vrp_files(root_folder)
df_results = analyze_vrp_results(problem_instances)
print(df_results)
