import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from workM import run_app

problem_instances = []

def analyze_vrp_results(problem_instances):

    
    df = pd.DataFrame(problem_instances, columns=["Problem Type", "Num Nodes", "Solution Quality", "Solve Time"])
    
    corr_matrix = df[["Num Nodes", "Solution Quality", "Solve Time"]].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation heatmap")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Num Nodes", y="Solve Time", hue="Problem Type", style="Problem Type")
    plt.xlabel("Problem scale (number of nodes)")
    plt.ylabel("Calculation time")
    plt.title("Problem dimension vs calculation time")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Problem Type", y="Solve Time")
    plt.xlabel("question type")
    plt.ylabel("Calculation time")
    plt.title("Distribution of computation time for different types of problems")
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

def test():
    
    file_path = r"P/P-n22-k8.vrp"
    optimal_solution_path = r"P/P-n22-k8.sol"
    run_app(file_path, optimal_solution_path)




root_folder = r"C:\Users\CZP\Desktop\启发式作业"
process_vrp_files(root_folder)
df_results = analyze_vrp_results(problem_instances)
print(df_results)

mean_deviation_A = df_results[df_results["Problem Type"] == "A"]["Solution Quality"].mean()
mean_deviation_M = df_results[df_results["Problem Type"] == "M"]["Solution Quality"].mean() if "M" in df_results["Problem Type"].values else None
mean_deviation_P = df_results[df_results["Problem Type"] == "P"]["Solution Quality"].mean() if "P" in df_results["Problem Type"].values else None

overall_mean_deviation = df_results["Solution Quality"].mean()


print(f"A 的平均偏差: {mean_deviation_A:.2f}")
if mean_deviation_M is not None:
    print(f"M 的平均偏差: {mean_deviation_M:.2f}")
if mean_deviation_P is not None:
    print(f"P 的平均偏差: {mean_deviation_P:.2f}")
print(f"整体平均偏差: {overall_mean_deviation:.2f}")
