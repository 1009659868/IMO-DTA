import matplotlib.pyplot as plt
import numpy as np

# 读取数据函数
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', ' ').split()
        data = list(map(float, data))  # 转换为浮点数
    return np.array(data)

if __name__ == "__main__":
    
    for i in range(5):
        print(f"No.{i}-----------------------------------------")
        # 读取数据
        affinity = f'./test/iter{i}affinities.txt'
        affinities = read_data(affinity)
        
        preaffinity = f'./test/iter{i}preaffinities.txt'
        preaffinities = read_data(preaffinity)
        
        true_values=affinities
        predicted_values=preaffinities
        # 绘制拟合图
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, color='blue', label='True vs Predicted')
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
        plt.title('')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 显示图像
        plt.savefig(f"./test/No_{i}.png")
