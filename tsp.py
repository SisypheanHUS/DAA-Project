# tsp.py

import os
import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import csv

# Hàm phụ trợ
def load_tsp_data(filename):
    """Đọc file .tsp và tạo ma trận khoảng cách Euclidean."""
    print(f"Đang đọc file: {filename}")
    if not os.path.exists(filename):
        print(f"Lỗi: File {filename} không tồn tại trong thư mục hiện tại: {os.getcwd()}")
        return None, None, None
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"Lỗi: File {filename} trống")
                return None, None, None
            if 'NODE_COORD_SECTION' not in content or 'EOF' not in content:
                print(f"Lỗi: File {filename} có định dạng không đúng (thiếu NODE_COORD_SECTION hoặc EOF)")
                return None, None, None
        problem = tsplib95.load(filename)
        nodes = list(problem.get_nodes())
        coords = problem.node_coords
        n = len(nodes)
        print(f"Số thành phố: {n}")
        dist_matrix = np.zeros((n, n))
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    dist_matrix[i-1][j-1] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        print(f"Đọc file {filename} thành công")
        return nodes, coords, dist_matrix
    except Exception as e:
        print(f"Lỗi khi đọc file {filename}: {e}")
        return None, None, None

def tour_length(tour, dist_matrix):
    """Tính tổng độ dài chu trình TSP."""
    length = 0
    for i in range(len(tour)):
        length += dist_matrix[tour[i]][tour[(i+1) % len(tour)]]
    return length

def plot_tour(tour, coords, title, ax, dataset_name):
    """Vẽ chu trình TSP với các thành phố và đường nối."""
    n = len(tour)
    x = [coords[i][0] for i in range(1, n+1)]
    y = [coords[i][1] for i in range(1, n+1)]
    ax.scatter(x, y, c='blue', label='Thành phố')
    for i in range(n):
        start = tour[i]
        end = tour[(i+1) % n]
        ax.plot([coords[start+1][0], coords[end+1][0]], 
                [coords[start+1][1], coords[end+1][1]], 'r-')
    ax.set_title(f"{title}\\n{dataset_name}")
    ax.legend()

def save_results_to_csv(results, datasets, filename="tsp_results.csv"):
    """Lưu kết quả vào file CSV."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Algorithm', 'Tour Length', 'Error (%)', 'Time (s)'])
        for dataset, optimal_length in datasets.items():
            for algo, res in results[dataset].items():
                error = (res['length'] - optimal_length) / optimal_length * 100
                writer.writerow([dataset, algo, res['length'], error, res['time']])
    print(f"Kết quả đã được lưu vào {filename}")

def plot_comparison(results, datasets, metric='length', filename="comparison.png"):
    """Vẽ biểu đồ cột so sánh các thuật toán theo độ dài hoặc thời gian."""
    fig, ax = plt.subplots(figsize=(12, 6))
    algorithms = list(next(iter(results.values())).keys())
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        values = []
        for dataset in datasets:
            if metric == 'length':
                optimal_length = datasets[dataset]
                length = results[dataset][algo]['length']
                error = (length - optimal_length) / optimal_length * 100
                values.append(error)
            else:
                values.append(results[dataset][algo]['time'])
        ax.bar(x + i*width, values, width, label=algo)
    
    ax.set_xlabel('Bộ dữ liệu')
    ax.set_ylabel('Sai lệch (%)' if metric == 'length' else 'Thời gian (s)')
    ax.set_title(f'So sánh các thuật toán theo {"Sai lệch" if metric == "length" else "Thời gian"}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets.keys())
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Biểu đồ so sánh đã được lưu vào {filename}")

# Thuật toán
def tabu_search(dist_matrix, max_iterations=1000, tabu_size=10, convergence_log=None):
    """Tabu Search cho TSP."""
    n = len(dist_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    tabu_list = []
    
    for iteration in range(max_iterations):
        i, j = random.sample(range(n), 2)
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        if (i, j) not in tabu_list:
            new_length = tour_length(new_tour, dist_matrix)
            if new_length < best_length:
                best_tour = new_tour.copy()
                best_length = new_length
            tour = new_tour
            tabu_list.append((i, j))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
        
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

def two_opt(tour, dist_matrix):
    """Tìm kiếm cục bộ 2-opt để cải thiện chu trình."""
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)):
                new_tour = tour.copy()
                new_tour[i:j] = new_tour[i:j][::-1]
                new_length = tour_length(new_tour, dist_matrix)
                if new_length < best_length:
                    best_tour = new_tour.copy()
                    best_length = new_length
                    improved = True
        tour = best_tour.copy()
    return best_tour, best_length

def perturb(tour, num_swaps=3):
    """Nhiễu chu trình bằng hoán đổi ngẫu nhiên."""
    new_tour = tour.copy()
    for _ in range(num_swaps):
        i, j = random.sample(range(len(tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def ils(dist_matrix, max_iterations=100, convergence_log=None):
    """Iterated Local Search cho TSP."""
    n = len(dist_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    tour, best_length = two_opt(tour, dist_matrix)
    best_tour = tour.copy()
    
    for iteration in range(max_iterations):
        new_tour = perturb(tour)
        new_tour, new_length = two_opt(new_tour, dist_matrix)
        if new_length < best_length:
            best_tour = new_tour.copy()
            best_length = new_length
        tour = new_tour
        
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

def swap(tour, i, j):
    """Hoán đổi hai thành phố."""
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def reverse_segment(tour, i, j):
    """Đảo ngược một đoạn chu trình."""
    new_tour = tour.copy()
    new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour

def insert(tour, i, j):
    """Chèn một thành phố vào vị trí khác."""
    new_tour = tour.copy()
    city = new_tour.pop(i)
    new_tour.insert(j, city)
    return new_tour

def vns(dist_matrix, max_iterations=100, convergence_log=None):
    """Variable Neighborhood Search cho TSP."""
    n = len(dist_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    
    neighborhoods = [swap, reverse_segment, insert]
    
    for iteration in range(max_iterations):
        k = 0
        while k < len(neighborhoods):
            for _ in range(10):
                i, j = random.sample(range(n), 2)
                new_tour = neighborhoods[k](tour, i, j)
                new_length = tour_length(new_tour, dist_matrix)
                if new_length < best_length:
                    best_tour = new_tour.copy()
                    best_length = new_length
                    tour = new_tour
                    k = 0
                    break
            else:
                k += 1
        
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

def sa(dist_matrix, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, convergence_log=None):
    """Simulated Annealing cho TSP."""
    n = len(dist_matrix)
    tour = list(range(n))
    random.shuffle(tour)
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    temp = initial_temp
    
    for iteration in range(max_iterations):
        i, j = random.sample(range(n), 2)
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_length = tour_length(new_tour, dist_matrix)
        
        if new_length < best_length:
            best_tour = new_tour.copy()
            best_length = new_length
            tour = new_tour
        elif random.random() < math.exp((best_length - new_length) / temp):
            tour = new_tour
        
        temp *= cooling_rate
        
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

# Hàm chính
def main():
    """Chạy và so sánh các thuật toán trên dữ liệu TSPLIB."""
    print("Bắt đầu chạy main()...")
    print(f"Thư mục hiện tại: {os.getcwd()}")
    
    datasets = {
        'eil51.tsp': 426,
        'berlin52.tsp': 7542,
        'st70.tsp': 675,
        'lin318.tsp': 42029
    }
    
    print("Datasets:", list(datasets.keys()))
    
    # Kiểm tra file tồn tại trước khi chạy
    for dataset in datasets:
        if not os.path.exists(dataset):
            print(f"Cảnh báo: File {dataset} không tồn tại trong thư mục {os.getcwd()}")
        else:
            print(f"Tìm thấy file: {dataset}")
    
    algorithms = {
        'Tabu Search': lambda x: tabu_search(x, max_iterations=1000, tabu_size=10, convergence_log=[]),
        'Iterated Local Search': lambda x: ils(x, max_iterations=100, convergence_log=[]),
        'Variable Neighborhood Search': lambda x: vns(x, max_iterations=100, convergence_log=[]),
        'Simulated Annealing': lambda x: sa(x, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, convergence_log=[])
    }
    
    print("Algorithms:", list(algorithms.keys()))
    
    results = {}
    convergence_logs = {}
    
    for dataset, optimal_length in datasets.items():
        print(f"\n=== Xử lý {dataset} ===")
        nodes, coords, dist_matrix = load_tsp_data(dataset)
        if dist_matrix is None:
            print(f"Bỏ qua {dataset} do lỗi đọc file")
            continue
        
        print(f"Đọc thành công {dataset}, số thành phố: {len(nodes)}")
        
        results[dataset] = {}
        convergence_logs[dataset] = {}
        
        for name, algo in algorithms.items():
            print(f"Chạy {name}...")
            convergence_log = []
            start_time = time.time()
            tour, length = algo(dist_matrix)
            end_time = time.time()
            results[dataset][name] = {'tour': tour, 'length': length, 'time': end_time - start_time}
            convergence_logs[dataset][name] = convergence_log
            print(f"Hoàn thành {name} trên {dataset}, độ dài: {length:.2f}")
        
        print(f"Kết quả trên {dataset}:")
        for name, res in results[dataset].items():
            error = (res['length'] - optimal_length) / optimal_length * 100
            print(f"{name}: Độ dài = {res['length']:.2f}, Sai lệch = {error:.2f}%, Thời gian = {res['time']:.2f}s")
        
        print(f"Vẽ biểu đồ chu trình cho {dataset}...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        for idx, (name, res) in enumerate(results[dataset].items()):
            plot_tour(res['tour'], coords, f"{name}\\nĐộ dài: {res['length']:.2f}", axes[idx], dataset)
        plt.tight_layout()
        plt.savefig(f"{dataset}_tours.png")
        plt.show()
        print(f"Biểu đồ chu trình đã được lưu vào {dataset}_tours.png")
    
    print("Lưu kết quả vào CSV...")
    save_results_to_csv(results, datasets)
    
    print("Vẽ biểu đồ so sánh...")
    plot_comparison(results, datasets, metric='length', filename='error_comparison.png')
    plot_comparison(results, datasets, metric='time', filename='time_comparison.png')
    
    print("Vẽ biểu đồ hội tụ...")
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, log in convergence_logs[dataset].items():
            ax.plot(log, label=name)
        ax.set_xlabel('Lần lặp')
        ax.set_ylabel('Độ dài chu trình')
        ax.set_title(f'Hội tụ của các thuật toán trên {dataset}')
        ax.legend()
        plt.savefig(f"{dataset}_convergence.png")
        plt.show()
        print(f"Biểu đồ hội tụ đã được lưu vào {dataset}_convergence.png")

if __name__ == "__main__":
    main()