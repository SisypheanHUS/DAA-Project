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
    ax.set_title(f'So sánh các thuật toán theo {"Sai lệch" if metric == 'length' else 'Thời gian'}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets.keys())
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Biểu đồ so sánh đã được lưu vào {filename}")

# Thuật toán
def tabu_search(dist_matrix, max_iterations=1000, tabu_size=20, convergence_log=None):
    """Tabu Search cho TSP."""
    n = len(dist_matrix)
    # Better initialization with nearest neighbor
    tour = nearest_neighbor(dist_matrix, random.randint(0, n-1))
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    current_length = best_length
    tabu_list = {}  # Dictionary to store move:expiration_time
    non_improvement = 0
    max_non_improvement = 100
    
    for iteration in range(max_iterations):
        # Find best non-tabu move
        best_move = None
        best_move_length = float('inf')
        candidates = []
        
        # Generate candidate moves
        for i in range(n):
            for j in range(i+1, n):
                # Swap
                candidates.append((i, j, "swap"))
                # Reverse segment
                candidates.append((i, j, "reverse"))
        
        # Evaluate all candidates
        random.shuffle(candidates)  # Randomize to avoid bias
        for i, j, move_type in candidates[:50]:  # Limit to 50 candidates for efficiency
            new_tour = tour.copy()
            if move_type == "swap":
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            else:  # reverse
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                
            new_length = tour_length(new_tour, dist_matrix)
            
            # Check if move is tabu
            tabu_key = (i, j, move_type)
            reverse_key = (j, i, move_type)
            is_tabu = (tabu_key in tabu_list and iteration < tabu_list[tabu_key]) or \
                      (reverse_key in tabu_list and iteration < tabu_list[reverse_key])
            
            # Accept if best move so far and either not tabu or satisfies aspiration criteria
            if (not is_tabu or new_length < best_length) and new_length < best_move_length:
                best_move = (i, j, move_type, new_tour, new_length)
                best_move_length = new_length
        
        # Apply best move
        if best_move:
            i, j, move_type, new_tour, new_length = best_move
            tour = new_tour
            current_length = new_length
            
            # Update tabu list
            tabu_list[(i, j, move_type)] = iteration + tabu_size + random.randint(0, 5)
            
            # Update best solution if improved
            if new_length < best_length:
                best_tour = new_tour.copy()
                best_length = new_length
                non_improvement = 0
            else:
                non_improvement += 1
        else:
            non_improvement += 1
        
        # Diversification if stuck
        if non_improvement >= max_non_improvement:
            tour = perturb_heavily(tour)
            current_length = tour_length(tour, dist_matrix)
            non_improvement = 0
        
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

def perturb_heavily(tour):
    """More aggressive perturbation for diversification."""
    n = len(tour)
    new_tour = tour.copy()
    
    # Apply multiple random moves
    for _ in range(int(n * 0.3)):  # Perturb 30% of the tour
        move_type = random.choice(["swap", "reverse", "insert"])
        
        if move_type == "swap":
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        elif move_type == "reverse":
            i, j = sorted(random.sample(range(n), 2))
            if j - i > 1:
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
        else:  # insert
            i, j = random.sample(range(n), 2)
            city = new_tour.pop(i)
            new_tour.insert(j, city)
    
    return new_tour

def ils(dist_matrix, max_iterations=100, convergence_log=None):
    """Iterated Local Search cho TSP."""
    n = len(dist_matrix)
    # Better initialization with nearest neighbor + 2-opt
    tour = nearest_neighbor(dist_matrix, random.randint(0, n-1))
    tour, tour_len = two_opt(tour, dist_matrix)
    
    best_tour = tour.copy()
    best_length = tour_len
    
    no_improvement = 0
    max_no_improvement = 20
    
    for iteration in range(max_iterations):
        # Perturb solution based on how long we've been stuck
        if no_improvement < 10:
            new_tour = perturb_adaptive(tour, int(n*0.1))
        else:
            new_tour = perturb_adaptive(tour, int(n*0.2))
            
        # Apply local search
        new_tour, new_length = two_opt_improved(new_tour, dist_matrix)
        
        # Acceptance criterion (only accept if better or with small probability)
        if new_length < tour_len or random.random() < 0.1:
            tour = new_tour
            tour_len = new_length
            
            # Update best solution if improved
            if new_length < best_length:
                best_tour = new_tour.copy()
                best_length = new_length
                no_improvement = 0
            else:
                no_improvement += 1
        else:
            no_improvement += 1
            
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

def perturb_adaptive(tour, num_moves):
    """More intelligent perturbation that avoids tiny changes."""
    new_tour = tour.copy()
    n = len(tour)
    
    moves = random.randint(max(3, num_moves-5), num_moves+5)  # Randomize intensity
    
    for _ in range(moves):
        move_type = random.choice(["double_bridge", "swap", "reverse"])
        
        if move_type == "double_bridge" and n > 7:  # Double bridge (4-opt) move
            points = sorted(random.sample(range(n), 4))
            p1, p2, p3, p4 = points
            # Apply 4-opt move: [0:p1] + [p3:p4] + [p2:p3] + [p1:p2] + [p4:]
            new_tour = new_tour[:p1] + new_tour[p3:p4] + new_tour[p2:p3] + new_tour[p1:p2] + new_tour[p4:]
        elif move_type == "swap":
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        else:  # reverse
            i, j = sorted(random.sample(range(n), 2))
            if j - i > 1:  # Only reverse if segment has length > 1
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
    
    return new_tour

def two_opt_improved(tour, dist_matrix):
    """Faster 2-opt implementation with first improvement strategy."""
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    improved = True
    n = len(tour)
    
    while improved:
        improved = False
        for i in range(1, n-2):
            # Only check a subset of possible moves for efficiency
            for j in range(i+1, min(i+20, n)):
                if j - i == 1:
                    continue  # Skip adjacent cities
                    
                # Calculate gain directly without full tour calculation
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[(j+1) % n]
                
                # If we replace edges (a,b) and (c,d) with (a,c) and (b,d)
                current_distance = dist_matrix[a][b] + dist_matrix[c][d]
                new_distance = dist_matrix[a][c] + dist_matrix[b][d]
                
                if new_distance < current_distance:
                    new_tour = best_tour.copy()
                    new_tour[i:j+1] = new_tour[i:j+1][::-1]
                    new_length = tour_length(new_tour, dist_matrix)  # Verify gain
                    
                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True
                        break  # First improvement
            
            if improved:
                break
    
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
    # Better initialization
    tour = nearest_neighbor(dist_matrix, random.randint(0, n-1))
    tour, tour_len = two_opt(tour, dist_matrix)
    
    best_tour = tour.copy()
    best_length = tour_len
    current_tour = tour
    current_length = tour_len
    
    # Define neighborhoods and their functions
    neighborhoods = [
        {"name": "swap", "function": lambda t, i, j: swap(t, i, j)},
        {"name": "reverse", "function": lambda t, i, j: reverse_segment(t, i, j)},
        {"name": "insert", "function": lambda t, i, j: insert(t, i, j)},
        {"name": "2-opt", "function": lambda t, i, j: two_opt_single_move(t, i, j)}
    ]
    
    for iteration in range(max_iterations):
        k = 0
        while k < len(neighborhoods):
            # Shaking phase - perturb the solution using kth neighborhood
            i, j = sorted(random.sample(range(n), 2))
            if j - i < 2:  # Ensure meaningful moves
                j = (i + 2) % n
                
            new_tour = neighborhoods[k]["function"](current_tour, i, j)
            
            # Local search phase - apply 2-opt to improve the solution
            new_tour, new_length = two_opt_first_improvement(new_tour, dist_matrix)
            
            # Move or not
            if new_length < current_length:
                current_tour = new_tour
                current_length = new_length
                
                # Update best solution
                if new_length < best_length:
                    best_tour = new_tour.copy()
                    best_length = new_length
                
                # Return to first neighborhood
                k = 0
            else:
                # Try next neighborhood
                k += 1
        
        # Apply perturbation to escape local optima
        if iteration % 10 == 0:
            current_tour = perturb_adaptive(current_tour, int(n*0.15))
            current_length = tour_length(current_tour, dist_matrix)
        
        if convergence_log is not None:
            convergence_log.append(best_length)
    
    return best_tour, best_length

def two_opt_single_move(tour, i, j):
    """Apply a single 2-opt move."""
    new_tour = tour.copy()
    new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour

def two_opt_first_improvement(tour, dist_matrix):
    """2-opt with first improvement strategy for VNS."""
    best_tour = tour.copy()
    best_length = tour_length(tour, dist_matrix)
    n = len(tour)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                new_tour = best_tour.copy()
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                new_length = tour_length(new_tour, dist_matrix)
                
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break  # First improvement
                    
            if improved:
                break
                
    return best_tour, best_length

def nearest_neighbor(dist_matrix, start_idx=0):
    """Create initial tour using nearest neighbor heuristic."""
    n = len(dist_matrix)
    tour = [start_idx]
    unvisited = set(range(n))
    unvisited.remove(start_idx)
    
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda i: dist_matrix[current][i])
        tour.append(nearest)
        unvisited.remove(nearest)
        
    return tour

def sa(dist_matrix, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, convergence_log=None):
    """Simulated Annealing cho TSP."""
    n = len(dist_matrix)
    # Initialize with nearest neighbor for better starting tour
    tour = nearest_neighbor(dist_matrix, random.randint(0, n-1))
    current_tour = tour.copy()
    current_length = tour_length(current_tour, dist_matrix)
    
    best_tour = tour.copy()
    best_length = current_length
    temp = initial_temp
    non_improving_iterations = 0
    
    for iteration in range(max_iterations):
        # Use multiple neighborhood types randomly
        move_type = random.choice(["swap", "reverse", "insert"])
        
        if move_type == "swap":
            i, j = random.sample(range(n), 2)
            new_tour = current_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        elif move_type == "reverse":
            i, j = sorted(random.sample(range(n), 2))
            new_tour = current_tour.copy()
            new_tour[i:j+1] = new_tour[i:j+1][::-1]
        else:  # insert
            i, j = random.sample(range(n), 2)
            new_tour = current_tour.copy()
            city = new_tour.pop(i)
            new_tour.insert(j, city)
            
        new_length = tour_length(new_tour, dist_matrix)
        delta = new_length - current_length
        
        # Acceptance criterion (fixed)
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = new_tour
            current_length = new_length
            
            # Update best solution if improved
            if new_length < best_length:
                best_tour = new_tour.copy()
                best_length = new_length
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1
        else:
            non_improving_iterations += 1
        
        # Reheating if stuck in local optimum
        if non_improving_iterations > 100:
            temp = initial_temp * 0.5
            non_improving_iterations = 0
        else:
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
        #'lin318.tsp': 42029
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