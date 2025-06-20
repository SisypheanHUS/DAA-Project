import os
import tsplib95
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import csv
from typing import Dict, List, Tuple, Optional, Callable

# Hàm tiện ích
def load_tsp_data(filepath: str) -> Tuple[Optional[List[int]], Optional[Dict[int, Tuple[float, float]]], Optional[np.ndarray]]:
    """Đọc dữ liệu TSP từ file và tính ma trận khoảng cách Euclidean.

    Args:
        filepath: Đường dẫn đến file TSP.

    Returns:
        Bộ ba chứa danh sách node, tọa độ, và ma trận khoảng cách, hoặc (None, None, None) nếu có lỗi.
    """
    if not os.path.exists(filepath):
        print(f"Lỗi: Không tìm thấy file '{filepath}' trong thư mục {os.getcwd()}")
        return None, None, None
    try:
        problem = tsplib95.load(filepath)
        nodes = list(problem.get_nodes())
        coords = problem.node_coords
        num_nodes = len(nodes)
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                xi, yi = coords[i + 1]
                xj, yj = coords[j + 1]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        print(f"Đã tải '{filepath}' với {num_nodes} thành phố")
        return nodes, coords, dist_matrix
    except Exception as e:
        print(f"Lỗi khi tải '{filepath}': {e}")
        return None, None, None

def compute_tour_length(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Tính tổng độ dài chu trình TSP.

    Args:
        tour: Danh sách các chỉ số node tạo thành chu trình.
        dist_matrix: Ma trận khoảng cách đã tính trước.

    Returns:
        Tổng độ dài chu trình.
    """
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def plot_tour(tour: List[int], coords: Dict[int, Tuple[float, float]], title: str, ax: plt.Axes, dataset_name: str) -> None:
    """Vẽ chu trình TSP với các thành phố và đường nối.

    Args:
        tour: Danh sách các chỉ số node.
        coords: Từ điển chứa tọa độ của các node.
        title: Tiêu đề biểu đồ.
        ax: Đối tượng trục của Matplotlib.
        dataset_name: Tên bộ dữ liệu.
    """
    num_nodes = len(tour)
    x_coords = [coords[i + 1][0] for i in range(num_nodes)]
    y_coords = [coords[i + 1][1] for i in range(num_nodes)]
    ax.scatter(x_coords, y_coords, c='blue', label='Thành phố')
    for i in range(num_nodes):
        start, end = tour[i], tour[(i + 1) % num_nodes]
        ax.plot([coords[start + 1][0], coords[end + 1][0]],
                [coords[start + 1][1], coords[end + 1][1]], 'r-')
    ax.set_title(f"{title}\n{dataset_name}")
    ax.legend()

def save_results_to_csv(results: Dict, datasets: Dict, filepath: str = "tsp_results.csv") -> None:
    """Lưu kết quả thuật toán vào file CSV.

    Args:
        results: Từ điển chứa kết quả của từng bộ dữ liệu và thuật toán.
        datasets: Từ điển chứa tên bộ dữ liệu và độ dài tối ưu.
        filepath: Đường dẫn file CSV đầu ra.
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Bộ dữ liệu', 'Thuật toán', 'Độ dài chu trình', 'Sai lệch (%)', 'Thời gian (s)'])
        for dataset, optimal_length in datasets.items():
            for algo, res in results[dataset].items():
                error = (res['length'] - optimal_length) / optimal_length * 100
                writer.writerow([dataset, algo, res['length'], f"{error:.2f}", f"{res['time']:.2f}"])
    print(f"Kết quả đã được lưu vào '{filepath}'")

def plot_comparison(results: Dict, datasets: Dict, metric: str = 'length', filepath: str = "comparison.png") -> None:
    """Vẽ biểu đồ cột so sánh các thuật toán theo chỉ số được chọn.

    Args:
        results: Từ điển chứa kết quả của từng bộ dữ liệu và thuật toán.
        datasets: Từ điển chứa tên bộ dữ liệu và độ dài tối ưu.
        metric: Chỉ số so sánh ('length' hoặc 'time').
        filepath: Đường dẫn file đầu ra của biểu đồ.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    algorithms = list(next(iter(results.values())).keys())
    dataset_names = list(datasets.keys())
    x = np.arange(len(dataset_names))
    width = 0.2
    for i, algo in enumerate(algorithms):
        values = [
            (results[ds][algo]['length'] - datasets[ds]) / datasets[ds] * 100 if metric == 'length'
            else results[ds][algo]['time']
            for ds in dataset_names
        ]
        ax.bar(x + i * width, values, width, label=algo)
    ax.set_xlabel('Bộ dữ liệu')
    ax.set_ylabel('Sai lệch (%)' if metric == 'length' else 'Thời gian (s)')
    ax.set_title(f"So sánh thuật toán theo {'Sai lệch' if metric == 'length' else 'Thời gian'}")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Biểu đồ so sánh đã được lưu vào '{filepath}'")

# Triển khai thuật toán
def nearest_neighbor(dist_matrix: np.ndarray, start: int) -> List[int]:
    """Tạo chu trình ban đầu bằng heuristic nearest neighbor.

    Args:
        dist_matrix: Ma trận khoảng cách đã tính trước.
        start: Chỉ số node bắt đầu.

    Returns:
        Chu trình ban đầu dưới dạng danh sách các chỉ số node.
    """
    num_nodes = len(dist_matrix)
    tour = [start]
    unvisited = set(range(num_nodes)) - {start}
    while unvisited:
        current = tour[-1]
        next_node = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
    return tour

def tabu_search(dist_matrix: np.ndarray, max_iterations: int = 1000, tabu_tenure: int = 20) -> Tuple[List[int], float]:
    """Giải TSP bằng Tabu Search với tenure động và danh sách ứng viên.

    Args:
        dist_matrix: Ma trận khoảng cách đã tính trước.
        max_iterations: Số lần lặp tối đa.
        tabu_tenure: Kích thước cơ bản của danh sách tabu.

    Returns:
        Chu trình tốt nhất và độ dài của nó.
    """
    num_nodes = len(dist_matrix)
    tour = nearest_neighbor(dist_matrix, random.randint(0, num_nodes - 1))
    best_tour, best_length = tour[:], compute_tour_length(tour, dist_matrix)
    tabu_list: Dict[Tuple[int, int, str], int] = {}
    non_improvement = 0
    max_non_improvement = 100

    for iteration in range(max_iterations):
        candidates = [(i, j, "swap") for i in range(num_nodes) for j in range(i + 1, num_nodes)][:50]
        random.shuffle(candidates)
        best_move = None
        best_new_length = float('inf')

        for i, j, move_type in candidates:
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_length = compute_tour_length(new_tour, dist_matrix)
            move_key = (i, j, move_type)
            is_tabu = move_key in tabu_list and iteration < tabu_list[move_key]
            if (not is_tabu or new_length < best_length) and new_length < best_new_length:
                best_move = (new_tour, new_length, move_key)
                best_new_length = new_length

        if best_move:
            tour, best_length_candidate, move_key = best_move
            tabu_list[move_key] = iteration + tabu_tenure + random.randint(0, 5)
            if best_length_candidate < best_length:
                best_tour, best_length = tour[:], best_length_candidate
                non_improvement = 0
            else:
                non_improvement += 1
        else:
            non_improvement += 1

        if non_improvement >= max_non_improvement:
            tour = perturb_heavily(tour, num_nodes)
            non_improvement = 0

    return best_tour, best_length

def two_opt(tour: List[int], dist_matrix: np.ndarray) -> Tuple[List[int], float]:
    """Cải thiện chu trình bằng tìm kiếm cục bộ 2-opt với chiến lược cải thiện đầu tiên.

    Args:
        tour: Chu trình ban đầu.
        dist_matrix: Ma trận khoảng cách đã tính trước.

    Returns:
        Chu trình được cải thiện và độ dài của nó.
    """
    best_tour, best_length = tour[:], compute_tour_length(tour, dist_matrix)
    improved = True
    num_nodes = len(tour)
    while improved:
        improved = False
        for i in range(1, num_nodes - 2):
            for j in range(i + 2, num_nodes):
                delta = (dist_matrix[best_tour[i - 1], best_tour[j]] + dist_matrix[best_tour[i], best_tour[(j + 1) % num_nodes]]) - \
                        (dist_matrix[best_tour[i - 1], best_tour[i]] + dist_matrix[best_tour[j], best_tour[(j + 1) % num_nodes]])
                if delta < -1e-10:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_length += delta
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_length

def perturb_heavily(tour: List[int], num_nodes: int) -> List[int]:
    """Nhiễu mạnh chu trình để thoát khỏi cực trị cục bộ.

    Args:
        tour: Chu trình hiện tại.
        num_nodes: Số lượng node trong chu trình.

    Returns:
        Chu trình bị nhiễu.
    """
    new_tour = tour[:]
    for _ in range(int(num_nodes * 0.3)):
        i, j = random.sample(range(num_nodes), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def iterated_local_search(dist_matrix: np.ndarray, max_iterations: int = 100) -> Tuple[List[int], float]:
    """Giải TSP bằng Iterated Local Search với nhiễu thích nghi.

    Args:
        dist_matrix: Ma trận khoảng cách đã tính trước.
        max_iterations: Số lần lặp tối đa.

    Returns:
        Chu trình tốt nhất và độ dài của nó.
    """
    num_nodes = len(dist_matrix)
    tour = nearest_neighbor(dist_matrix, random.randint(0, num_nodes - 1))
    tour, best_length = two_opt(tour, dist_matrix)
    best_tour = tour[:]
    no_improvement = 0
    max_no_improvement = 20

    for _ in range(max_iterations):
        perturbation_size = int(num_nodes * (0.1 if no_improvement < 10 else 0.2))
        new_tour = perturb_adaptive(tour, perturbation_size)
        new_tour, new_length = two_opt(new_tour, dist_matrix)
        if new_length < best_length:
            tour, best_tour, best_length = new_tour[:], new_tour[:], new_length
            no_improvement = 0
        else:
            tour = new_tour if random.random() < 0.1 else tour
            no_improvement += 1
        if no_improvement >= max_no_improvement:
            tour = perturb_heavily(tour, num_nodes)
            no_improvement = 0

    return best_tour, best_length

def perturb_adaptive(tour: List[int], num_moves: int) -> List[int]:
    """Nhiễu chu trình một cách thích nghi với các loại di chuyển khác nhau.

    Args:
        tour: Chu trình hiện tại.
        num_moves: Số lượng di chuyển nhiễu.

    Returns:
        Chu trình bị nhiễu.
    """
    new_tour = tour[:]
    num_nodes = len(tour)
    moves = random.randint(max(3, num_moves - 5), num_moves + 5)
    for _ in range(moves):
        move_type = random.choice(["swap", "reverse"])
        i, j = random.sample(range(num_nodes), 2)
        if move_type == "swap":
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        else:
            i, j = sorted([i, j])
            if j - i > 1:
                new_tour[i:j + 1] = new_tour[i:j + 1][::-1]
    return new_tour

def variable_neighborhood_search(dist_matrix: np.ndarray, max_iterations: int = 100) -> Tuple[List[int], float]:
    """Giải TSP bằng Variable Neighborhood Search.

    Args:
        dist_matrix: Ma trận khoảng cách đã tính trước.
        max_iterations: Số lần lặp tối đa.

    Returns:
        Chu trình tốt nhất và độ dài của nó.
    """
    num_nodes = len(dist_matrix)
    tour = nearest_neighbor(dist_matrix, random.randint(0, num_nodes - 1))
    tour, best_length = two_opt(tour, dist_matrix)
    best_tour = tour[:]
    neighborhoods = [lambda t, i, j: t[:i] + t[i:j + 1][::-1] + t[j + 1:],  # Đảo ngược
                     lambda t, i, j: swap(t, i, j)]  # Hoán đổi

    for iteration in range(max_iterations):
        k = 0
        while k < len(neighborhoods):
            i, j = sorted(random.sample(range(num_nodes), 2))
            if j - i < 2:
                j = (i + 2) % num_nodes
            new_tour = neighborhoods[k](tour[:], i, j)
            new_tour, new_length = two_opt(new_tour, dist_matrix)
            if new_length < best_length:
                tour, best_tour, best_length = new_tour[:], new_tour[:], new_length
                k = 0
            else:
                k += 1
        if iteration % 10 == 0:
            tour = perturb_adaptive(tour, int(num_nodes * 0.15))

    return best_tour, best_length

def swap(tour: List[int], i: int, j: int) -> List[int]:
    """Hoán đổi hai thành phố trong chu trình.

    Args:
        tour: Chu trình hiện tại.
        i: Chỉ số của thành phố thứ nhất.
        j: Chỉ số của thành phố thứ hai.

    Returns:
        Chu trình mới với hai thành phố được hoán đổi.
    """
    new_tour = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def simulated_annealing(dist_matrix: np.ndarray, initial_temp: float = 1000, cooling_rate: float = 0.995, max_iterations: int = 1000) -> Tuple[List[int], float]:
    """Giải TSP bằng Simulated Annealing với làm nguội thích nghi.

    Args:
        dist_matrix: Ma trận khoảng cách đã tính trước.
        initial_temp: Nhiệt độ ban đầu.
        cooling_rate: Tỷ lệ giảm nhiệt độ.
        max_iterations: Số lần lặp tối đa.

    Returns:
        Chu trình tốt nhất và độ dài của nó.
    """
    num_nodes = len(dist_matrix)
    tour = nearest_neighbor(dist_matrix, random.randint(0, num_nodes - 1))
    current_length = compute_tour_length(tour, dist_matrix)
    best_tour, best_length = tour[:], current_length
    temp = initial_temp
    non_improving = 0

    for _ in range(max_iterations):
        i, j = random.sample(range(num_nodes), 2)
        new_tour = tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_length = compute_tour_length(new_tour, dist_matrix)
        delta = new_length - current_length
        if delta < 0 or random.random() < math.exp(-delta / temp):
            tour, current_length = new_tour[:], new_length
            if new_length < best_length:
                best_tour, best_length = new_tour[:], new_length
                non_improving = 0
            else:
                non_improving += 1
        if non_improving > 100:
            temp = initial_temp * 0.5
            non_improving = 0
        temp *= cooling_rate

    return best_tour, best_length

# Hàm chính
def main() -> None:
    """Chạy và so sánh các thuật toán TSP trên các bộ dữ liệu TSPLIB."""
    print(f"Khởi động chương trình giải TSP tại {os.getcwd()}")
    datasets = {
        'eil51.tsp': 426,
        'berlin52.tsp': 7542,
        'st70.tsp': 675,
        'lin318.tsp': 42029  # Thêm bộ dữ liệu lin318.tsp
    }
    algorithms = {
        'Tabu Search': lambda x: tabu_search(x),
        'Iterated Local Search': lambda x: iterated_local_search(x),
        'Variable Neighborhood Search': lambda x: variable_neighborhood_search(x),
        'Simulated Annealing': lambda x: simulated_annealing(x)
    }
    results = {}

    for dataset, optimal_length in datasets.items():
        print(f"\nĐang xử lý '{dataset}'")
        nodes, coords, dist_matrix = load_tsp_data(dataset)
        if dist_matrix is None:
            continue
        results[dataset] = {}
        for name, algo in algorithms.items():
            start_time = time.perf_counter()
            tour, length = algo(dist_matrix)
            elapsed = time.perf_counter() - start_time
            results[dataset][name] = {'tour': tour, 'length': length, 'time': elapsed}
            error = (length - optimal_length) / optimal_length * 100
            print(f"{name}: Độ dài = {length:.2f}, Sai lệch = {error:.2f}%, Thời gian = {elapsed:.2f}s")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, (name, res) in enumerate(results[dataset].items()):
            plot_tour(res['tour'], coords, f"{name}\nĐộ dài: {res['length']:.2f}", axes.flat[idx], dataset)
        plt.tight_layout()
        plt.savefig(f"{dataset}_tours.png")
        plt.close()
        print(f"Biểu đồ chu trình đã được lưu vào '{dataset}_tours.png'")

    save_results_to_csv(results, datasets)
    plot_comparison(results, datasets, 'length', 'error_comparison.png')
    plot_comparison(results, datasets, 'time', 'time_comparison.png')

if __name__ == "__main__":
    main()