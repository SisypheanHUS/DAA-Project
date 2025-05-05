# Tổng quan Dự án: Bài toán Người Giao Hàng (Thuật toán Tìm kiếm Cục bộ)

## Thông tin Nhóm
- **Nhóm**: Gr06
- **Thành viên**:
  - Trịnh Đức Huy (22000097)
  - Đinh Thái Tuấn (22000130)
  - Nguyễn Đức Anh Quân (22000121)
  - Nguyễn Hữu Nhật Minh (22000108)
- **Chủ đề**: Giải bài toán Người Giao Hàng (Traveling Salesman Problem - TSP) bằng các thuật toán tìm kiếm cục bộ.

## Giới thiệu
Dự án tập trung vào việc nghiên cứu và áp dụng các thuật toán tìm kiếm cục bộ để giải bài toán Người Giao Hàng (TSP), một bài toán tối ưu hóa tổ hợp kinh điển. Mục tiêu là tìm đường đi ngắn nhất đi qua một tập hợp các thành phố và quay lại điểm xuất phát, tối ưu hóa tổng khoảng cách di chuyển.

## Mô tả Bài toán
Bài toán TSP yêu cầu tìm một vòng đi qua tất cả các thành phố trong một danh sách, mỗi thành phố chỉ được thăm đúng một lần, sau đó quay lại thành phố ban đầu sao cho tổng khoảng cách di chuyển là nhỏ nhất. Đây là bài toán NP-khó, do đó các phương pháp heuristic và metaheuristic như tìm kiếm cục bộ thường được sử dụng để tìm lời giải gần tối ưu.

## Mô hình hóa Bài toán
Bài toán TSP được mô hình hóa dưới dạng một đồ thị đầy đủ không hướng \( G = (V, E) \), trong đó:
- \( V \): Tập hợp các đỉnh (thành phố).
- \( E \): Tập hợp các cạnh (đường nối giữa các thành phố), mỗi cạnh \( (i, j) \) có trọng số \( w_{ij} \) biểu thị khoảng cách hoặc chi phí.
- Lời giải là một chu trình Hamilton (đường đi qua mỗi thành phố đúng một lần và quay lại điểm xuất phát).
- Mục tiêu: Tối thiểu hóa tổng trọng số của chu trình:  
min ∑ w_ij với (i, j) ∈ chu trình



## Phương pháp Áp dụng
Nhóm dự kiến triển khai và so sánh các thuật toán tìm kiếm cục bộ sau để giải bài toán TSP:
1. **Guided Local Search (GLS)**: Sử dụng các khoản phạt để thoát khỏi cực trị cục bộ bằng cách điều chỉnh hàm mục tiêu.
2. **Tabu Search (TS)**: Duy trì danh sách tabu để tránh quay lại các lời giải đã khám phá trước đó.
3. **Iterated Local Search (ILS)**: Kết hợp tìm kiếm cục bộ với các phép nhiễu để khám phá không gian lời giải mới.
4. **Variable Neighborhood Search (VNS)**: Thay đổi hệ thống các cấu trúc lân cận để đa dạng hóa tìm kiếm.
5. **Simulated Annealing (SA)**: Mô phỏng quá trình ủ kim loại, chấp nhận các lời giải tệ hơn với xác suất giảm dần theo thời gian.

Mỗi thuật toán sẽ bắt đầu với một lời giải ban đầu (ví dụ: một chu trình ngẫu nhiên hoặc lời giải tham lam) và cải thiện dần thông qua việc khám phá các lân cận.

## Thực nghiệm (Dự kiến)
- **Tập dữ liệu**: Sử dụng các bộ dữ liệu chuẩn của TSP (ví dụ: TSPLIB) với số lượng thành phố khác nhau.
- **Ngôn ngữ lập trình**: Các thuật toán sẽ được cài đặt bằng Python, kèm theo phần trực quan hóa để minh họa quá trình cải thiện lời giải.
- **Đánh giá**: Các thuật toán sẽ được so sánh dựa trên:
  - Chất lượng lời giải (tổng độ dài đường đi).
  - Thời gian tính toán.
  - Tốc độ hội tụ.

## Mục tiêu Dự án
Dự án hướng tới:
- Tìm hiểu sâu về các thuật toán tìm kiếm cục bộ và cách áp dụng chúng vào bài toán thực tế.
- Phát triển kỹ năng lập trình và khả năng nghiên cứu độc lập.
- So sánh hiệu quả của các phương pháp khác nhau trong việc giải bài toán TSP.

## Sản phẩm Dự kiến
- **Báo cáo**: Tài liệu (.pdf) trình bày lý thuyết, mô hình bài toán, các thuật toán và kết quả thực nghiệm.
- **Mã nguồn**: Các chương trình Python triển khai thuật toán và hỗ trợ thực nghiệm.
- **Slide trình bày**: Bản trình bày (.pdf) tóm tắt dự án để báo cáo trước lớp.

## Ngôn ngữ
Báo cáo, mã nguồn và slide trình bày sẽ sử dụng **tiếng Việt**, với một số thuật ngữ kỹ thuật giữ nguyên bằng tiếng Anh khi cần thiết.
