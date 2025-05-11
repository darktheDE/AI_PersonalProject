import tkinter as tk
from tkinter import ttk, messagebox
import time
import heapq
import copy
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class PuzzleState:
    """
    Đại diện cho một trạng thái của bài toán 8-puzzle.

    Lớp này lưu trữ cấu hình bảng, trạng thái cha, bước di chuyển dẫn đến trạng thái này,
    độ sâu trong cây tìm kiếm, và chi phí cho các thuật toán tìm đường đi.

    Thuộc tính:
        board (list): Ma trận 3x3 biểu diễn cấu hình puzzle
        parent (PuzzleState): Tham chiếu đến trạng thái cha
        move (str): Bước di chuyển dẫn đến trạng thái này ("UP", "DOWN", "LEFT", "RIGHT")
        depth (int): Độ sâu của trạng thái này trong cây tìm kiếm
        cost (int): Chi phí để đạt đến trạng thái này (sử dụng trong các thuật toán dựa trên chi phí)
        key (str): Biểu diễn dạng chuỗi của bảng để so sánh hiệu quả
        blank_row (int): Chỉ số hàng của ô trống (0)
        blank_col (int): Chỉ số cột của ô trống (0)
    """
    def __init__(self, board, parent=None, move="", depth=0, cost=0):
        """
        Khởi tạo một trạng thái puzzle mới.

        Tham số:
            board (list): Ma trận 3x3 biểu diễn cấu hình puzzle
            parent (PuzzleState, tùy chọn): Tham chiếu đến trạng thái cha. Mặc định là None.
            move (str, tùy chọn): Bước di chuyển dẫn đến trạng thái này. Mặc định là "".
            depth (int, tùy chọn): Độ sâu trong cây tìm kiếm. Mặc định là 0.
            cost (int, tùy chọn): Chi phí để đạt đến trạng thái này. Mặc định là 0.
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.key = str(board)  # For dictionary lookup
        
        # Find position of empty space (0)
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    self.blank_row = i
                    self.blank_col = j
    
    def __lt__(self, other):
        """
        So sánh nhỏ hơn để sắp xếp trong hàng đợi ưu tiên.

        Tham số:
            other (PuzzleState): Trạng thái để so sánh

        Trả về:
            bool: True nếu chi phí của trạng thái này nhỏ hơn chi phí của trạng thái khác
        """
        return self.cost < other.cost
    
    def __eq__(self, other):
        """
        So sánh bằng nhau giữa các trạng thái dựa trên cấu hình bảng.

        Tham số:
            other (PuzzleState): Trạng thái để so sánh

        Trả về:
            bool: True nếu cả hai trạng thái có cấu hình bảng giống nhau
        """
        return self.key == other.key
    
    def __hash__(self):
        """
        Hàm băm để sử dụng trạng thái trong tập hợp và từ điển.

        Trả về:
            int: Giá trị băm của khóa trạng thái
        """
        return hash(self.key)
    
    def get_children(self):
        """
        Tạo ra tất cả các trạng thái con có thể bằng cách di chuyển ô trống.

        Trả về:
            list: Danh sách các đối tượng PuzzleState con hợp lệ
        """
        children = []
        # Possible moves: up, down, left, right
        moves = [
            (-1, 0, "UP"),
            (1, 0, "DOWN"),
            (0, -1, "LEFT"),
            (0, 1, "RIGHT")
        ]
        
        for move in moves:
            new_row = self.blank_row + move[0]
            new_col = self.blank_col + move[1]
            move_name = move[2]
            
            # Check if move is valid
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # Create new board with the move applied
                new_board = [row[:] for row in self.board]
                new_board[self.blank_row][self.blank_col] = new_board[new_row][new_col]
                new_board[new_row][new_col] = 0
                
                # Create new state
                new_state = PuzzleState(
                    new_board,
                    parent=self,
                    move=move_name,
                    depth=self.depth + 1,
                    cost=self.depth + 1  # Cost is just the depth for UCS
                )
                children.append(new_state)
        
        return children

class PuzzleSolver:
    """
        Triển khai các thuật toán tìm kiếm khác nhau để giải bài toán 8-puzzle.

        Lớp này chứa các cài đặt của thuật toán tìm kiếm bao gồm BFS, DFS, UCS,
        Greedy Search, A*, IDA*, và Iterative Deepening. Nó cũng cung cấp các hàm
        tiện ích để tính toán heuristic và xác định trạng thái đích.

        Thuộc tính:
            goal_state (list): Ma trận 3x3 biểu diễn cấu hình đích [[1,2,3],[4,5,6],[7,8,0]]
        """
    def __init__(self):
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
    def get_manhattan_distance(self, state):
        """
        Tính toán heuristic khoảng cách Manhattan cho một trạng thái.

        Khoảng cách Manhattan tính tổng khoảng cách theo chiều ngang và dọc
        của mỗi ô so với vị trí đích của nó.

        Tham số:
            state (PuzzleState): Trạng thái puzzle cần đánh giá

        Trả về:
            int: Tổng khoảng cách Manhattan
        """
        distance = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 0:
                    # Find where this number should be in the goal state
                    value = state.board[i][j]
                    goal_row, goal_col = divmod(value - 1, 3)
                    if value == 0:
                        goal_row, goal_col = 2, 2
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance
    
    def get_misplaced_tiles(self, state):
        """
        Tính số ô không đúng vị trí trong một trạng thái.

        Đếm số ô không nằm đúng vị trí đích của chúng (không tính ô trống).

        Tham số:
            state (PuzzleState): Trạng thái puzzle cần đánh giá

        Trả về:
            int: Số ô không đúng vị trí
        """
        count = 0
        for i in range(3):
            for j in range(3):
                if state.board[i][j] != 0 and state.board[i][j] != self.goal_state[i][j]:
                    count += 1
        return count
    
    def is_goal(self, state):
        """
        Kiểm tra xem một trạng thái có khớp với cấu hình đích hay không.

        Tham số:
            state (PuzzleState): Trạng thái cần kiểm tra

        Trả về:
            bool: True nếu trạng thái khớp với cấu hình đích
        """
        return state.board == self.goal_state
    
    def get_path(self, state):
        """
        Tái tạo đường đi từ trạng thái ban đầu đến trạng thái cho trước.

        Tham số:
            state (PuzzleState): Trạng thái cuối để truy ngược lại

        Trả về:
            list: Danh sách các trạng thái theo thứ tự từ trạng thái ban đầu đến trạng thái cho trước
        """
        path = []
        while state.parent:
            path.append(state)
            state = state.parent
        path.append(state)  # Initial state
        path.reverse()
        return path
    def simple_hill_climbing(self, initial_state, callback=None, heuristic='manhattan', max_iterations=1000):
        """
        Thuật toán Simple Hill Climbing - Mỗi lần chỉ chọn trạng thái tốt hơn đầu tiên tìm được.
        
        Simple Hill Climbing di chuyển tới trạng thái láng giềng đầu tiên có giá trị heuristic tốt hơn
        trạng thái hiện tại. Thuật toán dừng khi không còn trạng thái tốt hơn. Do đặc tính "tham lam"
        và "cận thị", thuật toán dễ bị mắc kẹt ở cực đại địa phương.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            max_iterations (int, tùy chọn): Số lần lặp tối đa. Mặc định là 1000.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, giá trị heuristic cuối cùng và thời gian thực hiện
        """
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Trạng thái hiện tại, bắt đầu từ trạng thái ban đầu
        current_state = initial_state
        current_h = h_func(current_state)
        
        # Theo dõi thống kê
        nodes_expanded = 0
        path = [current_state]  # Lưu đường đi
        iterations = 0
        
        # Lặp cho đến khi tìm thấy giải pháp hoặc không thể cải thiện hơn
        while iterations < max_iterations:
            if self.is_goal(current_state):
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": 1,  # Hill Climbing chỉ giữ trạng thái hiện tại
                    "time": end_time - start_time,
                    "final_h": current_h
                }
            
            # Lấy tất cả trạng thái láng giềng
            children = current_state.get_children()
            nodes_expanded += 1
            iterations += 1
            
            if callback:
                callback(current_state, nodes_expanded, len(children), time.time() - start_time)
            
            # Tìm láng giềng đầu tiên tốt hơn trạng thái hiện tại
            improved = False
            for child in children:
                child_h = h_func(child)
                
                # Nếu tìm thấy trạng thái tốt hơn, di chuyển tới đó
                if child_h < current_h:
                    current_state = child
                    current_h = child_h
                    path.append(current_state)
                    improved = True
                    break
            
            # Nếu không tìm thấy trạng thái tốt hơn, dừng lại
            if not improved:
                break
        
        # Kiểm tra xem trạng thái cuối cùng có phải là mục tiêu không
        if self.is_goal(current_state):
            end_time = time.time()
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "final_h": current_h
            }
        else:
            # Kết thúc do không tìm thấy giải pháp
            end_time = time.time()
            return {
                "path": None,  # Không tìm thấy đường đi đến đích
                "partial_path": path,  # Đường đi một phần (đến cực đại địa phương)
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "final_h": current_h
            }
    
    def steepest_ascent_hill_climbing(self, initial_state, callback=None, heuristic='manhattan', max_iterations=1000):
        """
        Thuật toán Steepest-Ascent Hill Climbing - Chọn trạng thái tốt nhất trong tất cả các láng giềng.
        
        Khác với Simple Hill Climbing, thuật toán này xem xét tất cả các trạng thái láng giềng
        và chọn trạng thái có giá trị heuristic tốt nhất. Vẫn dễ bị mắc kẹt ở cực đại địa phương,
        nhưng thường tìm ra giải pháp tốt hơn so với Simple Hill Climbing.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            max_iterations (int, tùy chọn): Số lần lặp tối đa. Mặc định là 1000.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, giá trị heuristic cuối cùng và thời gian thực hiện
        """
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Trạng thái hiện tại, bắt đầu từ trạng thái ban đầu
        current_state = initial_state
        current_h = h_func(current_state)
        
        # Theo dõi thống kê
        nodes_expanded = 0
        path = [current_state]  # Lưu đường đi
        iterations = 0
        
        # Lặp cho đến khi tìm thấy giải pháp hoặc không thể cải thiện hơn
        while iterations < max_iterations:
            if self.is_goal(current_state):
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": 1,
                    "time": end_time - start_time,
                    "final_h": current_h
                }
            
            # Lấy tất cả trạng thái láng giềng
            children = current_state.get_children()
            nodes_expanded += 1
            iterations += 1
            
            if callback:
                callback(current_state, nodes_expanded, len(children), time.time() - start_time)
            
            # Tìm láng giềng tốt nhất
            best_child = None
            best_h = current_h
            
            for child in children:
                child_h = h_func(child)
                
                # Cập nhật nếu tìm thấy trạng thái tốt hơn
                if child_h < best_h:
                    best_child = child
                    best_h = child_h
            
            # Nếu tìm thấy trạng thái tốt hơn, di chuyển tới đó
            if best_child and best_h < current_h:
                current_state = best_child
                current_h = best_h
                path.append(current_state)
            else:
                # Không tìm thấy trạng thái tốt hơn, dừng lại
                break
        
        # Kiểm tra xem trạng thái cuối cùng có phải là mục tiêu không
        if self.is_goal(current_state):
            end_time = time.time()
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "final_h": current_h
            }
        else:
            # Kết thúc do không tìm thấy giải pháp
            end_time = time.time()
            return {
                "path": None,  # Không tìm thấy đường đi đến đích
                "partial_path": path,  # Đường đi một phần (đến cực đại địa phương)
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "final_h": current_h
            }
    
    def bfs(self, initial_state, callback=None):
        """
        Thực hiện tìm kiếm theo chiều rộng (BFS) từ trạng thái ban đầu.

        BFS mở rộng các nút theo thứ tự độ sâu của chúng trong cây tìm kiếm,
        đảm bảo đường đi ngắn nhất đến đích.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa và thời gian thực hiện
        """
        """Breadth-First Search"""
        start_time = time.time()
        queue = deque([initial_state])
        visited = set([initial_state.key])
        nodes_expanded = 0
        max_queue_size = 1
        
        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current = queue.popleft()
            
            if callback:
                callback(current, nodes_expanded, len(queue), time.time() - start_time)
            
            if self.is_goal(current):
                path = self.get_path(current)
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "time": end_time - start_time
                }
            
            nodes_expanded += 1
            
            for child in current.get_children():
                if child.key not in visited:
                    queue.append(child)
                    visited.add(child.key)
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "time": end_time - start_time
        }
    
    def dfs(self, initial_state, callback=None, max_depth=None):
        """
        Thực hiện tìm kiếm theo chiều sâu (DFS) từ trạng thái ban đầu.

        DFS mở rộng nút sâu nhất chưa mở rộng trước. Có thể cung cấp giới hạn
        độ sâu tùy chọn để giới hạn tìm kiếm.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            max_depth (int, tùy chọn): Độ sâu tìm kiếm tối đa. Mặc định là None.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa, thời gian thực hiện và cờ cắt
        """
        start_time = time.time()
        stack = [initial_state]
        visited = set([initial_state.key])
        nodes_expanded = 0
        max_stack_size = 1
        
        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            current = stack.pop()
            
            if callback:
                callback(current, nodes_expanded, len(stack), time.time() - start_time)
            
            if self.is_goal(current):
                path = self.get_path(current)
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_stack_size,
                    "time": end_time - start_time,
                    "cutoff_occurred": False
                }
            
            if max_depth is not None and current.depth >= max_depth:
                continue
                
            nodes_expanded += 1
            
            # Get children in reverse to maintain correct DFS order
            for child in reversed(current.get_children()):
                if child.key not in visited:
                    stack.append(child)
                    visited.add(child.key)
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_stack_size,
            "time": end_time - start_time,
            "cutoff_occurred": max_depth is not None
        }
    
    def ucs(self, initial_state, callback=None):
        """
        Thực hiện tìm kiếm chi phí đồng nhất (UCS) từ trạng thái ban đầu.

        UCS mở rộng các nút theo thứ tự chi phí đường đi, đảm bảo
        đường đi có chi phí tối thiểu đến đích.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa và thời gian thực hiện
        """
        start_time = time.time()
        priority_queue = [(initial_state.cost, 0, initial_state)]  # (cost, tiebreaker, state)
        counter = 1  # Tiebreaker for same cost states
        visited = set([initial_state.key])
        nodes_expanded = 0
        max_queue_size = 1
        
        while priority_queue:
            max_queue_size = max(max_queue_size, len(priority_queue))
            _, _, current = heapq.heappop(priority_queue)
            
            if callback:
                callback(current, nodes_expanded, len(priority_queue), time.time() - start_time)
            
            if self.is_goal(current):
                path = self.get_path(current)
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "time": end_time - start_time
                }
            
            nodes_expanded += 1
            
            for child in current.get_children():
                if child.key not in visited:
                    heapq.heappush(priority_queue, (child.cost, counter, child))
                    counter += 1
                    visited.add(child.key)
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "time": end_time - start_time
        }
    
    def greedy_search(self, initial_state, callback=None, heuristic='manhattan'):
        """
        Thực hiện tìm kiếm tham lam (Greedy Best-First Search) từ trạng thái ban đầu.

        Tìm kiếm tham lam mở rộng các nút chỉ dựa trên ước lượng heuristic về
        khoảng cách đến đích, không xem xét chi phí đường đi.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa và thời gian thực hiện
        """
        start_time = time.time()
        
        # Choose heuristic function
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Initialize with heuristic value as priority
        initial_h = h_func(initial_state)
        priority_queue = [(initial_h, 0, initial_state)]  # (heuristic, tiebreaker, state)
        counter = 1  # Tiebreaker for same heuristic values
        visited = set([initial_state.key])
        nodes_expanded = 0
        max_queue_size = 1
        
        while priority_queue:
            max_queue_size = max(max_queue_size, len(priority_queue))
            _, _, current = heapq.heappop(priority_queue)
            
            if callback:
                callback(current, nodes_expanded, len(priority_queue), time.time() - start_time)
            
            if self.is_goal(current):
                path = self.get_path(current)
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "time": end_time - start_time
                }
            
            nodes_expanded += 1
            
            for child in current.get_children():
                if child.key not in visited:
                    h_value = h_func(child)
                    heapq.heappush(priority_queue, (h_value, counter, child))
                    counter += 1
                    visited.add(child.key)
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "time": end_time - start_time
        }
    
    def a_star_search(self, initial_state, callback=None, heuristic='manhattan'):
        """
        Thực hiện tìm kiếm A* từ trạng thái ban đầu.

        A* kết hợp chi phí đường đi (g) và ước lượng heuristic (h) để hướng dẫn tìm kiếm,
        mở rộng các nút theo thứ tự f = g + h. Nó đảm bảo đường đi tối ưu
        khi sử dụng heuristic hợp lệ.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa và thời gian thực hiện
        """
        start_time = time.time()
        
        # Choose heuristic function
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Calculate initial heuristic value
        initial_h = h_func(initial_state)
        
        # Priority queue with (f, tiebreaker, state) where f = g + h
        # g = path cost (depth), h = heuristic estimate
        priority_queue = [(initial_state.depth + initial_h, 0, initial_state)]
        counter = 1  # Tiebreaker for same f-value states
        
        # Track visited states and their costs
        visited = {initial_state.key: initial_state.depth}
        nodes_expanded = 0
        max_queue_size = 1
        
        while priority_queue:
            max_queue_size = max(max_queue_size, len(priority_queue))
            _, _, current = heapq.heappop(priority_queue)
            
            if callback:
                callback(current, nodes_expanded, len(priority_queue), time.time() - start_time)
            
            if self.is_goal(current):
                path = self.get_path(current)
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "time": end_time - start_time
                }
            
            nodes_expanded += 1
            
            for child in current.get_children():
                # A* can re-expand nodes if a better path is found
                if child.key not in visited or child.depth < visited[child.key]:
                    visited[child.key] = child.depth
                    h_value = h_func(child)
                    f_value = child.depth + h_value  # f = g + h
                    heapq.heappush(priority_queue, (f_value, counter, child))
                    counter += 1
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "time": end_time - start_time
        }
    def ida_star_search(self, initial_state, callback=None, heuristic='manhattan'):
        """
        Thực hiện tìm kiếm IDA* (Iterative Deepening A*) từ trạng thái ban đầu.

        IDA* thực hiện một loạt các tìm kiếm theo chiều sâu với giới hạn giá trị f tăng dần,
        kết hợp hiệu quả bộ nhớ của iterative deepening với bản chất thông tin của A*.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa, thời gian thực hiện và giới hạn cuối
        """
        start_time = time.time()
        
        # Choose heuristic function
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Calculate initial heuristic value
        initial_h = h_func(initial_state)
        
        # Set initial bound to the initial heuristic value
        bound = initial_h
        
        # Statistics
        nodes_expanded_total = 0
        max_stack_size_total = 0
        
        # Path to the current node
        path = []
        
        def search(node, g, bound, nodes_expanded, path_so_far, max_stack_size):
            """
            Hàm tìm kiếm DFS đệ quy cho IDA* với giới hạn giá trị f.

            Tham số:
                node (PuzzleState): Nút hiện tại đang mở rộng
                g (int): Chi phí đường đi đến nút hiện tại
                bound (int): Giới hạn giá trị f hiện tại
                nodes_expanded (int): Bộ đếm cho số nút mở rộng
                path_so_far (list): Đường đi từ gốc đến nút hiện tại
                max_stack_size (int): Kích thước ngăn xếp tối đa đạt được

            Trả về:
                tuple: (new_bound, solution_path, nodes_expanded, max_stack_size)
                    new_bound: Giá trị f tối thiểu vượt quá giới hạn hiện tại, hoặc -1 nếu tìm thấy đích
                    solution_path: Đường đi hoàn chỉnh đến đích nếu tìm thấy, nếu không là None
                    nodes_expanded: Số nút mở rộng cập nhật
                    max_stack_size: Kích thước ngăn xếp tối đa cập nhật
            """
            path_so_far.append(node)
            
            # Calculate f-value (f = g + h)
            h = h_func(node)
            f = g + h
            
            if callback:
                callback(node, nodes_expanded, len(path_so_far), time.time() - start_time)
            
            # If f exceeds the bound, return f as the new minimum bound
            if f > bound:
                path_so_far.pop()
                return f, None, nodes_expanded, max_stack_size
            
            # If goal is reached, return the path
            if self.is_goal(node):
                return -1, path_so_far[:], nodes_expanded, max_stack_size
            
            # Track min value for next bound
            min_bound = float('inf')
            nodes_expanded += 1
            
            # Get all children
            children = node.get_children()
            max_stack_size = max(max_stack_size, len(children) + len(path_so_far))
            
            for child in children:
                # Don't revisit nodes in the current path
                if any(state.key == child.key for state in path_so_far):
                    continue
                
                # Recursive call
                new_bound, solution, nodes_expanded, max_stack_size = search(
                    child, g + 1, bound, nodes_expanded, path_so_far, max_stack_size
                )
                
                # If solution found, bubble it up
                if new_bound == -1:
                    return -1, solution, nodes_expanded, max_stack_size
                
                # Otherwise, update minimum bound for next iteration
                if new_bound < min_bound:
                    min_bound = new_bound
            
            # Remove current node from path before returning
            path_so_far.pop()
            
            return min_bound, None, nodes_expanded, max_stack_size
        
        # Iteratively deepen the bound
        while True:
            # For practical purposes, limit the maximum bound
            if bound > 80:  # A reasonable upper limit for 8-puzzle
                end_time = time.time()
                return {
                    "path": None,
                    "nodes_expanded": nodes_expanded_total,
                    "max_queue_size": max_stack_size_total,
                    "time": end_time - start_time,
                    "final_bound": bound
                }
            
            path = []
            nodes_expanded = 0
            max_stack_size = 1
            
            # Perform depth-first search with current bound
            new_bound, solution, nodes_expanded, max_stack_size = search(
                initial_state, 0, bound, nodes_expanded, path, max_stack_size
            )
            
            nodes_expanded_total += nodes_expanded
            max_stack_size_total = max(max_stack_size_total, max_stack_size)
            
            # If solution found, return it
            if new_bound == -1:
                end_time = time.time()
                return {
                    "path": solution,
                    "nodes_expanded": nodes_expanded_total,
                    "max_queue_size": max_stack_size_total,
                    "time": end_time - start_time,
                    "final_bound": bound
                }
            
            # No solution within current bound, increase the bound
            if new_bound == float('inf'):
                # No solution exists
                end_time = time.time()
                return {
                    "path": None,
                    "nodes_expanded": nodes_expanded_total,
                    "max_queue_size": max_stack_size_total,
                    "time": end_time - start_time,
                    "final_bound": bound
                }
            
            # Set new bound for next iteration
            bound = new_bound
    def random_restart_hill_climbing(self, initial_state, callback=None, heuristic='manhattan', max_restarts=10):
        """
        Thuật toán Hill Climbing với khởi tạo ngẫu nhiên - Giúp thoát khỏi cực đại địa phương.
        
        Thay vì chỉ chạy hill climbing từ một trạng thái xuất phát, thuật toán này
        thực hiện nhiều lần khởi động lại với các trạng thái ngẫu nhiên khi bị mắc kẹt.
        Chiến lược này giúp tăng khả năng tìm ra giải pháp tốt khi hill climbing đơn thuần
        thường bị mắc kẹt ở cực đại địa phương.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            max_restarts (int, tùy chọn): Số lần khởi động lại tối đa. Mặc định là 10.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, số lần khởi động lại và thời gian thực hiện
        """
        import random
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        best_solution = None
        best_h = float('inf')
        total_nodes_expanded = 0
        
        # Lưu trữ trạng thái ban đầu để có thể tạo trạng thái ngẫu nhiên sau này
        initial_board = initial_state.board
        
        # Thực hiện hill climbing với nhiều lần khởi động lại
        for restart in range(max_restarts):
            # Sử dụng trạng thái ban đầu cho lần đầu tiên, sau đó tạo trạng thái ngẫu nhiên
            if restart == 0:
                current_state = initial_state
            else:
                # Tạo một trạng thái ngẫu nhiên bằng cách trộn bảng
                random_board = self.get_random_state(initial_board)
                current_state = PuzzleState(random_board)
            
            # Thực hiện steepest-ascent hill climbing từ trạng thái hiện tại
            result = self.steepest_ascent_hill_climbing(current_state, callback, heuristic)
            
            # Cập nhật tổng số nút đã mở rộng
            total_nodes_expanded += result["nodes_expanded"]
            
            # Nếu tìm thấy giải pháp, trả về ngay lập tức
            if result["path"]:
                end_time = time.time()
                result["time"] = end_time - start_time
                result["restarts"] = restart
                result["nodes_expanded"] = total_nodes_expanded
                return result
            
            # Nếu không tìm thấy giải pháp, kiểm tra xem có tốt hơn kết quả tốt nhất hiện tại không
            if "partial_path" in result and result["final_h"] < best_h:
                best_solution = result
                best_h = result["final_h"]
        
        # Nếu không tìm thấy giải pháp sau tất cả các lần khởi động lại, trả về kết quả tốt nhất
        end_time = time.time()
        if best_solution:
            best_solution["time"] = end_time - start_time
            best_solution["restarts"] = max_restarts
            best_solution["nodes_expanded"] = total_nodes_expanded
            return best_solution
        else:
            return {
                "path": None,
                "partial_path": None,
                "nodes_expanded": total_nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "restarts": max_restarts
            }
        
    def get_random_state(self, initial_board):
        """
        Tạo một trạng thái ngẫu nhiên cho 8-puzzle bằng cách trộn ngẫu nhiên.
        
        Tham số:
            initial_board (list): Cấu hình ban đầu của bảng
            
        Trả về:
            list: Một cấu hình bảng ngẫu nhiên
        """
        import random
        
        # Tạo bản sao của bảng ban đầu
        flat_board = [item for row in initial_board for item in row]
        
        # Trộn ngẫu nhiên các ô (đảm bảo có thể giải được)
        # Lưu ý: Không phải mọi cấu hình ngẫu nhiên đều có thể giải được
        # Để đơn giản, chúng ta sẽ thực hiện n bước di chuyển hợp lệ ngẫu nhiên
        
        # Tìm vị trí ô trống
        blank_idx = flat_board.index(0)
        blank_row, blank_col = blank_idx // 3, blank_idx % 3
        
        # Thực hiện 50 bước di chuyển ngẫu nhiên
        board = [row[:] for row in initial_board]  # Deep copy
        
        for _ in range(50):
            # Các di chuyển có thể: lên, xuống, trái, phải
            possible_moves = []
            
            # Kiểm tra các di chuyển hợp lệ
            if blank_row > 0:  # Có thể di chuyển lên
                possible_moves.append((-1, 0))
            if blank_row < 2:  # Có thể di chuyển xuống
                possible_moves.append((1, 0))
            if blank_col > 0:  # Có thể di chuyển trái
                possible_moves.append((0, -1))
            if blank_col < 2:  # Có thể di chuyển phải
                possible_moves.append((0, 1))
            
            # Chọn một di chuyển ngẫu nhiên
            dr, dc = random.choice(possible_moves)
            new_row, new_col = blank_row + dr, blank_col + dc
            
            # Thực hiện di chuyển
            board[blank_row][blank_col] = board[new_row][new_col]
            board[new_row][new_col] = 0
            
            # Cập nhật vị trí ô trống
            blank_row, blank_col = new_row, new_col
        
        return board
    
    def iterative_deepening(self, initial_state, callback=None, max_depth=50):
        """
        Thực hiện tìm kiếm tăng độ sâu (Iterative Deepening Search) từ trạng thái ban đầu.

        Tìm kiếm tăng độ sâu thực hiện một loạt các tìm kiếm DFS có giới hạn độ sâu với
        giới hạn độ sâu tăng dần.

        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            max_depth (int, tùy chọn): Độ sâu tìm kiếm tối đa. Mặc định là 50.

        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa, thời gian thực hiện và độ sâu cuối
        """
        start_time = time.time()
        nodes_expanded_total = 0
        max_queue_size_total = 0
        
        for depth in range(max_depth + 1):
            result = self.dfs(initial_state, callback, depth)
            nodes_expanded_total += result["nodes_expanded"]
            max_queue_size_total = max(max_queue_size_total, result["max_queue_size"])
            
            if not result["cutoff_occurred"] or result["path"]:
                end_time = time.time()
                return {
                    "path": result["path"],
                    "nodes_expanded": nodes_expanded_total,
                    "max_queue_size": max_queue_size_total,
                    "time": end_time - start_time,
                    "final_depth": depth
                }
        
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded_total,
            "max_queue_size": max_queue_size_total,
            "time": end_time - start_time,
            "final_depth": max_depth
        }
    def simulated_annealing(self, initial_state, callback=None, heuristic='manhattan', max_iterations=1000, initial_temp=10.0, cooling_rate=0.95):
        """
        Thuật toán Simulated Annealing - Tìm kiếm dựa trên quá trình vật lý của luyện kim.
        
        Simulated Annealing tránh cực đại địa phương bằng cách đôi khi chấp nhận các trạng thái tồi hơn
        với xác suất giảm dần theo thời gian. Thuật toán bắt đầu với nhiệt độ cao (nhiều di chuyển tồi được chấp nhận)
        và dần dần "làm mát" hệ thống để tinh chỉnh giải pháp.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            max_iterations (int, tùy chọn): Số lần lặp tối đa. Mặc định là 1000.
            initial_temp (float, tùy chọn): Nhiệt độ ban đầu. Mặc định là 10.0.
            cooling_rate (float, tùy chọn): Tốc độ làm mát (0 < cooling_rate < 1). Mặc định là 0.95.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, nhiệt độ cuối cùng và thời gian thực hiện
        """
        import random
        import math
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Trạng thái hiện tại, bắt đầu từ trạng thái ban đầu
        current_state = initial_state
        current_h = h_func(current_state)
        
        # Theo dõi thống kê
        nodes_expanded = 0
        path = [current_state]  # Lưu đường đi
        best_state = current_state
        best_h = current_h
        
        # Nhiệt độ ban đầu
        temperature = initial_temp
        
        # Lặp cho đến khi tìm thấy giải pháp hoặc đạt đến số lần lặp tối đa
        for iteration in range(max_iterations):
            # Nếu tìm thấy giải pháp, trả về kết quả
            if self.is_goal(current_state):
                end_time = time.time()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": 1,  # Simulated Annealing chỉ giữ trạng thái hiện tại
                    "time": end_time - start_time,
                    "final_h": current_h,
                    "final_temp": temperature
                }
            
            # Lấy tất cả trạng thái láng giềng
            children = current_state.get_children()
            nodes_expanded += 1
            
            if callback:
                callback(current_state, nodes_expanded, 1, time.time() - start_time)
            
            if not children:
                break
            
            # Chọn một láng giềng ngẫu nhiên
            next_state = random.choice(children)
            next_h = h_func(next_state)
            
            # Tính toán delta E (sự thay đổi "năng lượng")
            delta_h = next_h - current_h
            
            # Nếu láng giềng tốt hơn hoặc chấp nhận xác suất
            if delta_h <= 0 or random.random() < math.exp(-delta_h / temperature):
                current_state = next_state
                current_h = next_h
                path.append(current_state)
                
                # Cập nhật trạng thái tốt nhất
                if current_h < best_h:
                    best_state = current_state
                    best_h = current_h
            
            # Làm mát hệ thống
            temperature *= cooling_rate
            
            # Nếu nhiệt độ quá thấp, dừng lại
            if temperature < 0.01:
                break
        
        # Kiểm tra xem trạng thái tốt nhất có phải là mục tiêu không
        if self.is_goal(best_state):
            # Tái tạo đường đi từ trạng thái ban đầu đến trạng thái cuối cùng
            # Lưu ý: Điều này chỉ trả về đường đi đến best_state, không phải đường đi tối ưu
            end_time = time.time()
            return {
                "path": self.get_path(best_state),
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "final_h": best_h,
                "final_temp": temperature
            }
        
        # Không tìm thấy giải pháp
        end_time = time.time()
        return {
            "path": None,  # Không tìm thấy đường đi đến đích
            "partial_path": path,  # Đường đi một phần
            "nodes_expanded": nodes_expanded,
            "max_queue_size": 1,
            "time": end_time - start_time,
            "final_h": best_h,
            "final_temp": temperature
        }
    
    def beam_search(self, initial_state, callback=None, heuristic='manhattan', beam_width=3):
        """
        Thực hiện tìm kiếm chùm tia (Beam Search) từ trạng thái ban đầu.
        
        Beam Search là sự kết hợp giữa BFS và Best-First search, chỉ giữ lại beam_width
        trạng thái tốt nhất ở mỗi độ sâu. Điều này giúp giảm bộ nhớ so với BFS đồng thời
        vẫn duy trì việc khám phá nhiều đường đi tiềm năng hơn so với Greedy Best-First.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi trạng thái mở rộng. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            beam_width (int, tùy chọn): Số lượng trạng thái tốt nhất giữ lại ở mỗi độ sâu. Mặc định là 3.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, kích thước hàng đợi tối đa và thời gian thực hiện
        """
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Khởi tạo với trạng thái ban đầu
        current_level = [initial_state]
        visited = {initial_state.key: initial_state.depth}
        nodes_expanded = 0
        max_queue_size = 1
        
        while current_level:
            max_queue_size = max(max_queue_size, len(current_level))
            
            # Danh sách trạng thái con được tạo từ tất cả các trạng thái ở mức hiện tại
            next_level = []
            
            # Mở rộng tất cả các trạng thái ở mức hiện tại
            for state in current_level:
                if callback:
                    callback(state, nodes_expanded, len(current_level), time.time() - start_time)
                
                # Kiểm tra nếu đạt đến mục tiêu
                if self.is_goal(state):
                    path = self.get_path(state)
                    end_time = time.time()
                    return {
                        "path": path,
                        "nodes_expanded": nodes_expanded,
                        "max_queue_size": max_queue_size,
                        "time": end_time - start_time,
                        "beam_width": beam_width
                    }
                
                # Mở rộng các trạng thái con
                nodes_expanded += 1
                for child in state.get_children():
                    # Kiểm tra nếu trạng thái đã được thăm qua hoặc là trạng thái tốt hơn
                    if child.key not in visited or child.depth < visited[child.key]:
                        visited[child.key] = child.depth
                        child.h = h_func(child)  # Lưu giá trị heuristic để sắp xếp
                        next_level.append(child)
            
            # Sắp xếp next_level theo giá trị heuristic và giữ lại tối đa beam_width phần tử
            next_level.sort(key=lambda x: x.h)
            current_level = next_level[:beam_width]
        
        # Không tìm thấy giải pháp
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "time": end_time - start_time,
            "beam_width": beam_width
        }
    
    def csp_callback(self, state, nodes_expanded, queue_size, elapsed_time):
        """Special callback for CSP algorithms to show intermediate steps."""
        self.update_puzzle_display(state.board)
        self.info_label.config(text=f"Building solution using CSP...\n"
                            f"Nodes expanded: {nodes_expanded}\n"
                            f"Time elapsed: {elapsed_time:.2f}s")
        self.root.update()
        time.sleep(0.1)  # Short delay to visualize the steps
    
    def backtracking_search(self, initial_state=None, callback=None):
        """
        Thuật toán Backtracking Search xây dựng cấu hình 8-puzzle từ đầu.
        
        Thay vì tìm đường đi từ cấu hình ban đầu, thuật toán này bắt đầu với bảng trống
        và lần lượt đặt các số từ 0-8 vào các ô, thỏa mãn các ràng buộc.
        
        Tham số:
            initial_state: Không sử dụng, chỉ để tương thích với giao diện
            callback: Hàm gọi lại để hiển thị tiến trình
            
        Trả về:
            dict: Kết quả chứa cấu hình xây dựng được
        """
        start_time = time.time()
        nodes_expanded = 0
        
        # Khởi tạo bảng trống
        empty_board = [[-1 for _ in range(3)] for _ in range(3)]
        
        # # Danh sách các giá trị cần gán (0-8)
        # values_to_assign = list(range(9))
        
        # Hàm check ràng buộc: mỗi số chỉ xuất hiện một lần
        def is_consistent(board, row, col, value):
            # Kiểm tra số đã tồn tại trong bảng chưa
            for i in range(3):
                for j in range(3):
                    if board[i][j] == value:
                        return False
            return value == self.goal_state[row][col]
        
            # Hàm backtracking đệ quy
        def backtrack(board, unassigned_positions):
            nonlocal nodes_expanded
            
            # Nếu không còn vị trí nào chưa gán, đã tìm thấy lời giải
            if not unassigned_positions:
                return board
            
            nodes_expanded += 1
            
            # Chọn vị trí tiếp theo để gán giá trị
            row, col = unassigned_positions[0]
            remaining_positions = unassigned_positions[1:]
            
            # Thử từng giá trị có thể - chỉ thử giá trị đúng với goal_state
            target_value = self.goal_state[row][col]
            if is_consistent(board, row, col, target_value):
                # Gán giá trị và tiếp tục đệ quy
                board[row][col] = target_value
                
                if callback:
                    callback_board = [row[:] for row in board]
                    callback_state = PuzzleState(callback_board)
                    callback(callback_state, nodes_expanded, 0, time.time() - start_time)
                
                result = backtrack(board, remaining_positions)
                if result:
                    return result
                
                # Quay lui
                board[row][col] = -1
            
            return None
        
        # Tạo danh sách các vị trí cần gán
        positions = [(i, j) for i in range(3) for j in range(3)]
        
        # Bắt đầu backtracking
        result_board = backtrack(empty_board, positions)
        
        end_time = time.time()
        
        if result_board:
            # Chuyển đổi kết quả thành PuzzleState
            final_state = PuzzleState(result_board)
            
            # Tạo đường đi giả định (chỉ trạng thái cuối)
            path = [final_state]
            
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "message": "Cấu hình được xây dựng từ đầu bằng CSP Backtracking"
            }
        else:
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "message": "Không thể xây dựng cấu hình hợp lệ"
            }

    def intelligent_backtracking(self, initial_state=None, callback=None):
        """
        Thuật toán Intelligent Backtracking sử dụng các heuristic MRV và LCV.
        
        MRV (Minimum Remaining Values): Chọn biến có ít giá trị hợp lệ nhất
        LCV (Least Constraining Value): Chọn giá trị ảnh hưởng ít nhất đến các biến khác
        
        Tham số:
            initial_state: Không sử dụng, chỉ để tương thích với giao diện
            callback: Hàm gọi lại để hiển thị tiến trình
            
        Trả về:
            dict: Kết quả chứa cấu hình xây dựng được
        """
        start_time = time.time()
        nodes_expanded = 0
        
        # Khởi tạo bảng trống
        empty_board = [[-1 for _ in range(3)] for _ in range(3)]
        
        # Giá trị ban đầu có thể gán cho mỗi ô
        domains = {}
        for i in range(3):
            for j in range(3):
                domains[(i, j)] = [self.goal_state[i][j]]
        
        # Hàm kiểm tra ràng buộc
        def is_consistent(board, row, col, value):
            # Kiểm tra số đã tồn tại trong bảng chưa
            for i in range(3):
                for j in range(3):
                    if board[i][j] == value:
                        return False
            return True
        
        # Hàm tính số giá trị còn lại cho mỗi biến (MRV)
        def count_remaining_values(board, domains, position):
            row, col = position
            count = 0
            for value in domains[position]:
                if is_consistent(board, row, col, value):
                    count += 1
            return count
        
        # Hàm chọn biến tiếp theo theo MRV
        def select_unassigned_variable(board, unassigned_positions, domains):
            # Chọn biến có ít giá trị hợp lệ nhất (MRV)
            return min(unassigned_positions, 
                    key=lambda pos: count_remaining_values(board, domains, pos))
        
        # Hàm sắp xếp các giá trị theo LCV
        def order_domain_values(board, var, domains, unassigned):
            # Đếm số biến bị ràng buộc khi gán mỗi giá trị
            def count_constraints(value):
                count = 0
                row, col = var
                temp_board = [row[:] for row in board]
                temp_board[row][col] = value
                
                for pos in unassigned:
                    if pos != var:
                        r, c = pos
                        for val in domains[pos]:
                            if not is_consistent(temp_board, r, c, val):
                                count += 1
                return count
            
            # Sắp xếp theo số lượng ràng buộc tăng dần (ít ràng buộc trước)
            return sorted(domains[var], key=count_constraints)
        
        # Hàm backtracking đệ quy với MRV và LCV
        def backtrack(board, unassigned_positions):
            nonlocal nodes_expanded
                        
            # Nếu không còn vị trí nào chưa gán, đã tìm thấy lời giải
            if not unassigned_positions:
                return board
            
            nodes_expanded += 1
            
            # Chọn vị trí tiếp theo dựa trên MRV
            var = select_unassigned_variable(board, unassigned_positions, domains)
            row, col = var
            
            # Lấy danh sách các vị trí chưa gán còn lại
            remaining_positions = [pos for pos in unassigned_positions if pos != var]
            
            # Thử các giá trị được sắp xếp theo LCV
            ordered_values = order_domain_values(board, var, domains, unassigned_positions)
            for value in ordered_values:
                if is_consistent(board, row, col, value):
                    # Gán giá trị và tiếp tục đệ quy
                    board[row][col] = value
                    domains[var].remove(value)
                    
                    if callback:
                        callback_board = [row[:] for row in board]
                        callback_state = PuzzleState(callback_board)
                        callback(callback_state, nodes_expanded, 0, time.time() - start_time)
                    
                    result = backtrack(board, remaining_positions)
                    if result:
                        return result
                    
                    # Quay lui
                    board[row][col] = -1
                    domains[var].append(value)
            
            return None
        
        # Tạo danh sách các vị trí cần gán
        positions = [(i, j) for i in range(3) for j in range(3)]
        
        # Bắt đầu backtracking
        result_board = backtrack(empty_board, positions)
        
        end_time = time.time()
        
        if result_board:
            # Chuyển đổi kết quả thành PuzzleState
            final_state = PuzzleState(result_board)
            
            # Tạo đường đi giả định (chỉ trạng thái cuối)
            path = [final_state]
            
            return {
                "path": path,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "message": "Cấu hình được xây dựng bằng CSP Intelligent Backtracking với MRV và LCV"
            }
        else:
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "message": "Không thể xây dựng cấu hình hợp lệ"
            }

    def min_conflicts(self, initial_state=None, callback=None, max_steps=1000):
        """
        Thuật toán Min-Conflicts xây dựng cấu hình 8-puzzle bằng cách giải quyết xung đột.
        
        Min-Conflicts bắt đầu với một cấu hình ngẫu nhiên và lặp đi lặp lại việc
        giải quyết xung đột bằng cách chọn biến có xung đột và gán giá trị
        gây ít xung đột nhất.
        
        Tham số:
            initial_state: Không sử dụng, chỉ để tương thích với giao diện
            callback: Hàm gọi lại để hiển thị tiến trình
            max_steps: Số bước tối đa
            
        Trả về:
            dict: Kết quả chứa cấu hình xây dựng được
        """
        import random
        start_time = time.time()
        nodes_expanded = 0
        
        # Tạo một cấu hình ngẫu nhiên ban đầu
        # (đảm bảo mỗi số từ 0-8 xuất hiện đúng một lần)
        values = list(range(9))
        random.shuffle(values)
        board = [values[i*3:(i+1)*3] for i in range(3)]
        
        # Hàm đếm số xung đột
        def count_conflicts(board):
            # Trong 8-puzzle, xung đột chỉ là khi một số không nằm đúng vị trí của nó
            # Chúng ta đếm số ô không nằm đúng vị trí (khoảng cách Manhattan)
            conflicts = 0
            for i in range(3):
                for j in range(3):
                    value = board[i][j]
                    if value != 0:  # Bỏ qua ô trống
                        # Vị trí đích của giá trị này trong goal_state
                        target_i, target_j = (value - 1) // 3, (value - 1) % 3
                        if value == 0:
                            target_i, target_j = 2, 2
                        conflicts += abs(i - target_i) + abs(j - target_j)
            return conflicts
        
        # Hàm đếm xung đột khi hoán đổi hai giá trị
        def count_conflicts_with_swap(board, pos1, pos2):
            # Tạo bản sao và thực hiện hoán đổi
            new_board = [row[:] for row in board]
            i1, j1 = pos1
            i2, j2 = pos2
            new_board[i1][j1], new_board[i2][j2] = new_board[i2][j2], new_board[i1][j1]
            return count_conflicts(new_board)
        
        # Thuật toán Min-Conflicts
        conflicts = count_conflicts(board)
        steps = 0
        
        while conflicts > 0 and steps < max_steps:
            nodes_expanded += 1
            steps += 1
            
            # Tạo danh sách tất cả các cặp vị trí có thể hoán đổi
            positions = [(i, j) for i in range(3) for j in range(3)]
            pos_pairs = [(p1, p2) for p1 in positions for p2 in positions if p1 < p2]
            
            # Tìm cặp vị trí mà khi hoán đổi sẽ giảm xung đột nhất
            best_swap = None
            min_new_conflicts = conflicts
            
            for pos1, pos2 in pos_pairs:
                new_conflicts = count_conflicts_with_swap(board, pos1, pos2)
                if new_conflicts < min_new_conflicts:
                    min_new_conflicts = new_conflicts
                    best_swap = (pos1, pos2)
            
            # Nếu không tìm thấy cặp nào giảm xung đột, chọn ngẫu nhiên để tránh bị kẹt
            if not best_swap or min_new_conflicts >= conflicts:
                best_swap = random.choice(pos_pairs)
                min_new_conflicts = count_conflicts_with_swap(board, best_swap[0], best_swap[1])
            
            # Thực hiện hoán đổi
            i1, j1 = best_swap[0]
            i2, j2 = best_swap[1]
            board[i1][j1], board[i2][j2] = board[i2][j2], board[i1][j1]
            conflicts = min_new_conflicts
            
            # Trong các hàm backtrack đệ quy:
            if callback:
                callback_board = [row[:] for row in board]
                # Không cần thay -1 bằng 0 vì chúng ta hiển thị trực tiếp các giá trị đang được đặt
                callback_state = PuzzleState(callback_board)
                callback(callback_state, nodes_expanded, 0, time.time() - start_time)
            
            # Nếu đạt được mục tiêu, dừng lại
            if conflicts == 0:
                break
        
        end_time = time.time()
        
        # Chuyển đổi kết quả thành PuzzleState
        final_state = PuzzleState([row[:] for row in board])
        
        if conflicts == 0:
            return {
                "path": [final_state],  # Không có đường đi, chỉ có trạng thái cuối
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "steps": steps,
                "message": "Cấu hình được xây dựng bằng CSP Min-Conflicts"
            }
        else:
            return {
                "path": None,
                "partial_state": final_state,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "steps": steps,
                "conflicts": conflicts,
                "message": f"Min-Conflicts không thể tìm được cấu hình không xung đột sau {max_steps} bước"
            }
    def q_learning(self, initial_state, callback=None, heuristic='manhattan', 
                alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000, max_steps=100):
        """
        Triển khai thuật toán Q-learning để giải bài toán 8-puzzle.
        
        Q-learning là thuật toán học tăng cường không dựa trên mô hình, học giá trị
        của các hành động trong các trạng thái bằng cách trải nghiệm môi trường qua nhiều tập.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho quá trình học
            callback (function, tùy chọn): Hàm gọi lại để hiển thị quá trình. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic để đánh giá trạng thái ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            alpha (float, tùy chọn): Tốc độ học (0 < alpha <= 1). Mặc định là 0.1.
            gamma (float, tùy chọn): Hệ số chiết khấu (0 <= gamma <= 1). Mặc định là 0.9.
            epsilon (float, tùy chọn): Tỷ lệ khám phá. Mặc định là 0.1.
            episodes (int, tùy chọn): Số tập huấn luyện. Mặc định là 1000.
            max_steps (int, tùy chọn): Số bước tối đa cho mỗi tập. Mặc định là 100.
            
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, thông tin huấn luyện và thời gian thực hiện
        """
        import random
        import numpy as np
        from collections import defaultdict
        
        start_time = time.time()
        
        # Choose heuristic function
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Initialize Q-table as a defaultdict to handle unseen states
        # Format: Q[(state_key, action)] = q_value
        Q = defaultdict(float)
        
        # Track statistics
        nodes_expanded = 0
        total_rewards = []
        best_solution_path = None
        best_solution_length = float('inf')
        
        # Create a discrete state representation (needed for dictionary keys)
        def state_to_key(state):
            return str(state.board)
        
        # Get valid actions for a state
        def get_valid_actions(state):
            return [child.move for child in state.get_children()]
        
        # Choose action using epsilon-greedy policy
        def select_action(state, valid_actions):
            state_key = state_to_key(state)
            
            # Explore: random action
            if random.random() < epsilon:
                return random.choice(valid_actions)
            
            # Exploit: best known action
            q_values = [Q[(state_key, action)] for action in valid_actions]
            max_q = max(q_values)
            
            # Handle multiple actions with the same max Q-value
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
        
        # Take action and get next state
        def take_action(state, action):
            children = state.get_children()
            for child in children:
                if child.move == action:
                    return child
            return None  # Should never reach here if action is valid
        
        # Training loop
        for episode in range(episodes):
            # Start from initial state
            current_state = PuzzleState([row[:] for row in initial_state.board])
            episode_rewards = 0
            path = [current_state]
            
            for step in range(max_steps):
                nodes_expanded += 1
                
                # Check if goal reached
                if self.is_goal(current_state):
                    # Large positive reward for reaching goal
                    episode_rewards += 100
                    
                    # Check if this is the best solution found so far
                    if len(path) < best_solution_length:
                        best_solution_path = path[:]
                        best_solution_length = len(path)
                    
                    break
                
                # Get valid actions
                valid_actions = get_valid_actions(current_state)
                if not valid_actions:  # No valid moves
                    break
                    
                # Select action using epsilon-greedy
                action = select_action(current_state, valid_actions)
                
                # Take action and observe next state
                next_state = take_action(current_state, action)
                
                # Compute reward (negative step cost, encourage shorter paths)
                reward = -1
                
                # If next state is closer to goal, give small positive reward
                current_h = h_func(current_state)
                next_h = h_func(next_state)
                if next_h < current_h:
                    reward += 0.5
                
                # Update Q-value using the Q-learning formula
                current_q = Q[(state_to_key(current_state), action)]
                
                if self.is_goal(next_state):
                    # Terminal state
                    max_next_q = 0
                    reward = 100  # High reward for reaching goal
                else:
                    # Non-terminal state
                    next_valid_actions = get_valid_actions(next_state)
                    max_next_q = max([Q[(state_to_key(next_state), a)] for a in next_valid_actions], default=0)
                
                # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
                new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                Q[(state_to_key(current_state), action)] = new_q
                
                # Update state and path
                current_state = next_state
                path.append(current_state)
                episode_rewards += reward
                
                # Optional callback for visualization
                if callback and episode == episodes - 1:  # Only show last episode
                    callback(current_state, nodes_expanded, 0, time.time() - start_time)
                    
            total_rewards.append(episode_rewards)
            
            # Decay exploration rate over time
            epsilon = max(0.01, epsilon * 0.99)
            
        # After training, generate solution path using learned policy
        if not best_solution_path:
            current_state = PuzzleState([row[:] for row in initial_state.board])
            solution_path = [current_state]
            
            for step in range(max_steps):
                if self.is_goal(current_state):
                    break
                    
                valid_actions = get_valid_actions(current_state)
                if not valid_actions:
                    break
                
                # Choose best action based on Q-values (no exploration)
                state_key = state_to_key(current_state)
                q_values = {action: Q[(state_key, action)] for action in valid_actions}
                best_action = max(q_values, key=q_values.get)
                
                next_state = take_action(current_state, best_action)
                current_state = next_state
                solution_path.append(current_state)
            
            if self.is_goal(solution_path[-1]):
                best_solution_path = solution_path
        
        end_time = time.time()
        
        # Return results
        if best_solution_path and self.is_goal(best_solution_path[-1]):
            return {
                "path": best_solution_path,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,  # Not applicable for Q-learning
                "time": end_time - start_time,
                "episodes": episodes,
                "final_epsilon": epsilon,
                "avg_reward_per_episode": np.mean(total_rewards) if total_rewards else 0
            }
        else:
            return {
                "path": None,
                "nodes_expanded": nodes_expanded,
                "max_queue_size": 1,
                "time": end_time - start_time,
                "episodes": episodes,
                "final_epsilon": epsilon,
                "avg_reward_per_episode": np.mean(total_rewards) if total_rewards else 0
            }
    def belief_state_search(self, initial_state, callback=None, num_hidden=2, max_depth=10):
        """
        Implements Belief State Search for a partially observable 8-puzzle.
        
        In this version, some tiles are initially hidden, and the agent must
        maintain beliefs about possible tile configurations.
        
        Parameters:
            initial_state (PuzzleState): Starting state for the search
            callback (function, optional): Callback function for visualization
            num_hidden (int, optional): Number of hidden tiles
            max_depth (int, optional): Maximum search depth
        """
        import random
        import copy
        from itertools import permutations
        
        start_time = time.time()
        
        # Create a partially observable initial state by hiding some tiles
        visible_board = [row[:] for row in initial_state.board]
        flat_indices = [(i, j) for i in range(3) for j in range(3)]
        
        # Choose positions to hide (excluding the blank)
        non_blank_positions = []
        for i, j in flat_indices:
            if visible_board[i][j] != 0:  # Not the blank space
                non_blank_positions.append((i, j))
        
        hidden_positions = random.sample(non_blank_positions, min(num_hidden, len(non_blank_positions)))
        
        # Hide the selected tiles
        hidden_values = []
        for i, j in hidden_positions:
            hidden_values.append(visible_board[i][j])
            visible_board[i][j] = -1  # Mark as hidden
        
        # Generate all possible assignments of values to hidden tiles
        possible_assignments = list(permutations(hidden_values))
        
        # Create initial belief state (set of possible board states)
        belief_boards = []
        for assignment in possible_assignments:
            board = [row[:] for row in visible_board]
            for idx, (i, j) in enumerate(hidden_positions):
                board[i][j] = assignment[idx]
            belief_boards.append(board)
        
        # Stats tracking
        nodes_expanded = 0
        max_frontier_size = 1
        
        # Create search frontier with (depth, tiebreaker, state, path)
        from queue import PriorityQueue
        frontier = PriorityQueue()
        
        # Create a new initial state with hidden tiles
        obs_initial_state = PuzzleState(visible_board)
        frontier.put((0, 0, obs_initial_state, []))  # (depth, tiebreaker, state, path)
        
        visited = set([obs_initial_state.key])
        tiebreaker = 1
        
        # Utility function to check if a board could be the goal
        def could_be_goal(board):
            # For each belief state, check if it matches the goal
            for belief_board in belief_boards:
                matches = True
                for i in range(3):
                    for j in range(3):
                        if board[i][j] == -1:  # Hidden tile
                            if belief_board[i][j] != self.goal_state[i][j]:
                                matches = False
                                break
                        elif board[i][j] != self.goal_state[i][j]:
                            matches = False
                            break
                    if not matches:
                        break
                if matches:
                    return True
            return False
        
        # Function to update beliefs when a previously hidden tile is moved
        def update_beliefs(current_board, next_board):
            nonlocal belief_boards
            
            # Find which position changed
            changed_pos = []
            for i in range(3):
                for j in range(3):
                    if current_board[i][j] != next_board[i][j]:
                        changed_pos.append((i, j))
            
            # If a hidden tile is now visible, update beliefs
            for i, j in changed_pos:
                if current_board[i][j] == -1 and next_board[i][j] != -1:
                    # Filter belief states that match the revealed tile
                    new_belief_boards = []
                    for belief_board in belief_boards:
                        if belief_board[i][j] == next_board[i][j]:
                            new_belief_boards.append(belief_board)
                    belief_boards = new_belief_boards
                    break
        
        while not frontier.empty():
            depth, _, current, path = frontier.get()
            current_board = current.board
            
            nodes_expanded += 1
            
            if callback:
                callback(current, nodes_expanded, frontier.qsize(), time.time() - start_time)
            
            # Check if current state could be the goal in any belief state
            if could_be_goal(current_board):
                end_time = time.time()
                return {
                    "path": path + [current],
                    "nodes_expanded": nodes_expanded,
                    "max_frontier_size": max_frontier_size,
                    "time": end_time - start_time,
                    "belief_size": len(belief_boards)
                }
            
            if depth >= max_depth:
                continue
            
            # Expand children
            for child in current.get_children():
                # Update beliefs if a hidden tile is revealed
                child_board = [row[:] for row in child.board]
                update_beliefs(current_board, child_board)
                
                # Add child to frontier if not visited
                if child.key not in visited:
                    visited.add(child.key)
                    frontier.put((depth + 1, tiebreaker, child, path + [current]))
                    tiebreaker += 1
            
            max_frontier_size = max(max_frontier_size, frontier.qsize())
        
        # No solution found
        end_time = time.time()
        return {
            "path": None,
            "nodes_expanded": nodes_expanded,
            "max_frontier_size": max_frontier_size,
            "time": end_time - start_time,
            "belief_size": len(belief_boards)
        }
    def and_or_graph_search(self, initial_state, callback=None, slip_probability=0.2, max_depth=10):
        """
        Implements AND-OR Graph Search for a nondeterministic 8-puzzle.
        
        In this version, moves have a probability of "slipping" to a different direction.
        The algorithm builds a conditional plan that handles all possible outcomes.
        
        Parameters:
            initial_state (PuzzleState): Starting state for the search
            callback (function, optional): Callback function for visualization
            slip_probability (float, optional): Probability that a move slips
            max_depth (int, optional): Maximum search depth
        """
        start_time = time.time()
        
        # Define moves and their possible slip directions
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        slip_moves = {
            "UP": ["LEFT", "RIGHT"],
            "DOWN": ["LEFT", "RIGHT"],
            "LEFT": ["UP", "DOWN"],
            "RIGHT": ["UP", "DOWN"]
        }
        
        # Track visited states and nodes expanded
        visited = {}
        nodes_expanded = 0
        
        def or_search(state, depth):
            """OR node: Agent chooses an action to maximize chance of success"""
            nonlocal nodes_expanded
            
            # Check if state is a goal
            if self.is_goal(state):
                return True, {}
            
            # Check depth limit
            if depth >= max_depth:
                return False, {}
            
            # Check if state already visited
            state_key = state.key
            if state_key in visited:
                return visited[state_key]
            
            nodes_expanded += 1
            
            if callback:
                callback(state, nodes_expanded, 0, time.time() - start_time)
            
            # Try each possible action
            for move in moves:
                # Get child states for this move
                child_states = []
                intended_child = None
                
                for child in state.get_children():
                    if child.move == move:
                        intended_child = child
                        break
                
                if not intended_child:  # Move not valid
                    continue
                
                # Add the intended outcome
                child_states.append((intended_child, 1.0 - slip_probability))
                
                # Add possible slip outcomes
                for slip_move in slip_moves[move]:
                    for child in state.get_children():
                        if child.move == slip_move:
                            child_states.append((child, slip_probability / len(slip_moves[move])))
                            break
                
                # Check if AND node succeeds for all outcomes
                and_succeeds = True
                sub_plans = {}
                
                for child, _ in child_states:
                    success, plan = or_search(child, depth + 1)
                    if not success:
                        and_succeeds = False
                        break
                    sub_plans[child.key] = plan
                
                if and_succeeds:
                    # This action works for all outcomes
                    plan = {move: sub_plans}
                    visited[state_key] = (True, plan)
                    return True, plan
            
            # No action works
            visited[state_key] = (False, {})
            return False, {}
        
        # Start the AND-OR search
        success, plan = or_search(initial_state, 0)
        
        end_time = time.time()
        
        # Try to construct a sample path from the plan
        path = None
        if success:
            # Simulate executing the plan in the most likely scenario
            current_state = initial_state
            path = [current_state]
            
            while not self.is_goal(current_state) and len(path) < max_depth * 2:
                # Find the action for current state
                state_key = current_state.key
                if state_key in plan:
                    move = list(plan[state_key].keys())[0]
                    
                    # Execute the move
                    for child in current_state.get_children():
                        if child.move == move:
                            current_state = child
                            path.append(current_state)
                            break
                    
                    # Update plan
                    plan = plan[state_key][move]
                else:
                    break
        
        return {
            "path": path,
            "plan": plan if success else None,
            "success": success,
            "nodes_expanded": nodes_expanded,
            "time": end_time - start_time
        }
    def genetic_algorithm(self, initial_state, callback=None, heuristic='manhattan', 
                        population_size=50, generations=100, mutation_rate=0.2):
        """
        Thuật toán di truyền (Genetic Algorithm) để giải 8-puzzle.
        
        Thuật toán di truyền mô phỏng quá trình tiến hóa tự nhiên, sử dụng các toán tử như
        lai ghép (crossover), đột biến (mutation), và lựa chọn (selection) để tìm giải pháp.
        Mỗi cá thể (individual) trong quần thể (population) là một trạng thái puzzle.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái bắt đầu cho tìm kiếm
            callback (function, tùy chọn): Hàm gọi lại cho mỗi thế hệ. Mặc định là None.
            heuristic (str, tùy chọn): Heuristic sử dụng ('manhattan' hoặc 'misplaced'). Mặc định là 'manhattan'.
            population_size (int, tùy chọn): Kích thước quần thể. Mặc định là 50.
            generations (int, tùy chọn): Số thế hệ tối đa. Mặc định là 100.
            mutation_rate (float, tùy chọn): Tỷ lệ đột biến (0.0 đến 1.0). Mặc định là 0.2.
        
        Trả về:
            dict: Kết quả chứa đường đi, số nút đã mở rộng, thời gian thực hiện và thế hệ cuối
        """
        import random
        import time
        
        start_time = time.time()
        
        # Chọn hàm heuristic
        if heuristic == 'manhattan':
            h_func = self.get_manhattan_distance
        else:
            h_func = self.get_misplaced_tiles
        
        # Lưu vị trí của ô trống ban đầu
        initial_board = initial_state.board
        blank_position = None
        for i in range(3):
            for j in range(3):
                if initial_board[i][j] == 0:
                    blank_position = (i, j)
                    break
            if blank_position:
                break
        
        # Khởi tạo quần thể ban đầu là các chuỗi di chuyển ngẫu nhiên
        population = []
        for _ in range(population_size):
            # Mỗi cá thể là một chuỗi di chuyển (tối đa 50 bước)
            moves = []
            for _ in range(random.randint(5, 30)):
                moves.append(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
            population.append(moves)
        
        # Hàm tính độ thích nghi (fitness): Giá trị thấp hơn = tốt hơn
        def calculate_fitness(moves):
            # Áp dụng chuỗi di chuyển lên bảng ban đầu
            board = [row[:] for row in initial_board]
            blank_i, blank_j = blank_position
            valid_moves = 0
            
            for move in moves:
                new_i, new_j = blank_i, blank_j
                if move == "UP" and blank_i > 0:
                    new_i -= 1
                elif move == "DOWN" and blank_i < 2:
                    new_i += 1
                elif move == "LEFT" and blank_j > 0:
                    new_j -= 1
                elif move == "RIGHT" and blank_j < 2:
                    new_j += 1
                else:
                    # Di chuyển không hợp lệ, bỏ qua
                    continue
                
                # Thực hiện di chuyển
                board[blank_i][blank_j] = board[new_i][new_j]
                board[new_i][new_j] = 0
                blank_i, blank_j = new_i, new_j
                valid_moves += 1
            
            # Tạo trạng thái mới để đánh giá
            state = PuzzleState(board)
            
            # Độ thích nghi là heuristic cộng với phạt cho chuỗi di chuyển dài
            fitness = h_func(state) + 0.01 * len(moves)
            
            # Nếu tìm thấy giải pháp, độ thích nghi sẽ rất thấp
            if self.is_goal(state):
                fitness = 0
                
            return fitness, state, valid_moves
        
        # Hàm lựa chọn (Tournament Selection)
        def selection(population, k=3):
            selected = random.sample(population, k)
            return min(selected, key=lambda x: calculate_fitness(x)[0])
        
        # Hàm lai ghép (Crossover)
        def crossover(parent1, parent2):
            if not parent1 or not parent2:
                return parent1[:] if parent1 else parent2[:]
            
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:point] + parent2[point:]
            return child
        
        # Hàm đột biến (Mutation)
        def mutate(individual, mutation_rate):
            mutated = individual[:]
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    mutated[i] = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            
            # Thêm hoặc xóa một di chuyển
            if random.random() < mutation_rate:
                if random.random() < 0.5 and mutated:
                    index = random.randint(0, len(mutated) - 1)
                    mutated.pop(index)
                else:
                    mutated.append(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
                    
            return mutated
        
        # Biến theo dõi thống kê
        nodes_expanded = 0
        best_fitness = float('inf')
        best_state = None
        best_individual = None
        best_valid_moves = 0
        
        # Tiến hóa qua các thế hệ
        for generation in range(generations):
            # Đánh giá quần thể
            evaluated_population = []
            for individual in population:
                fitness, state, valid_moves = calculate_fitness(individual)
                evaluated_population.append((individual, fitness))
                nodes_expanded += 1
                
                # Lưu lại cá thể tốt nhất
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual
                    best_state = state
                    best_valid_moves = valid_moves
                    
                # Kiểm tra nếu tìm thấy giải pháp
                if fitness == 0:
                    # Tạo đường đi từ chuỗi di chuyển
                    path = self.generate_path_from_moves(initial_state, individual[:valid_moves])
                    end_time = time.time()
                    return {
                        "path": path,
                        "nodes_expanded": nodes_expanded,
                        "max_queue_size": population_size,
                        "time": end_time - start_time,
                        "generations": generation + 1,
                        "final_fitness": fitness
                    }
            
            if callback:
                callback(best_state, nodes_expanded, population_size, time.time() - start_time)
            
            # Tạo thế hệ mới
            new_population = []
            
            # Giữ lại các cá thể tốt nhất (elitism)
            elite_size = max(1, int(population_size * 0.1))
            sorted_population = sorted(evaluated_population, key=lambda x: x[1])
            new_population.extend([ind for ind, _ in sorted_population[:elite_size]])
            
            # Tạo phần còn lại của quần thể mới
            while len(new_population) < population_size:
                parent1 = selection(population)
                parent2 = selection(population)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
        
        # Không tìm thấy giải pháp trong số thế hệ cho phép
        end_time = time.time()
        
        # Tạo đường đi từ chuỗi di chuyển tốt nhất
        path = self.generate_path_from_moves(initial_state, best_individual[:best_valid_moves])
        
        return {
            "path": path if self.is_goal(best_state) else None,
            "partial_path": path,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": population_size,
            "time": end_time - start_time,
            "generations": generations,
            "final_fitness": best_fitness
        }


    def generate_path_from_moves(self, initial_state, moves):
        """
        Tạo đường đi từ trạng thái ban đầu và chuỗi di chuyển.
        
        Tham số:
            initial_state (PuzzleState): Trạng thái ban đầu
            moves (list): Danh sách các di chuyển ("UP", "DOWN", "LEFT", "RIGHT")
            
        Trả về:
            list: Danh sách các PuzzleState tạo thành đường đi
        """
        path = [initial_state]
        current_state = initial_state
        board = [row[:] for row in initial_state.board]
        
        # Tìm vị trí ô trống
        blank_i, blank_j = None, None
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    blank_i, blank_j = i, j
                    break
            if blank_i is not None:
                break
        
        for move in moves:
            new_i, new_j = blank_i, blank_j
            
            if move == "UP" and blank_i > 0:
                new_i -= 1
            elif move == "DOWN" and blank_i < 2:
                new_i += 1
            elif move == "LEFT" and blank_j > 0:
                new_j -= 1
            elif move == "RIGHT" and blank_j < 2:
                new_j += 1
            else:
                # Di chuyển không hợp lệ, bỏ qua
                continue
            
            # Thực hiện di chuyển
            new_board = [row[:] for row in board]
            new_board[blank_i][blank_j] = new_board[new_i][new_j]
            new_board[new_i][new_j] = 0
            
            # Tạo trạng thái mới và thêm vào đường đi
            new_state = PuzzleState(
                new_board, 
                parent=current_state, 
                move=move, 
                depth=current_state.depth + 1,
                cost=current_state.depth + 1
            )
            path.append(new_state)
            current_state = new_state
            board = new_board
            blank_i, blank_j = new_i, new_j
        
        return path



class PuzzleGUI:
    """
    Giao diện đồ họa người dùng cho bộ giải 8-puzzle.

    Lớp này tạo và quản lý giao diện đồ họa Tkinter để hiển thị puzzle,
    chọn thuật toán và trực quan hóa giải pháp.

    Thuộc tính:
        root (tk.Tk): Cửa sổ Tkinter chính
        initial_state (list): Ma trận 3x3 biểu diễn cấu hình puzzle ban đầu
        goal_state (list): Ma trận 3x3 biểu diễn cấu hình đích
        solver (PuzzleSolver): Instance của PuzzleSolver để chạy thuật toán tìm kiếm
        solution_path (list): Danh sách các trạng thái trong đường đi giải pháp
        current_step (int): Vị trí hiện tại trong trực quan hóa giải pháp
        animation_speed (int): Độ trễ giữa các bước tính bằng mili giây
    """
    def plot_benchmark_results(self, results):
        """Vẽ đồ thị so sánh hiệu suất các thuật toán từ kết quả benchmark."""
        # Xóa các đồ thị cũ
        for widget in self.viz_container.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Chuẩn bị dữ liệu
        algorithms = []
        times = []
        nodes = []
        solutions_found = []
        
        for algo, data in results.items():
            algorithms.append(algo)
            times.append(data['time'])
            nodes.append(data['nodes'])
            solutions_found.append(1 if data['found_solution'] else 0)
        
        # Chuẩn bị vị trí các thanh trên trục x
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Vẽ biểu đồ thời gian thực hiện
        ax1.bar(x, times, width, label='Time (s)')
        ax1.set_title('Algorithm Execution Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.set_ylabel('Time (seconds)')
        
        # Vẽ biểu đồ số nodes mở rộng
        ax2.bar(x, nodes, width, label='Nodes Expanded', color='orange')
        ax2.set_title('Nodes Expanded by Algorithm')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.set_ylabel('Number of Nodes')
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_solution_stats(self):
        """Vẽ đồ thị phân tích giải pháp hiện tại."""
        if not self.solution_path:
            messagebox.showinfo("No Solution", "Please solve the puzzle first to generate statistics.")
            return
        
        # Xóa các đồ thị cũ
        for widget in self.viz_container.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Chuẩn bị dữ liệu
        steps = list(range(len(self.solution_path)))
        
        # Tính toán các giá trị heuristic cho đường dẫn
        manhattan_values = []
        misplaced_values = []
        
        for state in self.solution_path:
            manhattan_values.append(self.solver.get_manhattan_distance(state))
            misplaced_values.append(self.solver.get_misplaced_tiles(state))
        
        # Vẽ biểu đồ heuristic theo bước
        ax1.plot(steps, manhattan_values, 'b-', label='Manhattan Distance')
        ax1.plot(steps, misplaced_values, 'r-', label='Misplaced Tiles')
        ax1.set_title('Heuristic Values Along Solution Path')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Heuristic Value')
        ax1.legend()
        
        # Vẽ biểu đồ tốc độ hội tụ
        ax2.plot(steps, [len(self.solution_path) - i - 1 for i in steps], 'g-')
        ax2.set_title('Distance to Goal')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Steps Remaining')
        
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_search_tree(self):
        """Vẽ biểu đồ cây tìm kiếm đơn giản."""
        if not self.solution_path:
            messagebox.showinfo("No Solution", "Please solve the puzzle first.")
            return
        
        # Xóa các đồ thị cũ
        for widget in self.viz_container.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tạo một biểu đồ đơn giản hiển thị cấu trúc cây
        # (Đây chỉ là một minh họa đơn giản)
        x = list(range(len(self.solution_path)))
        y = [0] * len(self.solution_path)
        
        ax.plot(x, y, 'bo-')  # path
        
        # Thêm các nút con để tạo hiệu ứng cây (giả lập)
        for i in range(len(self.solution_path)-1):
            # Thêm một số nút con giả định
            num_children = min(3, i+1)
            for j in range(num_children):
                ax.plot([i, i+1], [j*0.5-0.5, 0], 'r--')
        
        ax.set_title('Simplified Search Tree Visualization')
        ax.set_xlabel('Depth')
        ax.set_yticks([])  # Hide y ticks
        
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def csp_callback(self, state, nodes_expanded, queue_size, elapsed_time):
        """Special callback for CSP algorithms to show intermediate steps."""
        self.update_puzzle_display(state.board)
        self.info_label.config(text=f"Building solution using CSP...\n"
                            f"Nodes expanded: {nodes_expanded}\n"
                            f"Time elapsed: {elapsed_time:.2f}s")
        self.root.update()
        time.sleep(0.1)  # Short delay to visualize the steps
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.geometry("800x600")
        
        # Set initial and goal states
        self.initial_state = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
        # new initial state for hill climbing
        # self.initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
        self.solver = PuzzleSolver()
        
        # Initialize variables needed by multiple methods
        self.speed_var = tk.IntVar(value=500)
        self.difficulty_var = tk.StringVar(value="medium")
        self.show_goal_var = tk.BooleanVar(value=False)
        self.algorithm_var = tk.StringVar(value="bfs")
        
        # Create GUI components
        self.create_widgets()
        
        # Solution variables
        self.solution_path = None
        self.current_step = 0
        self.animation_speed = 500  # milliseconds

    def save_configuration(self):
        """Save the current puzzle configuration to a file."""
        from tkinter import filedialog
        import json
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Puzzle Configuration"
        )
        
        if not file_path:
            return
        
        config = {
            "initial_state": self.initial_state,
            "goal_state": self.goal_state,
            "algorithm": self.algorithm_var.get(),
            "animation_speed": self.speed_var.get()
        }
        
        try:
            with open(file_path, 'w') as file:
                json.dump(config, file)
            messagebox.showinfo("Success", "Configuration saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def load_configuration(self):
        """Load a puzzle configuration from a file."""
        from tkinter import filedialog
        import json
        
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Puzzle Configuration"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
            
            self.initial_state = config.get("initial_state", self.initial_state)
            self.goal_state = config.get("goal_state", self.goal_state)
            self.algorithm_var.set(config.get("algorithm", "bfs"))
            self.speed_var.set(config.get("animation_speed", 500))
            
            # Reset display with new configuration
            self.update_puzzle_display(self.initial_state)
            messagebox.showinfo("Success", "Configuration loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def generate_random_puzzle(self):
        """Generate a random 8-puzzle configuration based on the selected difficulty."""
        import random
        
        difficulty = self.difficulty_var.get()
        moves = 10  # Easy by default
        
        if difficulty == "medium":
            moves = 25
        elif difficulty == "hard":
            moves = 40
        
        # Start from the goal state
        board = [row[:] for row in self.goal_state]
        
        # Find the blank position
        blank_i, blank_j = None, None
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    blank_i, blank_j = i, j
                    break
            if blank_i is not None:
                break
        
        # Make random moves
        for _ in range(moves):
            # Get possible moves
            possible_moves = []
            
            if blank_i > 0:  # Can move UP
                possible_moves.append((-1, 0))
            if blank_i < 2:  # Can move DOWN
                possible_moves.append((1, 0))
            if blank_j > 0:  # Can move LEFT
                possible_moves.append((0, -1))
            if blank_j < 2:  # Can move RIGHT
                possible_moves.append((0, 1))
            
            # Choose a random move
            di, dj = random.choice(possible_moves)
            new_i, new_j = blank_i + di, blank_j + dj
            
            # Make the move
            board[blank_i][blank_j] = board[new_i][new_j]
            board[new_i][new_j] = 0
            blank_i, new_i = new_i, blank_i
            blank_j, new_j = new_j, blank_j
        
        # Update the initial state and display
        self.initial_state = board
        self.update_puzzle_display(board)
        self.reset_puzzle()

    def toggle_goal_display(self):
        """Toggle between showing the current state and goal state."""
        if self.show_goal_var.get():
            # Save current display if not already saved
            if not hasattr(self, 'current_display'):
                self.current_display = [row[:] for row in self.initial_state]
            self.update_puzzle_display(self.goal_state)
        else:
            # Restore previous display
            if hasattr(self, 'current_display'):
                self.update_puzzle_display(self.current_display)
                delattr(self, 'current_display')

    def run_benchmark(self):
        """Run a benchmark comparing multiple algorithms on the current puzzle."""
        import threading
        
        # Disable buttons during benchmark
        self.root.config(cursor="wait")
        
        # Create a new window for results
        benchmark_window = tk.Toplevel(self.root)
        benchmark_window.title("Algorithm Benchmark Results")
        benchmark_window.geometry("600x400")
        
        # Results text widget
        result_frame = ttk.Frame(benchmark_window, padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        result_text = tk.Text(result_frame, wrap=tk.WORD, yscrollcommand=result_scroll.set)
        result_text.pack(fill=tk.BOTH, expand=True)
        result_scroll.config(command=result_text.yview)
        # Add a button to visualize results
        viz_button = ttk.Button(benchmark_window, text="Visualize Results", state=tk.DISABLED)
        viz_button.pack(pady=5)
        
        initial = PuzzleState(self.initial_state)
        
        # Algorithms to benchmark
        algorithms = [
            ("BFS", lambda: self.solver.bfs(initial)),
            ("DFS", lambda: self.solver.dfs(initial)),
            ("A* (Manhattan)", lambda: self.solver.a_star_search(initial, heuristic='manhattan')),
            ("Greedy (Manhattan)", lambda: self.solver.greedy_search(initial, heuristic='manhattan')),
            ("Hill Climbing (Manhattan)", lambda: self.solver.steepest_ascent_hill_climbing(initial, heuristic='manhattan')),
            ("Simulated Annealing", lambda: self.solver.simulated_annealing(initial)),
            ("Genetic Algorithm", lambda: self.solver.genetic_algorithm(initial, heuristic='manhattan')),
            ("IDA*", lambda: self.solver.ida_star_search(initial, heuristic='manhattan')),
            ("Min-Conflicts", lambda: self.solver.min_conflicts()),
            ("Backtracking", lambda: self.solver.backtracking_search()),
            ("Intelligent Backtracking (CSP)", lambda: self.solver.intelligent_backtracking()),
            ("Min-Conflicts (CSP)", lambda: self.solver.min_conflicts(initial, max_steps=1000)),
            ("Beam Search", lambda: self.solver.beam_search(initial, beam_width=3)),
            ("Q-Learning", lambda: self.solver.q_learning(initial, heuristic='manhattan', episodes=1000, max_steps=100)),
            ("AND-OR Graph Search", lambda: self.solver.and_or_graph_search(initial, slip_probability=0.2, max_depth=20)),
            ("Belief State Search", lambda: self.solver.belief_state_search(initial, max_depth=10))
        ]
        # Store benchmark results
        benchmark_results = {}
        def run_tests():
            for name, algo_func in algorithms:
                result_text.insert(tk.END, f"Running {name}...\n")
                benchmark_window.update()
                
                try:
                    start_time = time.time()
                    result = algo_func()
                    elapsed = time.time() - start_time
                    # Store results for visualization
                    benchmark_results[name] = {
                        'time': elapsed,
                        'nodes': result['nodes_expanded'],
                        'found_solution': result["path"] is not None
                    }
                    
                    # Display results
                    if result["path"]:
                        steps = len(result["path"]) - 1
                        result_text.insert(tk.END, f"✓ {name}: Solution found in {steps} steps\n")
                        result_text.insert(tk.END, f"  Time: {elapsed:.6f}s, Nodes: {result['nodes_expanded']}\n\n")
                    else:
                        result_text.insert(tk.END, f"✗ {name}: No solution found\n")
                        result_text.insert(tk.END, f"  Time: {elapsed:.6f}s, Nodes: {result['nodes_expanded']}\n\n")
                except Exception as e:
                    result_text.insert(tk.END, f"✗ {name}: Error - {str(e)}\n\n")
                    benchmark_results[name] = {
                        'time': 0,
                        'nodes': 0,
                        'found_solution': False
                    }
                
                benchmark_window.update()
            
            result_text.insert(tk.END, "Benchmark complete!\n")
            self.root.config(cursor="")
            
            # Enable visualization button
            viz_button.config(state=tk.NORMAL, command=lambda: self.plot_benchmark_results(benchmark_results))

        
        # Run benchmark in a separate thread
        threading.Thread(target=run_tests, daemon=True).start()

    def show_about(self):
        """Show about dialog with information about the application."""
        about_text = """8-Puzzle Solver v1.0
        
        This application implements various search algorithms 
        to solve the classic 8-puzzle problem.

        Algorithms include BFS, DFS, A*, Hill Climbing, 
        Simulated Annealing, Beam Search, Genetic, CSP,...!

        Created by: DoKienHung (Stu ID: 23133030)
        """
        
        messagebox.showinfo("About 8-Puzzle Solver", about_text)

    def show_help(self):
        """Show help documentation."""
        help_window = tk.Toplevel(self.root)
        help_window.title("8-Puzzle Solver Documentation")
        help_window.geometry("700x500")
        
        help_frame = ttk.Frame(help_window, padding="10")
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        help_scroll = ttk.Scrollbar(help_frame)
        help_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        help_text = tk.Text(help_frame, wrap=tk.WORD, yscrollcommand=help_scroll.set)
        help_text.pack(fill=tk.BOTH, expand=True)
        help_scroll.config(command=help_text.yview)
        
        help_content = """# 8-Puzzle Solver Documentation

    ## Overview
    The 8-puzzle consists of 8 numbered tiles and one blank space. 
    The goal is to rearrange the tiles to match the goal configuration.

    ## Algorithms
    This application implements multiple algorithms:

    ### Uninformed Search
    - BFS (Breadth-First Search): Explores all nodes at a level before moving to the next level.
    - DFS (Depth-First Search): Explores as deep as possible before backtracking.
    - UCS (Uniform Cost Search): Prioritizes paths with the lowest cost.

    ### Informed Search
    - Greedy Search: Chooses the path that appears closest to the goal.
    - A* Search: Combines path cost and heuristic for efficient search.
    - IDA*: Memory-efficient A* search.

    ### Local Search
    - Hill Climbing: Moves in the direction of increasing value.
    - Simulated Annealing: Accepts some worse moves to escape local optima.
    - Genetic Algorithm: Uses evolution-inspired operators to find solutions.
    - Beam Search: Keeps a limited number of best nodes at each level.

    ### CSP Algorithms
    - Backtracking: Basic depth-first search with constraint checking.
    - intelligent Backtracking: Uses heuristics to improve backtracking.
    - Min-Conflicts: Iteratively resolves conflicts.
    

    ## Heuristics
    - Manhattan Distance: Sum of horizontal and vertical distances.
    - Misplaced Tiles: Count of tiles not in their goal position.

    ## Using the Application
    1. Select an algorithm from the list on the right.
    2. Click "Solve" to run the selected algorithm.
    3. Use the animation controls to view the solution steps.
    4. Click "Reset" to return to the initial state.

    ## Tips
    - A* with Manhattan distance is typically the most efficient algorithm.
    - For very complex puzzles, IDA* might work better with limited memory.
    - Local search algorithms may get stuck in local optima.
    """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)
    
        
    def create_ribbon(self):
        """
        Creates a ribbon interface at the top of the application.
        The ribbon contains tabs with various controls for different functionality.
        """
        # Create ribbon frame
        ribbon_frame = ttk.Frame(self.root)
        ribbon_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Create notebook for ribbon tabs
        ribbon = ttk.Notebook(ribbon_frame)
        ribbon.pack(fill=tk.X)
        
        # Create ribbon tabs
        file_tab = ttk.Frame(ribbon)
        puzzle_tab = ttk.Frame(ribbon)
        tools_tab = ttk.Frame(ribbon)
        help_tab = ttk.Frame(ribbon)
        
        ribbon.add(file_tab, text="File")
        ribbon.add(puzzle_tab, text="Puzzle")
        ribbon.add(tools_tab, text="Tools")
        ribbon.add(help_tab, text="Help")
        
        # === File Tab ===
        file_frame = ttk.Frame(file_tab, padding="5")
        file_frame.pack(fill=tk.X)
        
        # Reset button
        reset_frame = ttk.Frame(file_frame)
        reset_frame.pack(side=tk.LEFT, padx=10, pady=5)
        reset_btn = ttk.Button(reset_frame, text="Reset", command=self.reset_puzzle)
        reset_btn.pack()
        ttk.Label(reset_frame, text="Reset").pack()
        
        # Save/Load config buttons
        config_frame = ttk.Frame(file_frame)
        config_frame.pack(side=tk.LEFT, padx=10, pady=5)
        save_btn = ttk.Button(config_frame, text="Save Config", command=self.save_configuration)
        save_btn.pack(side=tk.LEFT, padx=2)
        load_btn = ttk.Button(config_frame, text="Load Config", command=self.load_configuration)
        load_btn.pack(side=tk.LEFT, padx=2)
        
        # Exit button
        exit_frame = ttk.Frame(file_frame)
        exit_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        exit_btn = ttk.Button(exit_frame, text="Exit", command=self.root.quit)
        exit_btn.pack()
        ttk.Label(exit_frame, text="Exit").pack()
        
        # === Puzzle Tab ===
        puzzle_frame = ttk.Frame(puzzle_tab, padding="5")
        puzzle_frame.pack(fill=tk.X)
        
        # Random puzzle generator
        random_frame = ttk.Frame(puzzle_frame)
        random_frame.pack(side=tk.LEFT, padx=10, pady=5)
        random_btn = ttk.Button(random_frame, text="Random", command=self.generate_random_puzzle)
        random_btn.pack()
        ttk.Label(random_frame, text="Random").pack()
        
        # Difficulty selector
        difficulty_frame = ttk.Frame(puzzle_frame)
        difficulty_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(difficulty_frame, text="Difficulty:").pack(side=tk.LEFT)
        self.difficulty_var = tk.StringVar(value="medium")
        ttk.Combobox(difficulty_frame, textvariable=self.difficulty_var, 
                    values=["easy", "medium", "hard"]).pack(side=tk.LEFT, padx=5)
        
        # Goal state toggle
        goal_frame = ttk.Frame(puzzle_frame)
        goal_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        self.show_goal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(goal_frame, text="Show Goal State", 
                        variable=self.show_goal_var, 
                        command=self.toggle_goal_display).pack()
        
        # === Tools Tab ===
        tools_frame = ttk.Frame(tools_tab, padding="5")
        tools_frame.pack(fill=tk.X)
        
        # Algorithm benchmarks
        benchmark_frame = ttk.Frame(tools_frame)
        benchmark_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Button(benchmark_frame, text="Benchmark", 
                command=self.run_benchmark).pack()
        ttk.Label(benchmark_frame, text="Compare").pack()
        
        # Animation speed control
        speed_frame = ttk.Frame(tools_frame)
        speed_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(speed_frame, text="Animation Speed:").pack()
        ttk.Scale(speed_frame, from_=100, to=1000, variable=self.speed_var, 
                command=self.update_speed, length=150).pack()
        
        # === Help Tab ===
        help_frame = ttk.Frame(help_tab, padding="5")
        help_frame.pack(fill=tk.X)
        
        # About button
        about_frame = ttk.Frame(help_frame)
        about_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Button(about_frame, text="About", 
                command=self.show_about).pack()
        ttk.Label(about_frame, text="About").pack()
        
        # Documentation button
        docs_frame = ttk.Frame(help_frame)
        docs_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Button(docs_frame, text="Help", 
                command=self.show_help).pack()
        ttk.Label(docs_frame, text="Documentation").pack()
    def create_visualization_controls(self):
        """Tạo các điều khiển cho tab visualization."""
        controls_frame = ttk.Frame(self.viz_container)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Tạo notebook cho các loại biểu đồ khác nhau
        viz_notebook = ttk.Notebook(self.viz_container)
        viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab cho đồ thị heuristic
        self.heuristic_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(self.heuristic_tab, text="Heuristic Analysis")
        
        # Tab cho đồ thị cấu trúc tìm kiếm
        self.tree_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(self.tree_tab, text="Search Structure")
        
        # Tab cho đồ thị hiệu suất
        self.performance_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(self.performance_tab, text="Performance")
        
        # Nút để vẽ đồ thị
        ttk.Button(controls_frame, text="Generate Visualizations", 
                  command=self.generate_all_plots).pack(side=tk.LEFT, padx=5)
        
    def generate_all_plots(self):
        """Tạo tất cả các loại đồ thị."""
        if not self.solution_path:
            messagebox.showinfo("No Solution", "Please solve the puzzle first.")
            return
        
        # Vẽ đồ thị phân tích heuristic
        self.plot_heuristic_analysis()
        
        # Vẽ đồ thị cấu trúc tìm kiếm
        self.plot_search_structure()
        
        # Vẽ đồ thị hiệu suất
        self.plot_performance_metrics()
    
    def plot_heuristic_analysis(self):
        """Vẽ đồ thị phân tích heuristic."""
        # Xóa nội dung cũ
        for widget in self.heuristic_tab.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Chuẩn bị dữ liệu
        steps = list(range(len(self.solution_path)))
        manhattan_values = []
        misplaced_values = []
        
        for state in self.solution_path:
            manhattan_values.append(self.solver.get_manhattan_distance(state))
            misplaced_values.append(self.solver.get_misplaced_tiles(state))
        
        # Vẽ biểu đồ
        ax.plot(steps, manhattan_values, 'b-', label='Manhattan Distance')
        ax.plot(steps, misplaced_values, 'r-', label='Misplaced Tiles')
        ax.set_title('Heuristic Values Along Solution Path')
        ax.set_xlabel('Step')
        ax.set_ylabel('Heuristic Value')
        ax.legend()
        
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.heuristic_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_search_structure(self):
        """Vẽ đồ thị cấu trúc tìm kiếm."""
        # Xóa nội dung cũ
        for widget in self.tree_tab.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Tạo cây cấu trúc tìm kiếm đơn giản
        x = list(range(len(self.solution_path)))
        y = [0] * len(self.solution_path)
        
        ax.plot(x, y, 'bo-', label='Solution Path')
        
        # Thêm các nút mô phỏng nút đã mở rộng
        expanded_nodes = []
        for i in range(len(self.solution_path)-1):
            num_expanded = min(i+2, 5)
            for j in range(num_expanded):
                if j == 0:  # Nút trên đường đi
                    continue
                expanded_nodes.append((i, j*0.5))
                ax.plot([i, i+1], [j*0.5, 0], 'r--')
        
        ax.scatter([p[0] for p in expanded_nodes], [p[1] for p in expanded_nodes], 
                  color='r', label='Expanded Nodes', alpha=0.5)
        
        ax.set_title('Search Tree Structure')
        ax.set_xlabel('Depth')
        ax.set_yticks([])  # Ẩn ticks của trục y
        ax.legend()
        
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.tree_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_performance_metrics(self):
        """Vẽ đồ thị phân tích hiệu suất."""
        # Xóa nội dung cũ
        for widget in self.performance_tab.winfo_children():
            widget.destroy()
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Mô phỏng dữ liệu hiệu suất (thay bằng dữ liệu thực khi có sẵn)
        algorithm = self.algorithm_var.get()
        nodes_per_depth = [1, 3, 9, 27, 30, 20, 10, 5, 2, 1]
        depths = list(range(len(nodes_per_depth)))
        
        # Vẽ biểu đồ
        ax.bar(depths, nodes_per_depth, color='green', alpha=0.7)
        ax.set_title(f'Nodes Expanded per Depth Level - {algorithm}')
        ax.set_xlabel('Depth')
        ax.set_ylabel('Nodes Expanded')
        
        plt.tight_layout()
        
        # Hiển thị plot trong container
        canvas = FigureCanvasTkAgg(fig, master=self.performance_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def create_widgets(self):
        """
        Tạo và sắp xếp tất cả các thành phần GUI.

        Phương thức này tạo giao diện hiển thị puzzle, bảng thông tin, điều khiển
        lựa chọn thuật toán và điều khiển hoạt ảnh.
        """
        
        self.create_ribbon()
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for puzzle display
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for controls
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas with scrollbar for controls
        canvas = tk.Canvas(right_panel, borderwidth=0)
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        controls_frame = ttk.Frame(canvas, padding="10")
        
        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Create window in canvas for controls frame
        canvas_window = canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        
        # Update canvas scroll region when controls frame changes size
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        # Update canvas width when window resizes
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
            
        controls_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
            
        
        # Puzzle display
        puzzle_frame = ttk.LabelFrame(left_panel, text="Puzzle", padding="10")
        puzzle_frame.pack(fill=tk.BOTH, expand=True)
        
        
        # Create the puzzle grid
        self.cell_frames = []
        for i in range(3):
            row_frames = []
            for j in range(3):
                cell = ttk.Frame(puzzle_frame, width=80, height=80, relief="raised", borderwidth=2)
                cell.grid(row=i, column=j, padx=5, pady=5)
                cell.pack_propagate(False)  # Prevent the frame from shrinking
                
                value = self.initial_state[i][j]
                if value == 0:
                    label = ttk.Label(cell, text="", font=("Arial", 24, "bold"))
                else:
                    label = ttk.Label(cell, text=str(value), font=("Arial", 24, "bold"))
                label.pack(expand=True)
                
                row_frames.append((cell, label))
            self.cell_frames.append(row_frames)
        
        # Add after the puzzle grid in create_widgets

        # Create statistics and visualization tabs
        info_tabs = ttk.Notebook(left_panel)
        info_tabs.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Thêm vào phương thức create_widgets sau dòng tạo các tab:
        # Info_tabs.add(algo_details_frame, text="Algorithm Details")
        
        # Visualization tab
        viz_frame = ttk.Frame(info_tabs, padding="10")
        info_tabs.add(viz_frame, text="Visualization")
        
        # Tạo container để chứa đồ thị
        self.viz_container = ttk.Frame(viz_frame)
        self.viz_container.pack(fill=tk.BOTH, expand=True)
        # Information display tab
        info_frame = ttk.Frame(info_tabs, padding="10")
        info_tabs.add(info_frame, text="Information")

        self.info_label = ttk.Label(info_frame, text="Ready to solve", font=("Arial", 10))
        self.info_label.pack(fill=tk.X)
                
        self.steps_label = ttk.Label(info_frame, text="")
        self.steps_label.pack(fill=tk.X)

        # Statistics tab
        stats_frame = ttk.Frame(info_tabs, padding="10")
        info_tabs.add(stats_frame, text="Statistics")

        # Create statistics display
        self.nodes_per_second_label = ttk.Label(stats_frame, text="Nodes per second: N/A")
        self.nodes_per_second_label.pack(fill=tk.X)

        self.memory_usage_label = ttk.Label(stats_frame, text="Memory usage: N/A")
        self.memory_usage_label.pack(fill=tk.X)

        self.solution_quality_label = ttk.Label(stats_frame, text="Solution optimality: N/A")
        self.solution_quality_label.pack(fill=tk.X)

        # Algorithm details tab
        algo_details_frame = ttk.Frame(info_tabs, padding="10")
        info_tabs.add(algo_details_frame, text="Algorithm Details")

        # Create scrollable text widget for algorithm details
        algo_scroll = ttk.Scrollbar(algo_details_frame)
        algo_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.algo_details_text = tk.Text(algo_details_frame, height=8, wrap=tk.WORD, 
                                        yscrollcommand=algo_scroll.set)
        self.algo_details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        algo_scroll.config(command=self.algo_details_text.yview)

        # Set initial algorithm details
        csp_info = """CSP Model for 8-Puzzle:
        - Variables (X): 9 positions on the board
        - Domain (D): Numbers 0-8 that can be placed in each position
        - Constraints (C): Each number appears exactly once, and the configuration matches the goal state

        CSP algorithms use constraint propagation and search to find valid solutions."""

        self.algo_details_text.insert(tk.END, csp_info)
        self.algo_details_text.config(state=tk.DISABLED)
        
        # Information display
        info_frame = ttk.LabelFrame(left_panel, text="Information", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        
        self.steps_label = ttk.Label(info_frame, text="")
        self.steps_label.pack(fill=tk.X)
        
        # Controls - Sửa lại phần này
        controls_label_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_label_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        # Algorithm selection
        ttk.Label(controls_frame, text="Algorithm:").pack(anchor=tk.W, pady=(0, 5))
        
        self.algorithm_var = tk.StringVar(value="bfs")
        algorithms = [
            ("Breadth-First Search", "bfs"),
            ("Depth-First Search", "dfs"),
            ("Uniform Cost Search", "ucs"),
            ("Greedy Search (Manhattan)", "greedy_manhattan"),
            #("Greedy Search (Misplaced)", "greedy_misplaced"),
            ("A* Search (Manhattan)", "astar_manhattan"),
            #("A* Search (Misplaced)", "astar_misplaced"),
            ("Iterative Deepening", "id"),
            ("IDA* (Manhattan)", "idastar_manhattan"),
            #("IDA* (Misplaced)", "idastar_misplaced"),
            ("Simple Hill Climbing (Manhattan)", "shc_manhattan"),
            #("Simple Hill Climbing (Misplaced)", "shc_misplaced"),
            ("Steepest-Ascent Hill Climbing (Manhattan)", "sahc_manhattan"),
            #("Steepest-Ascent Hill Climbing (Misplaced)", "sahc_misplaced"),
            ("Random-Restart Hill Climbing (Manhattan)", "rrhc_manhattan"),
            #("Random-Restart Hill Climbing (Misplaced)", "rrhc_misplaced"),
            ("Simulated Annealing (Manhattan)", "sa_manhattan"),
            #("Simulated Annealing (Misplaced)", "sa_misplaced"),
            ("Beam Search (Manhattan)", "beam_manhattan"),
            #("Beam Search (Misplaced)", "beam_misplaced"),
            ("Genetic Algorithm (Manhattan)", "ga_manhattan"),
            #("Genetic Algorithm (Misplaced)", "ga_misplaced"),
            ("CSP - Backtracking", "csp_backtracking"),
            ("CSP - Intelligent Backtracking", "csp_intelligent_backtracking"),
            ("CSP - Min-Conflicts", "csp_min_conflicts"),
            ("Q-Learning (Reinforcement Learning)", "q_learning"),
            ("AND-OR Graph Search", "and_or_graph_search"),
            ("Belief State Search", "belief_state_search")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(controls_frame, text=text, variable=self.algorithm_var, value=value).pack(anchor=tk.W)
        
        # Solve button
        ttk.Button(controls_frame, text="Solve", command=self.solve_puzzle).pack(fill=tk.X, pady=10)
        
        # Animation controls
        animation_frame = ttk.Frame(controls_frame)
        animation_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(animation_frame, text="<<", command=self.step_back).pack(side=tk.LEFT)
        ttk.Button(animation_frame, text="Play", command=self.play_solution).pack(side=tk.LEFT, padx=5)
        ttk.Button(animation_frame, text=">>", command=self.step_forward).pack(side=tk.LEFT)
        
        # Speed control
        speed_frame = ttk.Frame(controls_frame)
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=500)
        ttk.Scale(speed_frame, from_=100, to=1000, variable=self.speed_var, 
                  command=self.update_speed).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Reset button
        ttk.Button(controls_frame, text="Reset", command=self.reset_puzzle).pack(fill=tk.X, pady=10)
        # Mouse wheel scrolling
        # Mouse wheel scrolling - sửa lại để xử lý đúng
        def _on_mousewheel(event):
            # Kiểm tra xem con trỏ có đang ở trong canvas không
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # Đảm bảo frame có kích thước đủ để hiển thị tất cả nội dung
        controls_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Thêm các điều khiển cho tab visualization
        self.create_visualization_controls()
        
    def update_puzzle_display(self, board):
        """
        Cập nhật lưới puzzle hiển thị để hiện thị cấu hình bảng đã cho.

        Tham số:
            board (list): Ma trận 3x3 biểu diễn cấu hình bảng để hiển thị
        """
        for i in range(3):
            for j in range(3):
                _, label = self.cell_frames[i][j]
                value = board[i][j]
                if value == 0:
                    label.config(text="")
                elif value == -1:  # For CSP: unassigned cell
                    label.config(text="?")  # or any symbol for unassigned
                else:
                    label.config(text=str(value))
    
    def solve_puzzle(self):
        """
        Giải puzzle sử dụng thuật toán đã chọn.

        Phương thức này chạy thuật toán tìm kiếm đã chọn và xử lý kết quả,
        hiển thị thông tin giải pháp và chuẩn bị cho hoạt ảnh.
        """
        algorithm = self.algorithm_var.get()
        # Special handling for CSP algorithms
        if algorithm.startswith("csp_"):
            # Clear the puzzle display to show an empty board
            empty_board = [[0 for _ in range(3)] for _ in range(3)]
            self.update_puzzle_display(empty_board)
            self.root.update()  # Force update the display
            
            # Call the appropriate CSP algorithm
            if algorithm == "csp_backtracking":
                result = self.solver.backtracking_search(callback=self.csp_callback)
                algorithm_name = "CSP - Backtracking"
            elif algorithm == "csp_intelligent_backtracking":
                result = self.solver.intelligent_backtracking(callback=self.csp_callback)
                algorithm_name = "CSP - Intelligent Backtracking"
            elif algorithm == "csp_min_conflicts":
                result = self.solver.min_conflicts(callback=self.csp_callback)
                algorithm_name = "CSP - Min-Conflicts"
        else:
            # Normal handling for non-CSP algorithms
            initial = PuzzleState(self.initial_state)

        
        # Select the algorithm
        if algorithm == "bfs":
            result = self.solver.bfs(initial)
            algorithm_name = "Breadth-First Search"
        elif algorithm == "dfs":
            result = self.solver.dfs(initial)
            algorithm_name = "Depth-First Search"
        elif algorithm == "ucs":
            result = self.solver.ucs(initial)
            algorithm_name = "Uniform Cost Search"
        elif algorithm == "greedy_manhattan":
            result = self.solver.greedy_search(initial, heuristic='manhattan')
            algorithm_name = "Greedy Search (Manhattan)"
        elif algorithm == "greedy_misplaced":
            result = self.solver.greedy_search(initial, heuristic='misplaced')
            algorithm_name = "Greedy Search (Misplaced)"
        elif algorithm == "id":
            result = self.solver.iterative_deepening(initial)
            algorithm_name = "Iterative Deepening"
        elif algorithm == "astar_manhattan":
            result = self.solver.a_star_search(initial, heuristic='manhattan')
            algorithm_name = "A* Search (Manhattan)"
        elif algorithm == "astar_misplaced":
            result = self.solver.a_star_search(initial, heuristic='misplaced')
            algorithm_name = "A* Search (Misplaced)"
        elif algorithm == "idastar_manhattan":
            result = self.solver.ida_star_search(initial, heuristic='manhattan')
            algorithm_name = "IDA* (Manhattan)"
        elif algorithm == "idastar_misplaced":
            result = self.solver.ida_star_search(initial, heuristic='misplaced')
            algorithm_name = "IDA* (Misplaced)"
        # Thêm vào phương thức solve_puzzle:
        elif algorithm == "shc_manhattan":
            result = self.solver.simple_hill_climbing(initial, heuristic='manhattan')
            algorithm_name = "Simple Hill Climbing (Manhattan)"
        elif algorithm == "shc_misplaced":
            result = self.solver.simple_hill_climbing(initial, heuristic='misplaced')
            algorithm_name = "Simple Hill Climbing (Misplaced)"
        elif algorithm == "sahc_manhattan":
            result = self.solver.steepest_ascent_hill_climbing(initial, heuristic='manhattan')
            algorithm_name = "Steepest-Ascent Hill Climbing (Manhattan)"
        elif algorithm == "sahc_misplaced":
            result = self.solver.steepest_ascent_hill_climbing(initial, heuristic='misplaced')
            algorithm_name = "Steepest-Ascent Hill Climbing (Misplaced)"
        elif algorithm == "rrhc_manhattan":
            result = self.solver.random_restart_hill_climbing(initial, heuristic='manhattan')
            algorithm_name = "Random-Restart Hill Climbing (Manhattan)"
        elif algorithm == "rrhc_misplaced":
            result = self.solver.random_restart_hill_climbing(initial, heuristic='misplaced')
            algorithm_name = "Random-Restart Hill Climbing (Misplaced)"
        elif algorithm == "sa_manhattan":
            result = self.solver.simulated_annealing(initial, heuristic='manhattan')
            algorithm_name = "Simulated Annealing (Manhattan)"
        elif algorithm == "sa_misplaced":
            result = self.solver.simulated_annealing(initial, heuristic='misplaced')
            algorithm_name = "Simulated Annealing (Misplaced)"
        elif algorithm == "beam_manhattan":
            result = self.solver.beam_search(initial, heuristic='manhattan', beam_width=3)
            algorithm_name = "Beam Search (Manhattan)"
        elif algorithm == "beam_misplaced":
            result = self.solver.beam_search(initial, heuristic='misplaced', beam_width=3)
            algorithm_name = "Beam Search (Misplaced)"
        elif algorithm == "ga_manhattan":
            result = self.solver.genetic_algorithm(initial, heuristic='manhattan')
            algorithm_name = "Genetic Algorithm (Manhattan)"
        elif algorithm == "ga_misplaced":
            result = self.solver.genetic_algorithm(initial, heuristic='misplaced')
            algorithm_name = "Genetic Algorithm (Misplaced)"
        # elif algorithm == "csp_backtracking":
        #     result = self.solver.backtracking_search()
        #     algorithm_name = "CSP - Backtracking"
        # elif algorithm == "csp_intelligent_backtracking":
        #     result = self.solver.intelligent_backtracking()
        #     algorithm_name = "CSP - Intelligent Backtracking"
        # elif algorithm == "csp_min_conflicts":
        #     result = self.solver.min_conflicts()
        #     algorithm_name = "CSP - Min-Conflicts"
        elif algorithm == "q_learning":
            result = self.solver.q_learning(initial, heuristic='manhattan', 
                                            episodes=500, max_steps=100)
            algorithm_name = "Q-Learning (Reinforcement Learning)"
        elif algorithm == "and_or_graph_search":
            result = self.solver.and_or_graph_search(initial, slip_probability=0.2, max_depth=10)
            algorithm_name = "AND-OR Graph Search"
        elif algorithm == "belief_state_search":
            result = self.solver.belief_state_search(initial, max_depth=10)
            algorithm_name = "Belief State Search"
        # Process result
        if result["path"]:
            self.solution_path = result["path"]
            self.current_step = 0
            self.update_puzzle_display(self.solution_path[0].board)
            
            # Update info
            steps = len(self.solution_path) - 1
            time_taken = result["time"]
            nodes = result["nodes_expanded"]
            
            info_text = f"Algorithm: {algorithm_name}\n"
            info_text += f"Solution found in {steps} steps\n"
            info_text += f"Time taken: {time_taken:.6f} seconds\n"
            info_text += f"Nodes expanded: {nodes}\n"
            info_text += f"Max queue size: {result['max_queue_size']}"
            
            self.info_label.config(text=info_text)
            self.steps_label.config(text=f"Step: 0/{steps}")
            
            messagebox.showinfo("Solution Found", f"Found solution in {steps} steps!")
        else:
            if "partial_path" in result:
                self.solution_path = result["partial_path"]
                self.current_step = 0
                self.update_puzzle_display(self.solution_path[0].board)
                
                steps = len(self.solution_path) - 1
                time_taken = result["time"]
                nodes = result["nodes_expanded"]
                
                info_text = f"Algorithm: {algorithm_name}\n"
                info_text += f"No solution found - got stuck at local optimum.\n"
                info_text += f"Steps taken: {steps}\n"
                
                # Add final_h only if it exists in the result
                if "final_h" in result:
                    info_text += f"Final heuristic: {result['final_h']}\n"
                    
                info_text += f"Time taken: {time_taken:.6f} seconds\n"
                info_text += f"Nodes expanded: {nodes}"
                
                self.info_label.config(text=info_text)
                self.steps_label.config(text=f"Step: 0/{steps}")
                
                messagebox.showinfo("No Solution", "Got stuck at local optimum!")
            else:
                self.info_label.config(text=f"No solution found with {algorithm_name}\nNodes expanded: {result['nodes_expanded']}")
                messagebox.showinfo("No Solution", "Could not find a solution!")
        # After getting the results, also update the statistics tab
        if result["path"] or "partial_path" in result:
            # Calculate nodes per second
            if result["time"] > 0:
                nodes_per_second = result["nodes_expanded"] / result["time"]
            else:
                nodes_per_second = 0
                
            self.nodes_per_second_label.config(text=f"Nodes per second: {nodes_per_second:.2f}")
            
            # Update algorithm details based on selected algorithm
            self.algo_details_text.config(state=tk.NORMAL)
            self.algo_details_text.delete(1.0, tk.END)
            
            if "csp" in algorithm:
                csp_info = """CSP Model for 8-Puzzle:
    - Variables (X): 9 positions on the board
    - Domain (D): Numbers 0-8 that can be placed in each position
    - Constraints (C): Each number appears exactly once, and the configuration matches the goal state

    CSP Algorithms:
    - Backtracking: Basic depth-first search with constraint checking
    - Intelligent Backtracking: Uses MRV (Minimum Remaining Values) and 
    LCV (Least Constraining Value) heuristics
    - Min-Conflicts: Iteratively resolves conflicts by minimizing constraint violations"""
                
                # Add algorithm-specific details
                if algorithm == "csp_backtracking":
                    csp_info += "\n\nBacktracking builds a solution incrementally, abandoning paths when constraints are violated."
                elif algorithm == "csp_intelligent_backtracking":
                    csp_info += "\n\nIntelligent Backtracking efficiently prunes the search space using variable/value ordering heuristics."
                elif algorithm == "csp_min_conflicts":
                    csp_info += "\n\nMin-Conflicts works by repeatedly selecting a conflicted variable and assigning it a value causing the minimum number of conflicts."
                
                self.algo_details_text.insert(tk.END, csp_info)
            else:
                # Add descriptions for other algorithm types
                if "astar" in algorithm or "idastar" in algorithm:
                    algo_info = """A* Search:
    A best-first search algorithm that combines path cost (g) and heuristic estimate (h).
    f(n) = g(n) + h(n)

    IDA* (Iterative Deepening A*):
    Memory-efficient variant that runs a series of depth-limited searches."""
                elif "hill" in algorithm:
                    algo_info = """Hill Climbing:
    A local search algorithm that continually moves in the direction of increasing value.
    - Simple: Takes the first better neighbor
    - Steepest-Ascent: Evaluates all neighbors and selects the best one
    - Random-Restart: Multiple attempts from random starting points"""
                else:
                    algo_info = f"Details for {algorithm_name}"
                    
                self.algo_details_text.insert(tk.END, algo_info)
            
            self.algo_details_text.config(state=tk.DISABLED)
            
    def step_forward(self):
        """
        Di chuyển một bước về phía trước trong hoạt ảnh giải pháp.

        Cập nhật hiển thị để hiện thị trạng thái tiếp theo trong đường đi giải pháp.
        """
        if self.solution_path and self.current_step < len(self.solution_path) - 1:
            self.current_step += 1
            current_state = self.solution_path[self.current_step]
            self.update_puzzle_display(current_state.board)
            self.steps_label.config(text=f"Step: {self.current_step}/{len(self.solution_path) - 1}")
    
    def step_back(self):
        """
        Di chuyển một bước về phía sau trong hoạt ảnh giải pháp.

        Cập nhật hiển thị để hiện thị trạng thái trước đó trong đường đi giải pháp.
        """
        if self.solution_path and self.current_step > 0:
            self.current_step -= 1
            current_state = self.solution_path[self.current_step]
            self.update_puzzle_display(current_state.board)
            self.steps_label.config(text=f"Step: {self.current_step}/{len(self.solution_path) - 1}")
    
    def play_solution(self):
        """
        Tự động phát hoạt ảnh giải pháp.

        Tự gọi lại chính nó đệ quy cho đến khi đạt đến cuối đường đi giải pháp.
        """
        if self.solution_path and self.current_step < len(self.solution_path) - 1:
            self.step_forward()
            self.animation_id = self.root.after(self.animation_speed, self.play_solution)
        else:
            # Animation complete
            pass
    
    def update_speed(self, event=None):
        """
        Cập nhật tốc độ hoạt ảnh dựa trên giá trị thanh trượt.

        Tham số:
            event: Sự kiện kích hoạt hàm này (không sử dụng nhưng Tkinter yêu cầu)
        """
        self.animation_speed = self.speed_var.get()
    
    def reset_puzzle(self):
        """
        Đặt lại puzzle về trạng thái ban đầu.

        Dừng mọi hoạt ảnh đang chạy và xóa thông tin giải pháp.
        """
        if hasattr(self, 'animation_id'):
            self.root.after_cancel(self.animation_id)
        
        self.solution_path = None
        self.current_step = 0
        self.update_puzzle_display(self.initial_state)
        self.info_label.config(text="Ready to solve")
        self.steps_label.config(text="")
 
def main():
    """
    Điểm vào chính của ứng dụng.

    Tạo cửa sổ gốc Tkinter và khởi tạo giao diện puzzle.
    """
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
