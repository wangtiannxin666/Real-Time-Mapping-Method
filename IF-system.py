import tkinter as tk
from tkinter import Frame, Text
from tkinter import simpledialog
import customtkinter as ctk
import requests
import json
from neo4j import GraphDatabase
import time
import math
import random
import arcpy
from matplotlib import pyplot as plt
import numpy as np

API_KEY = "OvYc55VAZLH1E9K3MMDfYYQU"
SECRET_KEY = "IkYVHPNTg3qeKljl7RXIignAGPsMGGeh"

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "wtx20010123666"))

class QASystemGUI:
    def __init__(self, master):
        self.master = master
        master.title("IF-ERNIE-Bot4")
        master.geometry("800x600")
        master.configure(bg="#1e1e2e")

        self.chat_frame = Frame(master, bg="#1e1e2e")
        self.chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.font = ("Helvetica", 12)
        self.chat_text = Text(self.chat_frame, bg="#1e1e2e", fg="white", wrap=tk.WORD, padx=10, pady=10, font=self.font)
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_text.config(state=tk.DISABLED)

        self.scrollbar = ctk.CTkScrollbar(self.chat_frame, command=self.chat_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text.configure(yscrollcommand=self.scrollbar.set)

        self.input_frame = tk.Frame(master, bg="#1e1e2e")
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.input_field = ctk.CTkEntry(self.input_frame, font=self.font)
        self.input_field.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.input_field.bind("<Return>", self.on_enter)

        self.send_button = ctk.CTkButton(self.input_frame, text="发送", command=self.send_message, font=self.font)
        self.send_button.pack(side=tk.RIGHT)

        self.access_token = self.get_access_token()
        self.url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={self.access_token}"
        self.conversation_context = None

        self.question_file_map = {
            "什么是常见的室内火灾隐患": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\1.txt",
            "在不同类型的建筑中，有哪些特殊的防火要求": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\3.txt",
            "发生室内火灾时，正确的逃生步骤是什么": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\2.txt",
            "火灾发生时，如何正确使用灭火器": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\4.txt",
            "为什么电器设备是室内火灾的主要原因之一，如何安全使用电器": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\5.txt",
            "在火灾现场，如何识别和避开危险区域": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\6.txt",
            "发生火灾时，为什么不应该使用电梯": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\7.txt",
            "什么是防火分区，它是如何限制火灾蔓延的": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\8.txt",
            "在火灾逃生中，老弱病残人员有哪些特殊考虑": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\9.txt",
            "定期进行火灾演习有什么重要性，应该如何组织": "C:\\Users\\admin\\Desktop\\论文\\知识图谱问题回复\\10.txt"
        }

        self.chat_text.tag_configure("user", justify="right", foreground="#DCF8C6",font=('Arial', 12, 'bold'))
        self.chat_text.tag_configure("bot", justify="left", foreground="#E2E2E2",font=('Arial', 11))

        self.use_customtkinter = True  # 设置为 True 使用 customtkinter，False 使用 simpledialog
    def on_enter(self, event):
        self.send_message()

    def handle_evacuation_map_request(self):
        map_type = self.get_user_input("请选择疏散路线图类型 (1-无栅格疏散路线图, 2-栅格疏散路线图): ")
        if map_type not in ['1', '2']:
            return "无效的选择，请重新输入。"

        path_color = self.get_user_input("请输入路线颜色: ") or 'green'
        door_color = self.get_user_input("请输入门颜色: ") or 'purple'
        obstacle_color = self.get_user_input("请输入障碍物颜色: ") or 'gray'

        response = generate_evacuation_map(map_type, path_color, door_color, obstacle_color)
        return response

    def get_user_input(self, prompt):
        if self.use_customtkinter:
            return ctk.CTkInputDialog(text=prompt, title="Input").get_input()
        else:
            return simpledialog.askstring("Input", prompt, parent=self.master)

    def handle_evacuation_map_request(self):
        map_type = self.get_user_input("请选择疏散路线图类型 (1-无栅格疏散路线图, 2-栅格疏散路线图): ")
        if map_type not in ['1', '2']:
            return "无效的选择，请重新输入。"

        path_color = self.get_user_input("请输入路线颜色: ") or 'green'
        door_color = self.get_user_input("请输入门颜色: ") or 'purple'
        obstacle_color = self.get_user_input("请输入障碍物颜色: ") or 'gray'

        response = generate_evacuation_map(map_type, path_color, door_color, obstacle_color)
        return response

    def send_message(self):
        user_input = self.input_field.get()
        if user_input:
            self.display_message(f"User: {user_input}\n", "user")
            self.input_field.delete(0, tk.END)

            self.master.update()  # 立即更新界面，显示用户输入

            response = ""  # 初始化 response 变量

            if "疏散路线图" in user_input:
                response = self.handle_evacuation_map_request()
            elif user_input in self.question_file_map:
                response = self.read_from_txt_file(self.question_file_map[user_input])
            else:
                query = "MATCH (n) WHERE n.stmc = $user_input RETURN n LIMIT 1"
                result = self.execute_query(query, {"user_input": user_input})

                if result:
                    response = self.generate_response_from_kg(user_input)
                else:
                    payload = self.build_payload(user_input)
                    headers = {'Content-Type': 'application/json'}
                    response_json = requests.post(self.url, headers=headers, data=json.dumps(payload)).json()
                    response = response_json['result']
                    self.conversation_context = response_json.get("conversation_context")

            if response:  # 只有当 response 不为空时才显示
                self.display_message(f"IF-Bot: {response}\n\n", "bot")
            else:
                self.display_message("IF-Bot: 抱歉，我无法处理这个请求。\n\n", "bot")

    def display_message(self, message, sender):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, message, sender)
        self.chat_text.insert(tk.END, "\n", sender)  # 添加额外的换行来增加段落间距
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return requests.post(url, params=params).json().get("access_token")

    def build_payload(self, user_input):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
        if self.conversation_context:
            payload["conversation_context"] = self.conversation_context
        return payload

    def execute_query(self, query, parameters=None):
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def generate_response_from_kg(self, user_input):
        query = """
        MATCH (n)-[r]->(m)
        WHERE (n.stmc CONTAINS $user_input OR m.stmc CONTAINS $user_input)
          AND NOT r.relation STARTS WITH '被'
        RETURN DISTINCT n, r.relation AS relation, m
        ORDER BY toInteger(n.ID), toInteger(m.ID)
        """
        result = self.execute_query(query, {"user_input": user_input})

        if result:
            response = "\n"
            for record in result:
                n = record["n"]
                relation = record["relation"]
                m = record["m"]
                response += f"- {n['stmc']} {relation} {m['stmc']}\n"
            return response
        else:
            return None

    def read_from_txt_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        time.sleep(3)  # 引入3秒的延迟
        return content

# 以下是从main-new4.py中整合的代码

def node_interval(cur_node, next_node):
    parts1 = cur_node.split('-')
    parts2 = next_node.split('-')
    if abs(int(parts1[1]) - int(parts2[1])) >= 2:
        return True
    return False

def plot_hexagon(data, ax, is_fire=False):
    geometry = data[0]
    for part in geometry:
        x_coords = np.array([point.X for point in part])
        y_coords = np.array([point.Y for point in part])
        if is_fire == False:
            ax.fill(x_coords, y_coords, color='green', alpha=0.5)
        else:
            ax.fill(x_coords, y_coords, color='#FF2D2D', alpha=0.7)

def letter_add_str(s):
    result = ""
    if chr(ord(s[-1])) != 'Z':
        result = s[:-1] + chr(ord(s[-1]) + 1)
    else:
        result = s + 'A'
    return result

def letter_subtract_str(s):
    result = ""
    if s == 'A':
        return ""
    if chr(ord(s[-1])) != 'A':
        result = s[:-1] + chr(ord(s[-1]) - 1)
    else:
        result = s[:-2] + chr(ord(s[-2]) - 1) + 'A'
    return result

def find_next_grid_id(s: str):
    ids = []
    parts = s.split('-')
    if parts[0] != 'A':
        num1 = int(parts[1])
        str1 = letter_subtract_str(parts[0]) + "-" + str(num1)
        ids.append(str1)
        num1 = int(parts[1]) - 1
        str1 = letter_subtract_str(parts[0]) + "-" + str(num1)
        ids.append(str1)

    num1 = int(parts[1]) + 1
    str1 = parts[0] + "-" + str(num1)
    ids.append(str1)
    num1 = int(parts[1]) - 1
    str1 = parts[0] + "-" + str(num1)
    ids.append(str1)
    num1 = int(parts[1])
    str1 = letter_add_str(parts[0]) + "-" + str(num1)
    ids.append(str1)
    num1 = int(parts[1]) - 1
    str1 = letter_add_str(parts[0]) + "-" + str(num1)
    ids.append(str1)

    return ids

class AntColonyOptimization:
    def __init__(self, num_ants, num_nodes, hexagon_dict, fire_start_node, fire_speed=0.35, alpha=1, beta=2,
                 evaporation_rate=0.5, Hexagonal_diameter=1.7):
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.hexagon_dict: dict = hexagon_dict
        self.fire: list = None
        self.fire_speed: float = fire_speed
        self.Hexagonal_diameter = Hexagonal_diameter
        self.fire_start_node = fire_start_node

    def fire_init(self, time):
        fire_radius = time * self.fire_speed
        fire_start_point = self.fire_start_node[0].centroid

        self.fire = []
        for key, node in self.hexagon_dict.items():
            node_center = node[0].centroid
            distance = self.distance_between_points(fire_start_point, node_center.X, node_center.Y)

            if distance <= fire_radius:
                node_list = list(node)
                node_list[3] = 0
                self.hexagon_dict[key] = tuple(node_list)
                self.fire.append(self.hexagon_dict[key])

        print("fire init finish")

    def distance_between_points(self, point1, x, y):
        return math.sqrt((x - point1.X) ** 2 + (y - point1.Y) ** 2)

    def check_distance(self, point, point_list, threshold=10):
        for point_l in point_list:
            dist = self.distance_between_points(point, point_l.centroid.X, point_l.centroid.Y)
            if dist < threshold:
                return True
        return False

    def choose_next_node(self, current_node, path: list):
        current_time = int(time.time())
        random.seed(current_time)
        neighbor_weights = []

        for neighbor_key in find_next_grid_id(current_node[1]):
            neighbor = self.hexagon_dict.get(neighbor_key)
            if neighbor and neighbor[2] != 0:
                pheromone_weight = neighbor[2] ** self.alpha
                heuristic_weight = 1 * random.random()
                total_weight = pheromone_weight * heuristic_weight

                if neighbor[3] == 0:
                    continue
                if node_interval(current_node[1], neighbor[1]):
                    continue

                neighbor_weights.append((neighbor, total_weight))

        if not neighbor_weights:
            return None

        cumulative_weight = 0
        res = None
        for neighbor, weight in neighbor_weights:
            if weight >= cumulative_weight:
                cumulative_weight = weight
                res = neighbor

        if res:
            return res
        return None

    def update_pheromone(self, paths, path_costs):
        delta_pheromone = 1 / np.array(path_costs)

        for key in self.hexagon_dict:
            hexagon_info_list = list(self.hexagon_dict[key])
            hexagon_info_list[2] *= self.evaporation_rate
            self.hexagon_dict[key] = tuple(hexagon_info_list)

        for i, path in enumerate(paths):
            for node in path:
                nc = list(node)
                nc[2] += delta_pheromone[i]
                node = tuple(nc)
                self.hexagon_dict[node[1]] = node

    def ant_path_search(self, start_node, end_nodes: list, max_iterations):
        best_path = None
        best_path_cost = float('inf')

        for _ in range(max_iterations):
            paths = []
            path_costs = []

            for _ in range(self.num_ants):
                current_node = start_node
                path = [current_node]
                path_cost = 1

                while current_node is not None and not self.check_distance(current_node[0].centroid, end_nodes, 2):
                    next_node = self.choose_next_node(current_node, path)

                    if next_node is None:
                        break

                    if next_node in self.fire:
                        path_cost = float('inf')
                        break
                    else:
                        path_cost += next_node[2]
                        path.append(next_node)
                        current_node = next_node

                paths.append(path)
                path_costs.append(path_cost)
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost

            self.update_pheromone(paths, path_costs)

        return best_path

    def generate_path(self, start_node, end_nodes: list):
        return self.ant_path_search(start_node, end_nodes, 10)


def choose_next_node(self, current_node, path: list):
    current_time = int(time.time())
    random.seed(current_time)
    neighbor_weights = []

    for neighbor_key in find_next_grid_id(current_node[1]):
        neighbor = self.hexagon_dict.get(neighbor_key)
        if neighbor and neighbor[2] != 0:
            pheromone_weight = neighbor[2] ** self.alpha
            heuristic_weight = 1 * random.random()
            total_weight = pheromone_weight * heuristic_weight

            if neighbor[3] == 0:
                continue
            if node_interval(current_node[1], neighbor[1]):
                continue

            neighbor_weights.append((neighbor, total_weight))

    if not neighbor_weights:
        return None

    cumulative_weight = 0
    res = None
    for neighbor, weight in neighbor_weights:
        if weight >= cumulative_weight:
            cumulative_weight = weight
            res = neighbor

    if res:
        return res
    return None


# 设置工作空间
arcpy.env.workspace = r"E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb"


def plot_path(time, path_color='green', door_color='purple', obstacle_color='gray'):
    grid_dict = {}
    fig, ax = plt.subplots()

    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\grid_hexagons',
                               ["SHAPE@", "GRID_ID", "pheromone", "passable"]) as cursor:
        for row in cursor:
            grid_dict[row[1]] = row

    geometry_list = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\door', "SHAPE@") as cursor:
        for row in cursor:
            geometry = row[0]
            geometry_list.append(geometry)

    x_coords = [point.X for geometry in geometry_list for point in geometry]
    y_coords = [point.Y for geometry in geometry_list for point in geometry]

    plt.scatter(x_coords, y_coords, color=door_color, marker='o', s=200)

    line_coords = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\BJ', ["SHAPE@"]) as cursor:
        for row in cursor:
            line = row[0]
            for part in line:
                for pnt in part:
                    line_coords.append((pnt.X, pnt.Y))

    line_coords_array = np.array(line_coords)
    ax.plot(*line_coords_array.T, color='black', zorder=2)

    ants = AntColonyOptimization(10, grid_dict.__len__(), grid_dict, grid_dict.get("AP-44"))
    ants.fire_init(time)

    for node in ants.fire:
        geometry = node[0]
        for part in geometry:
            x_coords = np.array([point.X for point in part])
            y_coords = np.array([point.Y for point in part])
            ax.fill(x_coords, y_coords, color='red', alpha=0.7)

    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\obstacle',
                               ["SHAPE@"]) as cursor:
        for row in cursor:
            geometry = row[0]
            for part in geometry:
                x_coords = np.array([point.X for point in part])
                y_coords = np.array([point.Y for point in part])
                ax.fill(x_coords, y_coords, color=obstacle_color, alpha=0.8)

    line_coords = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\path',
                               ["SHAPE@"]) as cursor:
        for row in cursor:
            line = row[0]
            for part in line:
                for pnt in part:
                    line_coords.append((pnt.X, pnt.Y))

    line_coords_array = np.array(line_coords)
    ax.plot(*line_coords_array.T, color=path_color, linewidth=2, zorder=3)

    plt.show()


def plot_path_new(time, path_color='green', door_color='purple', obstacle_color='gray'):
    grid_dict = {}
    fig, ax = plt.subplots()

    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\grid_hexagons',
                               ["SHAPE@", "GRID_ID", "pheromone", "passable"]) as cursor:
        for row in cursor:
            grid_dict[row[1]] = row
            for part in row[0]:
                x_coords = np.array([point.X for point in part])
                y_coords = np.array([point.Y for point in part])
                if row[3] > 0:
                    ax.plot(x_coords, y_coords, color='grey', alpha=0.5)
                else:
                    ax.fill(x_coords, y_coords, color='black', alpha=0.5)

    geometry_list = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\door', "SHAPE@") as cursor:
        for row in cursor:
            geometry = row[0]
            geometry_list.append(geometry)

    x_coords = [point.X for geometry in geometry_list for point in geometry]
    y_coords = [point.Y for geometry in geometry_list for point in geometry]

    plt.scatter(x_coords, y_coords, color=door_color, marker='o', s=200)

    line_coords = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\BJ', ["SHAPE@"]) as cursor:
        for row in cursor:
            line = row[0]
            for part in line:
                for pnt in part:
                    line_coords.append((pnt.X, pnt.Y))

    line_coords_array = np.array(line_coords)
    ax.plot(*line_coords_array.T, color='black', zorder=2)

    ants = AntColonyOptimization(10, grid_dict.__len__(), grid_dict, grid_dict.get("AP-44"))
    ants.fire_init(time)

    for node in ants.fire:
        geometry = node[0]
        for part in geometry:
            x_coords = np.array([point.X for point in part])
            y_coords = np.array([point.Y for point in part])
            ax.fill(x_coords, y_coords, color='red', alpha=0.7)

    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\obstacle',
                               ["SHAPE@"]) as cursor:
        for row in cursor:
            geometry = row[0]
            for part in geometry:
                x_coords = np.array([point.X for point in part])
                y_coords = np.array([point.Y for point in part])
                ax.fill(x_coords, y_coords, color=obstacle_color, alpha=0.8)
    line_coords = []
    with arcpy.da.SearchCursor(r'E:\技术2\aaa\MyProject-000\MyProject-000\MyProject-000.gdb\path',
                               ["SHAPE@"]) as cursor:
        for row in cursor:
            line = row[0]
            for part in line:
                for pnt in part:
                    line_coords.append((pnt.X, pnt.Y))

    line_coords_array = np.array(line_coords)
    ax.plot(*line_coords_array.T, color=path_color, linewidth=2, zorder=3)

    plt.show()


def generate_evacuation_map(map_type, path_color, door_color, obstacle_color):
    try:
        if map_type == '1':
            plot_path(20, path_color, door_color, obstacle_color)
            return "已生成无栅格疏散路线图。"
        elif map_type == '2':
            plot_path_new(20, path_color, door_color, obstacle_color)
            return "已生成栅格疏散路线图。"
        else:
            return "无效的选择，请重新输入。"
    except Exception as e:
        return f"生成疏散路线图时发生错误: {str(e)}"


if __name__ == '__main__':
    root = tk.Tk()
    app = QASystemGUI(root)
    root.mainloop()