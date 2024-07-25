import heapq
from collections import defaultdict

class Graph:
    def _init_(self):
        self.nodes = set()
        self.edges = defaultdict(list)

    def add_edge(self, start, end, weight):
        self.nodes.add(start)
        self.nodes.add(end)
        self.edges[start].append((end, weight))
        self.edges[end].append((start, weight))

class TrainScheduler:
    def _init_(self, graph):
        self.graph = graph

    def dijkstra(self, start):
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0
        queue = [(0, start)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph.edges[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))

        return distances

    def route_train(self, start, end, track_condition_monitor):
        distances = self.dijkstra(start)
        optimal_route = []
        current_node = end
        while current_node != start:
            for neighbor, weight in self.graph.edges[current_node]:
                if distances[current_node] - weight == distances[neighbor]:
                    if track_condition_monitor.get_track_condition((current_node, neighbor)) != "Under Maintenance":
                        optimal_route.append((current_node, neighbor))
                        current_node = neighbor
                        break
        optimal_route.reverse()
        return optimal_route


class TrackMaintenanceScheduler:
    def _init_(self):
        self.tasks = []

    def schedule_maintenance(self, task, priority):
        heapq.heappush(self.tasks, (priority, task))

    def execute_next_maintenance(self):
        if self.tasks:
            return heapq.heappop(self.tasks)
        else:
            return None

class TrainPerformanceMonitor:
    def _init_(self):
        self.performance_data = defaultdict(list)

    def update_performance(self, train_id, speed, delay):
        self.performance_data[train_id].append({'speed': speed, 'delay': delay})

    def get_average_speed(self, train_id):
        speeds = [data['speed'] for data in self.performance_data[train_id]]
        return sum(speeds) / len(speeds) if speeds else 0

    def get_average_delay(self, train_id):
        delays = [data['delay'] for data in self.performance_data[train_id]]
        return sum(delays) / len(delays) if delays else 0

class RealTimeTracking:
    def _init_(self, graph):
        self.graph = graph
        self.train_positions = {}

    def update_train_position(self, train_id, position):
        self.train_positions[train_id] = position

    def get_train_position(self, train_id):
        return self.train_positions.get(train_id, None)

class TrackConditionMonitor:
    def _init_(self):
        self.track_conditions = defaultdict(lambda: 'Good')

    def update_track_condition(self, track, condition):
        self.track_conditions[track] = condition

    def get_track_condition(self, track):
        condition = self.track_conditions[track]
        if condition != 'Good':
            return condition
        else:
            return None  # Return None if condition is 'Good'

class RailwaySystem:
    def _init_(self):
        self.railway_network = Graph()
        self.train_scheduler = TrainScheduler(self.railway_network)
        self.track_maintenance_scheduler = TrackMaintenanceScheduler()
        self.train_performance_monitor = TrainPerformanceMonitor()
        self.real_time_tracking = RealTimeTracking(self.railway_network)
        self.track_condition_monitor = TrackConditionMonitor()

    def add_track(self, start, end, weight):
        self.railway_network.add_edge(start, end, weight)

    def route_train(self, start, end):
        return self.train_scheduler.route_train(start, end, self.track_condition_monitor)

    def schedule_maintenance(self, task=None, priority=None):
        if task is None:
            task = input("Enter maintenance task: ")
        if priority is None:
            priority = int(input("Enter priority for the task: "))
        self.track_maintenance_scheduler.schedule_maintenance(task, priority)

    def execute_next_maintenance(self):
        next_task = self.track_maintenance_scheduler.execute_next_maintenance()
        if next_task:
            print("Next maintenance task:", next_task)
        else:
            print("No pending maintenance tasks.")

    def update_train_performance(self, train_id=None, speed=None, delay=None):
        if train_id is None:
            train_id = input("Enter train ID: ")
        if speed is None:
            speed = float(input("Enter train speed: "))
        if delay is None:
            delay = float(input("Enter train delay: "))
        self.train_performance_monitor.update_performance(train_id, speed, delay)

    def get_average_speed(self,train_id=None):
        if train_id is None:
            train_id = input("Enter train ID: ")
        return self.train_performance_monitor.get_average_speed(train_id)

    def get_average_delay(self, train_id=None):
        if train_id is None:
            train_id = input("Enter train ID: ")
        return self.train_performance_monitor.get_average_delay(train_id)

    def update_train_position(self, train_id=None, position=None):
        if train_id is None:
            train_id = input("Enter train ID: ")
        if position is None:
            position = input("Enter train position: ")
        self.real_time_tracking.update_train_position(train_id, position)

    def get_train_position(self, train_id=None):
        if train_id is None:
            train_id = input("Enter train ID: ")
        position = self.real_time_tracking.get_train_position(train_id)
        if position:
            print(f"Position of train {train_id}: {position}.")
        else:
            print(f"Train {train_id} position not found.")

    def update_track_condition(self, start, end, condition):
        track = (start, end)
        self.track_condition_monitor.update_track_condition(track,condition)

    def check_track_condition(self, start, end):
        track = (start, end)
        condition = self.track_condition_monitor.get_track_condition(track)
        if condition:
            print(f"Track condition between {start} and {end}: {condition}.")
        else:
            print(f"No specific track condition found between {start} and {end}.")

    def shortest_distance_between_stations(self, start=None, end=None):
        if start is None:
            start = input("Enter start station: ")
        if end is None:
            end = input("Enter end station: ")
        distances = self.train_scheduler.dijkstra(start)
        shortest_distance = distances[end]
        print(f"Shortest distance between stations {start} and {end}: {shortest_distance}.")
        return shortest_distance

    def findPlatform(self, arr, dep):
        arr2 = []
        for i in range(len(arr)):
            arr2.append([arr[i], dep[i]])
        arr2.sort()
        p = []
        count = 1
        heapq.heappush(p, arr2[0][1])
        for i in range(1, len(arr)):
            if p[0] >= arr2[i][0]:
                count += 1
            else:
                heapq.heappop(p)
            heapq.heappush(p, arr2[i][1])
        print(f"Minimum number of platforms required: {count}.")
        return count

if _name_ == "_main_":
    # Creating a railway system
    railway_system = RailwaySystem()

    # Adding tracks to the railway network
    while True:
        start_station = input("Enter start station (or 'done' to finish adding tracks): ")
        if start_station.lower() == "done":
            break
        end_station = input("Enter end station: ")
        weight = int(input("Enter distance between stations: "))
        railway_system.add_track(start_station, end_station, weight)

    # Updating track conditions
    while True:
        start = input("Enter start station for track maintenance (or 'done' to finish): ")
        if start.lower() == "done":
            break
        end = input("Enter end station for track maintenance: ")
        condition = input("Enter track condition: ")
        railway_system.update_track_condition(start, end, condition)

    # Routing a train
    start_station = input("Enter start station for the train: ")
    end_station = input("Enter end station for the train: ")
    optimal_route = railway_system.route_train(start_station, end_station)
    print("Optimal route:", optimal_route)

    # Scheduling maintenance tasks
    while True:
        task = input("Enter maintenance task (or 'done' to finish): ")
        if task.lower() == "done":
            break
        priority = int(input("Enter priority for the task: "))
        railway_system.schedule_maintenance(task, priority)

    # Executing next maintenance task
    railway_system.execute_next_maintenance()

    # Updating train performance data
    train_id = input("Enter train ID: ")
    speed = float(input("Enter train speed: "))
    delay = float(input("Enter train delay: "))
    railway_system.update_train_performance(train_id, speed, delay)

    # Retrieving average speed and delay for a train
    train_id = input("Enter train ID to get average speed and delay: ")
    avg_speed = railway_system.get_average_speed(train_id)
    avg_delay = railway_system.get_average_delay(train_id)
    print("Average speed:", avg_speed)
    print("Average delay:", avg_delay)

    # Updating train position
    train_id = input("Enter train ID: ")
    position = input("Enter train position: ")
    railway_system.update_train_position(train_id, position)
    railway_system.get_train_position(train_id)

    # Finding the shortest distance between stations
    start_station = input("Enter start station: ")
    end_station = input("Enter end station: ")
    shortest_distance = railway_system.shortest_distance_between_stations(start_station, end_station)
    print("Shortest distance:", shortest_distance)

    # Finding the minimum number of platforms required
    arr = []
    dep = []
    while True:
        arrival = input("Enter arrival time (or 'done' to finish): ")
        if arrival.lower() == "done":
            break
        departure = input("Enter departure time: ")
        arr.append(int(arrival))
        dep.append(int(departure))
    railway_system.findPlatform(arr, dep)