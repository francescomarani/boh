import numpy as np
import pybullet as p
import math
import random
import heapq
import time

class PRMPlanner:
    def __init__(self, bounds, robot_radius=0.35):
        """
        :param bounds: [x_min, x_max, y_min, y_max]
        :param robot_radius: Safety radius for collision checking
        """
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.radius = robot_radius
        self.graph = {} 
        self.samples = [] 
        
        # Create a "Ghost" sphere for collision checking
        # Visual (Red, semi-transparent)
        #visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=[1, 0, 0, 0.5])
        # Collision (Actual size)
        #collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
        
        #self.ghost_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape)
        
        # Hide ghost initially
        #p.resetBasePositionAndOrientation(self.ghost_id, [0, 0, -10], [0,0,0,1])

    def is_free(self, pos, debug=False):
        """
        Checks if a position is free using AABB (Axis Aligned Bounding Box).
        This is immune to 'Static Object Sleeping' bugs.
        """
        x, y = pos
        r = self.radius
        
        # Define a Box around the robot position
        # Z: Start at 0.1 (above floor) and go up to 1.5 (head height)
        aabb_min = [x - r, y - r, 0.1]
        aabb_max = [x + r, y + r, 1.5]
        
        # Ask PyBullet: "What objects are inside this box?"
        overlaps = p.getOverlappingObjects(aabb_min, aabb_max)
        
        if overlaps:
            for body_id, link_idx in overlaps:
                # Filter out the robot itself (if it wasn't hidden correctly)
                # and ignore the Floor (usually ID 0 or 1, but we check names to be safe)
                body_name = p.getBodyInfo(body_id)[1].decode('utf-8')
                
                # Ignore the floor plane
                if "floor" in body_name.lower() or "plane" in body_name.lower():
                    continue
                
                if debug:
                    print(f"Blocked at {pos} by: {body_name} (ID {body_id})")
                
                return False # Found a valid obstacle
        
        return True # Box is empty

    def build_roadmap(self, n_samples=1000, connect_dist=5.0, k_neighbors=10):
        """ Uses K-Nearest Neighbors (KNN) via NumPy. """
        print(f"PRM: Sampling {n_samples} nodes...")
        t_start = time.time()
        self.samples = []
        self.graph = {}
        
        # 1. Random Sampling
        attempts = 0
        while len(self.samples) < n_samples and attempts < n_samples * 5:
            attempts += 1
            rand_x = random.uniform(self.x_min, self.x_max)
            rand_y = random.uniform(self.y_min, self.y_max)
            if self.is_free((rand_x, rand_y)):
                self.samples.append((rand_x, rand_y))
                self.graph[len(self.samples)-1] = []

        print(f"PRM: Connecting nodes (KNN={k_neighbors})...")
        # 2. Vectorized KNN Connection
        if len(self.samples) > 1:
            sample_array = np.array(self.samples)
            for i, point in enumerate(sample_array):
                dists = np.linalg.norm(sample_array - point, axis=1)
                closest_indices = np.argsort(dists)[1:k_neighbors+1]
                
                for j in closest_indices:
                    if dists[j] > connect_dist: continue
                    if j <= i: continue
                    
                    if self.check_edge(self.samples[i], self.samples[j]):
                        self.graph[i].append(j)
                        self.graph[j].append(i)

        print(f"PRM: Built {len(self.samples)} nodes in {time.time() - t_start:.2f}s.")

        # Hide ghost
        #p.resetBasePositionAndOrientation(self.ghost_id, [0, 0, -10], [0,0,0,1])
        #print(f"PRM: Roadmap built with {len(self.samples)} nodes and {edges} edges in {time.time() - t_start:.2f}s.")

    def check_edge(self, p1, p2, step_size=0.1):
        """Checks if straight line between p1 and p2 is clear."""
        dist = math.dist(p1, p2)
        steps = int(dist / step_size)
        if steps == 0: return True
        
        dx = (p2[0] - p1[0]) / steps
        dy = (p2[1] - p1[1]) / steps
        
        cx, cy = p1[0], p1[1]
        for _ in range(steps):
            cx += dx
            cy += dy
            if not self.is_free((cx, cy)):
                return False
        return True

    def find_path(self, start, goal):
        """A* Search."""
        # Check Validity
        if not self.is_free(start):
            print("Start is obstructed!")
            return []
        if not self.is_free(goal):
            print("Goal is obstructed!")
            return []

        # Add Start/Goal to graph temporarily
        start_idx = len(self.samples)
        self.samples.append(start)
        self.graph[start_idx] = []
        
        goal_idx = len(self.samples)
        self.samples.append(goal)
        self.graph[goal_idx] = []
        
        # Connect Start/Goal to nearest valid neighbors
        sample_array = np.array(self.samples[:-2])
        if len(sample_array) > 0:
            for pt, idx in [(start, start_idx), (goal, goal_idx)]:
                dists = np.linalg.norm(sample_array - np.array(pt), axis=1)
                # Try connecting to the 15 closest nodes
                nearest = np.argsort(dists)[:15] 
                
                for j in nearest:
                    if dists[j] < 5.0 and self.check_edge(pt, self.samples[j]):
                        self.graph[idx].append(j)
                        self.graph[j].append(idx)

        # A* Algorithm
        queue = [(0, start_idx, [])] # Cost, Current Node, Path
        visited = set()
        
        while queue:
            cost, curr, path = heapq.heappop(queue)
            
            if curr in visited: continue
            visited.add(curr)
            
            path = path + [self.samples[curr]]
            
            if curr == goal_idx:
                print(f"Path found! Length: {len(path)}")
                return path 
            
            for neighbor in self.graph.get(curr, []):
                if neighbor not in visited:
                    # Cost = Current Cost + Distance + Heuristic
                    dist = math.dist(self.samples[curr], self.samples[neighbor])
                    heuristic = math.dist(self.samples[neighbor], goal)
                    heapq.heappush(queue, (cost + dist + heuristic, neighbor, path))
                    
        print("PRM: No path found!")
        return []
    def draw_roadmap(self):
        print(f"Drawing full roadmap with {len(self.samples)} nodes...")
        for i, neighbors in self.graph.items():
            p1 = self.samples[i]
            for j in neighbors:
                # Only draw if i < j to avoid drawing every line twice
                if i < j:
                    p2 = self.samples[j]
                    # Blue lines, thin
                    p.addUserDebugLine([p1[0], p1[1], 0.1], 
                                       [p2[0], p2[1], 0.1], 
                                       [0, 0, 1], lineWidth=1, lifeTime=0)
    def draw_path(self, path):
        if not path: return
        p.removeAllUserDebugItems()
        for i in range(len(path) - 1):
            p.addUserDebugLine([path[i][0], path[i][1], 0.1], 
                               [path[i+1][0], path[i+1][1], 0.1], 
                               [0, 1, 0], lineWidth=3)