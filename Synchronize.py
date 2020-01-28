from SensorSynchronization.Sensor import *
from typing import List
# comment the following if you don't want debug mode
import matplotlib.pyplot as plt
import networkx as nx
import datetime
from uuid import uuid4
import os



class Synchronize(object):
    def __init__(self, sensors, max_skew=None, minimum_intervals=None, accuracy_similarity=1):
        self.sensors = sensors
        self.max_skew = max_skew
        self.minimum_intervals = minimum_intervals
        self.groups = []
        self.final_sets = []
        self.accuracy_similarity = accuracy_similarity
        self.debug = False # by setting true, a graph for a different state will be created

        if self.debug:
            if not os.path.exists('graphs'):
                os.makedirs('graphs')


    def synchronize(self) -> List[SensorGroup]:
        self.create_groups()  # creating nodes(group of sensors) s.t. two sensors are in the same group if they have signal from the same set of TXs
        self.create_graph()  # creating graph and its edges(new nodes of intersection of two nodes might be created)
        self.extract_distinct_intruders()  # analyze the graph and finding smallest set(most probably with size one) of TXs
        return self.get_intruders_list()  # return TXs sets and their sensors list along with Observation Vector(as a numpy vector)

    def create_groups(self):
        for sensor in self.sensors:
            if sensor.number_of_one < self.minimum_intervals:
                continue
            existed = False
            for group in self.groups:
                _, common_number_one, _ = group.sensor.compare(sensor, self.max_skew)
                #if self.is_the_same(group.sensor, sensor, common_number_pulse):
                if self.is_the_same(group.sensor, sensor, common_number_one) == 0:
                    group.append_sensor(sensor.id)
                    existed = True
                    break
            if not existed:
                self.groups.append(SensorGroup(id=len(self.groups), signal=sensor.zero_one_signal,
                                               sensors_list=[sensor.id], accuracy_parameter=self.accuracy_similarity))

    def path_exist(self, u, v):  # check recursively if there is a path between u and v
        if not self.graph[u]:
            return False
        if v in self.graph[u]:
            return True
        for uu in self.graph[u]:
            if self.path_exist(uu, v):
                return True
        return False

    def merge_nodes(self, u, v):
        # merging two nodes u and v s.t. nodes v will be deleted
        self.graph[u] = self.graph[u] | self.graph[v]  # adding edges(v, x), i.e. ongoing edges from v, to node u
        for x in self.graph:  # changing edges (x, v), i.e. incoming edges to v, to (x, u)
            if v in x:
                x.remove(v)
                x.add(u)
        self.graph[v] = None  # removing all edges from v
        v.is_exist = False  # delete node v

    def remove_extra_edges(self, indices): # remove edges till every node is connected to just its parent not grandparents
        for i in indices:
            neighbors = set(self.graph[i])
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 == neighbor2 or neighbor2 not in self.graph[i]:  # already deleted
                        continue
                    if self.path_exist(neighbor1, neighbor2):
                        # if node i is connected to both neighbor1 and neighbor2 and there os path from neighbor1 and
                        # neighbor2(neighbor1 is a child of neighbor2), edge (i, neighbor2) can be deleted safely.
                        self.graph[i].remove(neighbor2)

    def new_node(self, u, v, common_signal):  # create a new node of intersection of u, v
        new_group_id = len(self.graph)
        self.graph.append({u, v})  # add edges from new node to u and v
        return SensorGroup(id=new_group_id, signal=common_signal)

    def merge_new_groups(self, new_bag: set()):  # find relation between new added groups and previous one;
                                                    # delete the new one if already exist
        if not new_bag or len(new_bag) == 0:
            return []
        new_new_bag = []
        candid_bag = [group.id for group in self.groups if group.is_exist]
        old_bag = [x for x in candid_bag if x not in new_bag]

        # be_kept = True
        for ng_id in new_bag:
            ng = self.find_set(ng_id)
            if not ng or not ng.is_exist:
                continue
            be_kept = True
            for g_id in old_bag:
                g = self.find_set(g_id)
                if not g or not g.is_exist or self.path_exist(ng_id, g_id) or self.path_exist(g_id, ng_id):
                    continue
                [_, common_number_one, _] = ng.sensor.compare(g.sensor, self.max_skew)
                                                 #  two vertices i and j are the same
                relation = self.is_the_same(g.sensor, ng.sensor, common_number_one)
                if relation == 0:
                    self.merge_nodes(g, ng)
                    be_kept = False
                    break
                if relation == 1 and self.graph[ng.id]:  # i is a subset of j
                    self.graph[g.id].add(ng.id)
                elif relation == -1 and self.graph[ng.id]:  # j is a subset of i
                    self.graph[ng.id].add(g.id)
            if be_kept:
                new_new_bag.append(ng_id)
        return new_new_bag

    def merge_groups_after_subtraction(self):
        candid_nodes = self.zero_indegree_nodes()
        for g1_idx, g1 in enumerate(candid_nodes):
            for g2_idx, g2 in enumerate(candid_nodes):
                if g2_idx <= g1_idx:
                    continue
                g1_node = self.find_set(g1)
                g2_node = self.find_set(g2)
                [_, comm_number_one, _] = g1_node.sensor.compare(g2_node.sensor, self.max_skew)
                if self.is_the_same(g1_node.sensor, g2_node.sensor, comm_number_one) == 0:
                    self.merge_nodes(g1_node, g2_node)

    def is_the_same(self, sensor1, sensor2, common_number_one):
        if max(sensor1.number_of_one, sensor2.number_of_one) - common_number_one \
                < self.minimum_intervals <= common_number_one:
            if common_number_one >= self.accuracy_similarity * max(sensor1.number_of_one, sensor2.number_of_one):
                return 0  # sensor1 and sensor2 have the same signal

        sensor1_similarity = 1 - abs(common_number_one- sensor1.number_of_one) / max(common_number_one,
                                                                                     sensor1.number_of_one)
        if sensor1_similarity >= self.accuracy_similarity:
            return 1  # sensor1's signal is a subset of sensor2's signal

        sensor2_similarity = 1 - abs(common_number_one - sensor2.number_of_one) / max(common_number_one,
                                                                                      sensor2.number_of_one)
        if sensor2_similarity >= self.accuracy_similarity:
            return -1  # sensor2's signal is a subset of sensor1's signal

        return 2  # sensor's and sensor2's signal are not from the same set of TXs

    def plot_graph(self):
        plt.figure()
        graph_adj_mat = np.zeros((len(self.graph), len(self.graph)), dtype=int)
        for u, u_list in enumerate(self.graph):
            for v in u_list:
                graph_adj_mat[u][v] = 1
        rows, cols = np.where(graph_adj_mat == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.ion()
        plt.show()

    def zero_indegree_nodes(self):
        candid_nodes = {group.id for group in self.groups if group.is_exist}
        for u_list in self.graph:
            if u_list:
                candid_nodes -= u_list
        return candid_nodes

    def neighbours(self, u):
        return np.where(self.graph[u] == 1)[0]

    def all_path(self, u):
        if not self.graph[u]:
            return [[u]]
        paths = []
        for v in self.graph[u]:
            for n_path in self.all_path(v):
                paths.append([u] + n_path)
        return paths

    @staticmethod
    def all_nodes_in_paths(paths):
        nodes = []
        for path in paths:
            nodes += path
        return list(set(nodes))

    def update_graph_removed_nodes(self):
        for g in self.groups:
            if g.sensor.number_of_one == 0 and g.is_exist:  # < self.minimum_intervals:
                self.graph[g.id] = None
                g.is_exist = False

    def find_set(self, u):
        for g in self.groups:
            if g.id == u:
                return g
        return None

    def create_graph(self):
        groups_size = len(self.groups)
        self.graph = [set() for _ in range(groups_size)]  # graph representation as adjacency list
        old_bag = range(groups_size)
        # new_bag = []
        # new_groups = []
        while len(old_bag) != 0:     # This while intersection between sets till there is no more intersection
            new_bag = []
            for i_idx, i in enumerate(old_bag):
                if not self.find_set(i).is_exist:
                    continue
                for j_idx, j in enumerate(old_bag):
                    if not self.find_set(j).is_exist or i_idx >= j_idx:  #  i_idx >= j_idx is because we already finds out the realation in previous loop
                        continue
                    if self.path_exist(i, j) or self.path_exist(j, i):
                        continue  # if there is already relationship, continue
                    [common_signal, common_number_one, _] = self.groups[i].sensor.compare(self.groups[j].sensor,
                                                                                          self.max_skew)
                    relation = self.is_the_same(self.groups[i].sensor, self.groups[j].sensor, common_number_one)
                    if relation == 0:  # two vertices i and j are the same
                        self.merge_nodes(self.find_set(i), self.find_set(j))
                        # new_bag.remove(j)
                    elif relation == 1:  # i is a subset of j
                        self.graph[i].add(j)
                    elif relation == -1:  # j is a subset of i
                        self.graph[j].add(i)
                    elif common_number_one >= self.minimum_intervals:  # add a new vertex of intersection of two vertices i and j
                        existed_group = self.group_exist(common_signal)  # if intersection already exist
                        if existed_group:
                            self.graph[existed_group.id] |= {i, j}
                        else:  # if not, create a new one
                            g = self.new_node(i, j, common_signal)
                            self.groups.append(g)  # add new group to the list
                            new_bag.append(g.id)
            new_bag = self.merge_new_groups(set(new_bag))
            if self.debug:
                self.plot_graph()
                plt.savefig("graphs/" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))
            self.remove_extra_edges(old_bag)
            old_bag = new_bag
        if self.debug:
            self.plot_graph()
            plt.savefig("graphs/" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))

    def group_exist(self, new_signal):  # check if the new signal does not match any existed group
        new_group = SensorGroup(id=None, signal=new_signal, accuracy_parameter=self.accuracy_similarity)
        for g in self.groups:
            if not g.is_exist:
                continue
            [_, common_number_one, _] = new_group.sensor.compare(g.sensor, self.max_skew)
            if self.is_the_same(new_group.sensor, g.sensor, common_number_one) == 0:
                return g
        return None

    def extract_distinct_intruders(self):
        zero_in_nodes = self.zero_indegree_nodes()
        while zero_in_nodes:
            for zero_node in zero_in_nodes:
                paths = self.all_path(zero_node)
                # print ("paths for node #"+ str(zero_node) + str(paths))
                group = self.find_set(zero_node)
                final_set = SensorGroup(id=len(self.final_sets), signal=group.sensor.signal)
                for path in paths:
                    for path_node in path:
                        group_node = self.find_set(path_node)
                        if group_node:
                            final_set.append_group(group_node.sensors_list)
                self.final_sets.append(final_set)
                nodes = self.all_nodes_in_paths(paths)
                self.graph[zero_node] = None
                group.is_exist = False
                for node in nodes:
                    if node == zero_node:
                        continue
                    node_group = self.find_set(node)
                    node_group.sensor.subtract_signal(group.sensor.zero_one_signal, skew=self.max_skew)
            self.update_graph_removed_nodes()
            self.merge_groups_after_subtraction()
            zero_in_nodes = self.zero_indegree_nodes()
            if self.debug and zero_in_nodes:
                self.plot_graph()
                plt.savefig("graphs/" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))

        #  Creating groups based on some singular vertices(groups) that are remained
        for g in self.groups:
            if g.is_exist:
                self.final_sets.append(SensorGroup(id=len(self.final_sets),
                                                   signal=g.sensor.signal, sensors_list=g.sensors_list))
        # ********* Get Observation value(maximum) for each sensor in final sets
        number_of_sensors = len(self.sensors)
        for final_set in self.final_sets:
            observation_vector = np.zeros(len(final_set.sensors_list), dtype=float)
            for observed_idx, sensor_idx in enumerate(final_set.sensors_list):
                observation_vector[observed_idx] = self.sensors[sensor_idx].observed_value(final_set.sensor.signal,
                                                                                           self.max_skew)
            final_set.observation_vector = observation_vector

    def get_intruders_list(self):
        return self.final_sets

