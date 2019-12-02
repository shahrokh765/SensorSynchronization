from synchronization.Sensor import *
import matplotlib.pyplot as plt
import networkx as nx
import datetime
from uuid import uuid4


class Synchronize(object):
    def __init__(self, sensors, max_skew=None, minimum_intervals=None, accuracy_similarity=1):
        self.sensors = sensors
        self.max_skew = max_skew
        self.minimum_intervals = minimum_intervals
        self.groups = []
        self.final_sets = []
        self.accuracy_similarity = accuracy_similarity
        self.debug = False

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
                                               group_list=[sensor.id], accuracy_parameter=self.accuracy_similarity))

    def path_exist(self, u, v):
        if self.graph[u][v] == 1:
            return True
        bag = np.where(self.graph[u] == 1)[0]
        for uu in bag:
            return self.path_exist(uu, v)
        return False

    def merge_nodes(self, u, v):
        self.graph[u.id] = np.logical_or(self.graph[u.id], self.graph[v.id]) # merge edges from these two nodes
        self.graph[:, u.id] = np.logical_or(self.graph[:, u.id], self.graph[:, v.id]) #merge edges to these nodes
        self.graph[u.id][u.id] = 0
        self.graph[v.id] = 0 * self.graph[v.id]
        self.graph[:, v.id] = 0 * self.graph[:, v.id]
        v.is_exist = False

    def remove_extra_edges(self, indices): # remove edges till every node is connected to just its parent not grandparents
        for i in indices:
            #neighbors = np.where(self.graph[i] == 1)[0]
            neighbors = self.neighbours(i)
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 == neighbor2:
                        continue
                    #print ("node #" + str(i) + " neighbour "+str(neighbor1) + " neighbour " + str(neighbor2))
                    if self.path_exist(neighbor1, neighbor2):
                        self.graph[i][neighbor2] = 0

    def new_node(self, u, v, common_signal): # create a new node of intersection of u, v
        new_group_id = self.graph.shape[0]
        # id_available = np.concatenate(id_available, [True])
        #self.groups.append(SensorGroup(id=new_group_id, signal=common_signal))
        self.graph = np.concatenate((self.graph, np.zeros((new_group_id, 1))), axis=1)
        self.graph = np.concatenate((self.graph, np.zeros((1, new_group_id + 1))), axis=0)
        # print("line 68: " + "u=" + str(u) + ", v=" + str(v) + ", new_id=" + str(new_group_id))
        self.graph[new_group_id][u] = 1
        self.graph[new_group_id][v] = 1
        return SensorGroup(id=new_group_id, signal=common_signal)

    # def merge_new_group(self, new_bag, new_groups):
    #     new_new_group = []
    #     be_kept = True
    #     for ng in new_groups:
    #         be_kept = True
    #         for g in self.groups:
    #             [common_signal, common_number_pulse] = ng.sensor.compare(g.sensor, self.max_skew)
    #             #if common_number_pulse == ng.sensor.number_of_pulse \
    #             #        and common_number_pulse == g.sensor.number_of_pulse:  # two vertices i and j are the same
    #             if self.is_the_same(g.sensor, ng.sensor, common_number_pulse):
    #                 self.merge_nodes(g, ng)
    #                 be_kept = False
    #                 new_bag.remove(ng.id)
    #                 break
    #             if common_number_pulse == g.sensor.number_of_pulse and not self.path_exist(g.id, ng.id) \
    #                     and self.graph[ng.id, :].sum() != 0:# i is a subset of j
    #                 self.graph[g.id][ng.id] = 1
    #             elif common_number_pulse == ng.sensor.number_of_pulse and not self.path_exist(ng.id, g.id)\
    #                     and self.graph[ng.id, :].sum() != 0: # j is a subset of i
    #                 self.graph[ng.id][g.id] = 1
    #         if be_kept:
    #             new_new_group.append(ng)
    #     return new_bag, new_new_group
    def merge_new_group(self, new_bag, old_bag):
        new_new_bag = []
        candid_bag = []
        candid_bag += list(np.where(self.graph.sum(axis=0) > 0)[0])
        candid_bag += list(np.where(self.graph.sum(axis=1) > 0)[0])
        candid_bag = list(set(candid_bag))
        old_bag = [x for x in candid_bag if x not in new_bag]
        if type(new_bag) is None or len(new_bag) == 0:
            return new_new_bag
        be_kept = True
        for ng_id in new_bag:
            ng = self.find_set(ng_id)
            be_kept = True
            for g_id in old_bag:
                g = self.find_set(g_id)
                [_, common_number_one, _] = ng.sensor.compare(g.sensor, self.max_skew)
                                                 #  two vertices i and j are the same
                relation = self.is_the_same(g.sensor, ng.sensor, common_number_one)
                if relation == 0:
                    self.merge_nodes(g, ng)
                    be_kept = False
                    #new_bag.remove(ng.id)
                    break
                if relation == 1 and not self.path_exist(g.id, ng.id) \
                        and self.graph[ng.id, :].sum() != 0:  # i is a subset of j
                    # print("line 121: " + "g=" + str(g.id) + ", ng=" + str(ng.id))
                    self.graph[g.id][ng.id] = 1
                elif relation == -1 and not self.path_exist(ng.id, g.id)\
                        and self.graph[ng.id, :].sum() != 0:  # j is a subset of i
                    # print("line 125: " + "g=" + str(g.id) + ", ng=" + str(ng.id))
                    self.graph[ng.id][g.id] = 1
            if be_kept:
                new_new_bag.append(ng_id)
        return new_new_bag

    # def merge_groups_after_subtraction(self):
    #     for g1_idx, g1 in enumerate(self.groups):
    #         if self.graph[g1_idx, :].sum() == 0:
    #             continue
    #         for g2_idx, g2 in enumerate(self.groups):
    #             if g2_idx < g1_idx:
    #                 continue
    #             [comm_signal, comm_number_pulse] = g1.sensor.compare(g2.sensor, self.max_skew)
    #             if self.is_the_same(g1.sensor, g2.sensor, comm_number_pulse):
    #                 self.merge_nodes(g1, g2)
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
                return 0
        sensor1_similarity = 1 - abs(common_number_one- sensor1.number_of_one) / max(common_number_one,
                                                                                     sensor1.number_of_one)
        if sensor1_similarity >= self.accuracy_similarity:
            return 1
        sensor2_similarity = 1 - abs(common_number_one - sensor2.number_of_one) / max(common_number_one,
                                                                                      sensor2.number_of_one)
        if sensor2_similarity >= self.accuracy_similarity:
            return -1
        return 2

    def plot_graph(self):
        plt.figure()
        rows, cols = np.where(self.graph == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.ion()
        plt.show()

    def zero_indegree_nodes(self):
        candid_nodes = np.where(self.graph.sum(axis=0) == 0)[0]
        deleted_nodes = np.where(self.graph.sum(axis=1) == 0)[0]
        nodes = [x for x in candid_nodes if x not in deleted_nodes]
        # nodes = []
        # for n in candid_nodes:
        #     if self.graph[n, :].sum() != 0:
        #         nodes.append(n)
        return nodes

    def neighbours(self, u):
        return np.where(self.graph[u] == 1)[0]

    def all_path(self, u):
        paths = []
        u_neighbors = self.neighbours(u)
        if len(u_neighbors) == 0:
            return [u]
        for neighbor in u_neighbors:
            path = [u]
            neighbor_paths = self.all_path(neighbor)
            if type(neighbor_paths[0]) == list:
                for neighbor_path in neighbor_paths:
                    neighbor_path = path + neighbor_path
                    paths.append(neighbor_path)
            else:
                path = path + neighbor_paths
                paths.append(path)
        if len(paths) < 2:
            return paths[0]
        return paths

    @staticmethod
    def all_nodes_in_paths(paths):
        nodes = []
        if type(paths[0]) != list:
            if paths[0] is None:
                return None
            else:
                return paths
        for path in paths:
            nodes = nodes + path
        #print (paths)
        #print (nodes)
        nodes_distinct = set(nodes)
        return list(nodes_distinct)

    def update_graph_removed_nodes(self):
        for g in self.groups:
            if g.sensor.number_of_one == 0: # < self.minimum_intervals:
                self.graph[g.id] = 0
                g.is_exist = False

    # def all_path(self, u, path=[]):
    #     paths = []
    #     u_neighbors = self.neighbours(u)
    #     if len(u_neighbors) == 0:
    #         return path + [u]
    #     for neighbor in u_neighbors:
    #         path = path + [u]
    #         # path = path + self.all_path(neighbor)
    #         #paths.append(path)
    #         return self.all_path(neighbor, path)
    #         paths.append(path)
    #     # return self.all_path(neighbor, paths)
    #     #return paths

    def find_set(self, u):
        for g in self.groups:
            if g.id == u:
                return g
        return None

    def create_graph(self):
        groups_size = len(self.groups)
        self.graph = np.zeros((groups_size, groups_size), dtype=int)
        #id_available = np.ones(groups_size, dtype=bool)
        old_bag = range(groups_size)
        new_bag = []
        new_groups = []
        while len(old_bag) != 0:     # This while intersection between sets till there is no more intersection
            new_bag = []
            new_groups = []
            # print (old_bag)
            for i_idx, i in enumerate(old_bag):
                if not self.find_set(i).is_exist:
                    continue
                for j_idx, j in enumerate(old_bag):
                    if not self.find_set(j).is_exist:
                        continue
                    # print (str(i) + " " + str(j))
                    if max(self.graph[i][j], self.graph[j][i]) != 0 or i_idx >= j_idx:
                        continue # if there is already relationship, continue
                    # print("i = " + str(i) + ", j = " + str(j)) # Comment
                    [common_signal, common_number_one, _] = self.groups[i].sensor.compare(self.groups[j].sensor,
                                                                                          self.max_skew)
                    relation = self.is_the_same(self.groups[i].sensor, self.groups[j].sensor, common_number_one)
                    if relation == 0:  # two vertices i and j are the same
                        self.merge_nodes(self.find_set(i), self.find_set(j))
                        # new_bag.remove(j)
                    elif relation == 1:
                        if not self.path_exist(i, j):  # i is a subset of j
                            # print("line 276: " + "i=" + str(i) + ", j=" + str(j))
                            self.graph[i][j] = 1
                    elif relation == -1:
                        if not self.path_exist(j, i):  # j is a subset of i
                            # print("line 280: " + "i=" + str(i) + ", j=" + str(j))
                            self.graph[j][i] = 1
                    elif common_number_one >= self.minimum_intervals: # add a new vertex of intersection of two vertices i and j
                        # [is_group_existed, existed_group_number] = self.group_existed(old_bag, new_groups, common_signal)
                        # if is_group_existed:
                        #     # print("line 283: " + "i="+str(i) + ", j=" + str(j) + ", new_id=" + str(existed_group_number))
                        #     self.graph[existed_group_number][i] = 1
                        #     self.graph[existed_group_number][j] = 1
                        # else:
                        g = self.new_node(i, j, common_signal)
                        new_groups.append(g)
                        new_bag.append(g.id)
            # self.plot_graph()
            self.groups += new_groups ###### need to be changed
            new_bag = self.merge_new_group(new_bag, old_bag)
            if self.debug:
                self.plot_graph()
                plt.savefig("graphs\\" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))
            self.remove_extra_edges(old_bag)
            # self.plot_graph()
            old_bag = new_bag
        if self.debug:
            self.plot_graph()
            plt.savefig("graphs\\" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))

    def group_existed(self, old_bag, new_groups, new_signal):
        new_group = SensorGroup(id=None, signal=new_signal, accuracy_parameter=self.accuracy_similarity)
        for ob_idx, ob in enumerate(old_bag):
            obg = self.find_set(ob)
            [_, common_number_one, _] = new_group.sensor.compare(obg.sensor, self.max_skew)
            if self.is_the_same(new_group.sensor, obg.sensor, common_number_one) == 0:
                return True, ob
        for ng in new_groups:
            [_, common_number_one, _] = new_group.sensor.compare(ng.sensor, self.max_skew)
            if self.is_the_same(new_group.sensor, ng.sensor, common_number_one) == 0:
                return True, ng.id
        return False, None


    def extract_distinct_intruders(self):
        zero_in_nodes = self.zero_indegree_nodes()
        while len(zero_in_nodes) != 0:
            for zero_node in zero_in_nodes:
                paths = self.all_path(zero_node)
                # print ("paths for node #"+ str(zero_node) + str(paths))
                group = self.find_set(zero_node)
                final_set = SensorGroup(id=len(self.final_sets), signal=group.sensor.signal)

                if type(paths[0]) != list:
                    path_temp = []
                    path_temp.append(paths)
                    paths = path_temp
                for path in paths:
                    for path_node in path:
                        group_node = self.find_set(path_node)
                        if group_node is not None:
                            final_set.append_group(group_node.group_list)
                self.final_sets.append(final_set)
                nodes = self.all_nodes_in_paths(paths)
                self.graph[zero_node] = 0
                if nodes is None:
                    break
                for node in nodes:
                    if node == zero_node:
                        continue
                    node_group = self.find_set(node)
                    node_group.sensor.subtract_signal(group.sensor.zero_one_signal, skew=self.max_skew)
                group.is_exist = False
            self.update_graph_removed_nodes()
            self.merge_groups_after_subtraction()
            zero_in_nodes = self.zero_indegree_nodes()
            if self.debug:
                self.plot_graph()
                plt.savefig("graphs\\" + datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()))

        #  Creating groups based on some singular vertices(groups) that are remained
        for g in self.groups:
            if g.is_exist:
                self.final_sets.append(SensorGroup(id=len(self.final_sets),
                                                  signal=g.sensor.signal, group_list=g.group_list))
        # ********* Get Observation value(maximum) for each sensor in final sets
        number_of_sensors = len(self.sensors)
        for final_set in self.final_sets:
            observation_vector = np.zeros(len(final_set.group_list), dtype=float)
            for observed_idx, sensor_idx in enumerate(final_set.group_list):
                observation_vector[observed_idx] = self.sensors[sensor_idx].observed_value(final_set.sensor.signal,
                                                                                           self.max_skew)
            final_set.observation_vector = observation_vector

    def get_intruders_list(self):
        return self.final_sets

    def get_final_locations(self, locations):
        return list(set(locations))
