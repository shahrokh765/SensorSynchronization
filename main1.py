import numpy as np
import random as rd
from synchronization.Sensor import *
from synchronization.Synchronize import *
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import datetime
from uuid import uuid4
from os import path
import numpy as np
import sys
#print sys.path
sys.path.append(path.abspath('..\\Localization\\'))
from Localization.utility import generate_intruders
from Localization.utility import read_config, ordered_insert, power_2_db, power_2_db_, db_2_power, db_2_power_, find_elbow
from Localization.sensor import Sensor as LocSensor
from Localization.localize import Localization
import time
from Localization.plots import visualize_localization, visualize_sensor_output
from sklearn.cluster import KMeans
from collections import defaultdict


def print_nodes(synchronize, ground_truth_list_tx_sensors):

    f = open("nodes.txt", "w")
    f.write("Nodes:")
    for group_idx, group in enumerate(synchronize.groups):
        if ground_truth_list_tx_sensors is None or len(ground_truth_list_tx_sensors)==0:
            f.write("\n\nnode " + str(group_idx) + ":\t\t ")
        else:
            f.write("\n\nnode " + str(group_idx) + ":\t\t Tx(s):{" +
                    str(ground_truth_list_tx_sensors[group.group_list[0]]) + str("}"))
        f.write("\n\t\tlist sensors:{" + str(group.group_list) + "}")
    f.close()


def print_final_sets(final_sets):
    if final_sets is None:
        return
    f = open("nodes.txt", "a+")
    f.write("\n\n\nFinal sets:")
    for set_idx, final_set in enumerate(final_sets):
        f.write("\n\nset " + str(set_idx) + ":\t\t Sensors:{" +
                str(final_set.group_list) + str("}"))
    f.close()


def dist_to_centroids(points, centroids, points_index):
    dist = 0
    i = 0
    for x, y in points:
        centroid = centroids[points_index[i]]
        dist += math.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
        i += 1
    return dist


def centroid_to_cell_centers(locations):
    if locations is None or len(locations) == 0:
        return []
    cell_indices = []
    for x, y in locations:
        cell_indices.append(int(round(x) * grid_len + round(y)))
    return selectsensor.convert_to_pos(cell_indices)


def clustering(points, max_k):  # weight should be added to points
    points_array = []
    for point in points:
        points_array += point
    points_array = np.array(points_array)
    sum_dist_to_centroid = np.zeros(max_k, dtype=float)
    centroids = []
    for k in range(1, max_k + 1):
        if len(points_array) < k:
            centroids.append(points_array)
            sum_dist_to_centroid[k-1] = 0
        else:
            try:
                kmeans = KMeans(n_clusters=k).fit(points_array)
                centroids.append(kmeans.cluster_centers_)
                sum_dist_to_centroid[k-1] = dist_to_centroids(points_array, kmeans.cluster_centers_, kmeans.labels_)
            except Exception as e:
                break
            except Warning as w:
                break
    if max_k < 4:
        if len(centroids[1]) == 0 or centroids[1] is None:
            if len(centroids[0]) == 0 or centroids[0] is None:
                return 0, None
            return 1, centroids[0]
        x0, y0 = centroids[max_k - 1][0]
        x1, y1 = centroids[max_k - 1][1]
        if x0 == x1 and y0 == y1:
            return 1, centroids[0]
        return max_k, centroids[max_k - 1]
    for k in range(1, max_k-1):  # choose appropriate k. the point where the previous and next slope are not that different
        prev_slope = abs(sum_dist_to_centroid[k] - sum_dist_to_centroid[k-1])
        next_slope = abs(sum_dist_to_centroid[k+1] - sum_dist_to_centroid[k])
        if next_slope < 0.3 * prev_slope:
            return k+1, centroids[k]
    return max_k, centroids[max_k-1]


def draw_signals(transmitters_signal, time_space, type):
    num_of_signals = len(transmitters_signal)
    fig, axs = plt.subplots(num_of_signals, 1, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs[0].set_title(type + " signals")
    axs = axs.ravel()

    for i in range(num_of_signals):
        time = np.arange(0, len(transmitters_signal[i]) * time_space, time_space)
        axs[i].plot(time, transmitters_signal[i])
        axs[i].set_xlabel(type + " #" + str(i))


def draw_groups(synchronize, time_space):
    group_signals = []
    for group in synchronize.groups:
        group_signals.append(group.sensor.zero_one_signal)
    draw_signals(group_signals, time_space, "group")


def draw_graph(graph):
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
    G = nx.DiGraph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(G)
    # nx.draw(G, pos)
    nx.draw(G, node_size=500, with_labels=True)

    # show graph
    plt.show()


def non_overlapping_txs():
    # true_powers = [2, 2, 2, 2]  # extra
    # true_indices = [1515, 423, 1292, 2287]  # extra
    # true_indices = [1179, 1079]# [705, 855] # , 705]
    # true_powers = [2, 2]
    true_powers = np.random.choice(a=range(min_power, max_power), size=(num_intruders))
    true_indices, true_powers = generate_intruders(grid_len=grid_len, edge=2, num=num_intruders,
                                                   min_dist=min_dist_intruders, powers=true_powers)
    for intrud_ind in true_indices:  # extra
        intruders_location.append(Point(x=intrud_ind // grid_len, y=intrud_ind % grid_len))  # extra
    number_of_interval = np.zeros(num_intruders, dtype=int)
    # ********************** Creating non-overlapping random pulses for each intruder *****************
    while min(number_of_interval) - minimum_ni < 0:
        if np.random.choice(a=[True, False], size=(1), p=[t_on_prob, 1 - t_on_prob]):
            intrud_id = rd.randrange(num_intruders)
            #length = rd.randrange(maximum_ton_length) + 2
            length = rd.randint(minimum_ton_length, maximum_ton_length)
            if len(transmitters_signal[intrud_id]) > 0:
                if transmitters_signal[intrud_id][-1] > noise_level:
                    continue
            # transmitters_signal[intrud_id] = np.append(transmitters_signal[intrud_id],
            #                                            np.random.uniform(noise_level+0.001, 0, length))
            transmitters_signal[intrud_id] = np.append(transmitters_signal[intrud_id],
                                                       true_powers[intrud_id] * np.ones(length, dtype=float))
            length2 = rd.randrange(maximum_ton_length) + 1
            # transmitters_signal[intrud_id] = np.append(transmitters_signal[intrud_id],
            #                                            np.random.uniform(noise_level - 50, noise_level - 15, length2))
            transmitters_signal[intrud_id] = np.append(transmitters_signal[intrud_id],
                                                       t_of_signal_power * np.ones(length2, dtype=float))
            number_of_interval[intrud_id] += 1
            length += length2
            for i in range(num_intruders):
                if i == intrud_id:
                    continue
                transmitters_signal[i] = np.append(transmitters_signal[i],
                                                   t_of_signal_power * np.ones(length, dtype=float))
                                                   # np.random.uniform(noise_level - 50, noise_level - 15, length))
        else:
            length = rd.randrange(maximum_ton_length) + 1
            for i in range(num_intruders):
                transmitters_signal[i] = np.append(transmitters_signal[i],
                                                   t_of_signal_power * np.ones(length, dtype=float))
                                                   # np.random.uniform(noise_level - 50, noise_level - 15, length))

    return true_powers, true_indices


def generating_transmitters():
    true_powers = np.random.choice(a=range(min_power, max_power), size=(num_intruders))
    true_indices, true_powers = generate_intruders(grid_len=grid_len, edge=2, num=num_intruders,
                                                   min_dist=min_dist_intruders, powers=true_powers)
    #true_powers = [-2, 0, -2, 0]
    #true_powers = [2, 2, 2, 2]
    #true_indices = [1179, 1079]  # [705, 855] # , 705]
    #true_powers = [2, 2]
    max_length = 0
    for intrud_ind in true_indices:
        intruders_location.append(Point(x=intrud_ind//grid_len, y=intrud_ind % grid_len))
    for i in range(num_intruders):
        number_of_pulses = rd.randint(minimum_ni, maximum_ni)
        interval_diff = np.random.poisson(lam, number_of_pulses)
        for int_diff in interval_diff:
            # transmitters_signal[i] = np.append(transmitters_signal[i], # add noise between two pulses
            #                                    np.random.uniform(noise_level - 50, noise_level - 20, int_diff))
            transmitters_signal[i] = np.append(transmitters_signal[i],  # add noise between two pulses
                                               t_of_signal_power * np.ones(int_diff))
            transmitters_signal[i] = np.append(transmitters_signal[i],
                                               true_powers[i] * np.ones(rd.randint(minimum_ton_length,
                                                                                   maximum_ton_length)))
        transmitters_signal[i] = np.append(transmitters_signal[i], [t_of_signal_power])
        if len(transmitters_signal[i]) > max_length:
            max_length = len(transmitters_signal[i])
    # padding some noise values to transmitters to have equal length
    for i in range(num_intruders):
        if len(transmitters_signal[i]) != max_length:
            # transmitters_signal[i] = np.append(transmitters_signal[i], # add noise between two pulses
            #                                     np.random.uniform(noise_level - 50, noise_level - 20,
            #                                                       max_length-len(transmitters_signal[i])))
            transmitters_signal[i] = np.append(transmitters_signal[i],  # make all the signals the same size
                                               t_of_signal_power * np.ones(max_length - len(transmitters_signal[i])))
    return true_powers, true_indices


def creating_sensors():
    # ****************** Creating Sensors with a random combination of intruders' signal ******
    sensors = []
    ground_truth_list_tx_sensors = []
    length_signal = len(transmitters_signal[0])
    for i in range(number_of_sensors):
        list_of_txs = rd.sample(range(0, num_intruders), max(2, rd.randrange(num_intruders) - 1))
        ground_truth_list_tx_sensors.append(list_of_txs)
        signal = np.zeros(length_signal, dtype=float)
        for tx in list_of_txs:  # sum all the intruders hearing from
            signal = np.add(signal, np.power(10, np.true_divide(transmitters_signal[tx], 10)))
        signal = np.multiply(10, np.log10(signal))
        # pad zero signals to simulate drifts between sensors' clock
        b = (noise_level - 50) * np.ones(rd.randrange(max_skew), dtype=float)
        signal = np.concatenate((b, signal))
        sensors.append(Sensor(id=i, signal=signal, noise_floor=noise_level, keep_interval=False))
    return sensors, ground_truth_list_tx_sensors


def creating_sensors2(sensor_file, accuracy_similarity):  # read sensor from file
    # **************** Creating Sensors based on their locations and intruders location

    length_signal = len(transmitters_signal[0])
    max_sensor_length = length_signal

    # Read sensors from file
    with open(sensor_file, 'r') as f:
        max_gain = 0.5 * num_intruders  # len(self.transmitters)
        index = 0
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            x, y, std, cost = int(line[0]), int(line[1]), float(line[2]), float(line[3])
            std = 1 # based on caitao
            location = Point(x, y)
            signal = np.zeros(length_signal, dtype=float)
            list_sensor_tx = []
            for tx_idx, tx_signal in enumerate(transmitters_signal):  # sum all the intruders hearing from
                distance = location.distance(intruders_location[tx_idx])
                if distance < 1:
                    list_sensor_tx.append(tx_idx)
                else:
                    if max(true_powers) - path_loss_coef * math.log10(distance) > noise_level:
                        list_sensor_tx.append(tx_idx)
                signal_temp = tx_signal
                if distance > 1:
                    signal_temp = tx_signal - path_loss_coef * math.log10(distance)  # path loss
                signal = np.add(signal, np.power(10, np.true_divide(signal_temp, 10)))
            ground_truth_list_tx_sensors.append(list_sensor_tx)
            signal = np.multiply(10, np.log10(signal))
            signal = np.power(10, np.true_divide(signal, 10))
            if randomness:
                signal = np.add(signal, np.power(10, np.true_divide(np.random.normal(noise_level - 20,
                                                                                     std, len(signal)), 10)))
            signal = np.multiply(10, np.log10(signal))

            aligned_sensors.append(Sensor(id=index, signal=signal, location=location,
                                          noise_floor=noise_level, keep_interval=False, cost=cost, std=std,
                                          accuracy_parameter=accuracy_similarity))
            # pad zero signals to simulate drifts between sensors' clock
            b = (noise_level - 50) * np.ones(rd.randrange(max_skew), dtype=float)
            signal = np.concatenate((b, signal))
            sensors.append(Sensor(id=index, signal=signal, location=location,
                                  noise_floor=noise_level, keep_interval=False, cost=cost, std=std,
                                  accuracy_parameter=accuracy_similarity))
            localize_sensors.append(LocSensor(x, y, std, cost, gain_up_bound=max_gain, index=index))  # uniform sensors
            if len(signal) > max_sensor_length:
                max_sensor_length = len(signal)
            index += 1
    for sensor in sensors:
        length_tmp = len(sensor.signal)
        if length_tmp == max_sensor_length:
            continue
        sensor.signal = np.concatenate((sensor.signal, (noise_level - 50) * np.ones(max_sensor_length - length_tmp,
                                                                                    dtype=float)))
        sensor.size = len(sensor.signal)
    return len(sensors)

if __name__ == '__main__':
    # aa = np.random.poisson(7, 1000)


    #aa = np.array([[1, 2], [2, 3], [3, 4], (3, 5)])
    f = open("results/error.txt", "w")

    # *************** Initializing Parameters ************************88
    path_loss_coef = 49 # 40
    accuracy_similarity = 0.8
    max_number_intruders = 7
    min_number_intruders = 7
    repeats = 50
    debug = False
    # *********** Intruders ********************
    lam = 8  # lambda of poisson distribution of arrival rate of pulses
    num_intruders = 10
    time_span = 0.000001
    t_on_prob = 0.5
    minimum_ni = 10  # minimum number of pulses
    maximum_ni = 15  # maximum number of pulses
    maximum_ton_length = 3
    minimum_ton_length = 2
    min_power = -36 # -2
    max_power = -32 # 2
    min_dist_intruders = 10  # minimum distance between intruders
    length_signal = 0
    true_powers = None
    true_indices = None


    grid_len = 50
    # **************** Sensors *****************8
    noise_level = -80
    #number_of_sensors = 4
    accuracy_parameter = 0.9
    sensor_file_path = '../Localization/data50/homogeneous-50/sensors'
    number_of_sensors = 50
    randomness = True  # add gaussian noise with mean=0 and a std

    t_of_signal_power = noise_level - 20

    min_max_skew = 5
    max_max_skew = 5
    skew_step = 1

    aligned_splot_time = 0.0
    splot_time = 0.0
    synchronize_time = 0.0

    errors_skewed_all = []
    misses_skewed_all = []
    false_alarms_skewed_all = []
    errors_aligned_all = []
    misses_aligned_all = []
    false_alarms_aligned_all = []
    errors_sync_all = []
    misses_sync_all = []
    false_alarms_sync_all = []
    splot_time_all = []
    aligned_splot_time_all = []
    synchronize_time_all = []

    true_locations_all = defaultdict(list)
    pred_locations_skewed_all = defaultdict(list)
    pred_locations_algn_all = defaultdict(list)
    pred_locations_sync_all = defaultdict(list)

    var_f = open("results/variables_" + str(number_of_sensors) + "sensors_" + str(min_dist_intruders) + "dist_" +
                 str(min_number_intruders) + "to" + str(max_number_intruders) + "intruders_" + str(max_max_skew) + "maxSkew_" +
                 datetime.datetime.now().strftime('_%Y%m_%d%H_%M')+ ".txt", "wb")
    #b = range(1, 50, 5)
    for max_skew in range(min_max_skew, max_max_skew + 1, skew_step):
        print ("\nMax_Skew: " + str(max_skew) + "**********************")
        for num_intruders in range(min_number_intruders, max_number_intruders + 1):
            errors_skewed = []
            misses_skewed = []
            false_alarms_skewed = []
            errors_aligned = []
            misses_aligned = []
            false_alarms_aligned = []
            errors_sync = []
            misses_sync = []
            false_alarms_sync = []
            splot_time = 0.0
            aligned_splot_time = 0.0
            synchronize_time = 0.0

            print("------------------- Intruders: " + str(num_intruders) + " ------------------------")
            for repeat in range(1, repeats + 1):
                sensors = []
                aligned_sensors = []
                localize_sensors = []
                ground_truth_list_tx_sensors = []
                pred_locations = []  # used for synchronize&splot
                pred_locations_splot = []  # used for skewed&splot
                pred_locations_algn_splot = []  # used for aligned&splot
                intruders_splot = []  # intruders for splot, used for visualizing and error calculations
                transmitters_signal = [[] for i in range(num_intruders)]
                intruders_location = []
                selectsensor = None
                dictinory_ind = max_skew * 100000 + num_intruders * 1000 + repeat
                print("\n\n** Iteration: " + str(repeat) + "**")

                try:
                    #true_powers, true_indices = generating_transmitters()

                    true_powers, true_indices = non_overlapping_txs()
                    number_of_sensors = creating_sensors2(sensor_file_path, accuracy_parameter)
                except Exception as e:
                    print(e)
                    continue

                # draw_signals(transmitters_signal, time_span, 'Transmitters')
                # length_signal = len(transmitters_signal[0])
                # #transmitters_signal = non_overlapping_txs()
                # draw_signals([sensors[0].signal, sensors[0].zero_one_signal, sensors[1].signal,
                #               sensors[1].zero_one_signal], time_span, 'Transmitters')
                #sensors, ground_truth_list_tx_sensors = creating_sensors()
                #

                # # # ************** load a specific configuration to debug ****************
                # file = open('tx_signal.txt', 'w')
                # # pickle.dump(transmitters_signal, file)
                # file = open('tx_signal.txt', 'r')
                # transmitters_signal = pickle.load(file)
                # file.close()
                # # file1 = open('tx.txt', 'w')
                # # pickle.dump(ground_truth_list_tx_sensors, file1)
                # file1 = open('tx.txt', 'r')
                # ground_truth_list_tx_sensors = pickle.load(file1)
                # file1.close()
                # # file2 = open('sensors.txt', 'w')
                # # pickle.dump(sensors, file2)
                # file2 = open('sensors.txt', 'r')
                # sensors = pickle.load(file2)
                # file2.close()
                sensor_outputs = np.zeros(number_of_sensors, dtype=float)
                selectsensor = Localization(grid_len=50, debug=debug)
                selectsensor.sensors = localize_sensors

                true_locations = selectsensor.convert_to_pos(true_indices)
                true_locations_all[dictinory_ind].append(true_locations)

                for ind in true_indices:
                    intruders_splot.append(selectsensor.transmitters[ind])

                i = 0
                if 1 == 1:  # make it true to run splot for each observation vector
                    start = time.time()
                    length_signal = len(sensors[0].signal)
                    for i in range(length_signal): # pass observation vector for each observation point
                        for sensor_idx, sensor in enumerate(sensors):
                            sensor_outputs[sensor_idx] = sensor.signal[i]

                        pred_location = selectsensor.splot_localization(sensor_outputs, intruders_splot, fig=i)
                        pred_locations_splot.append(pred_location)
                    k, pred_locations_splot_cluster = clustering(pred_locations_splot, 2*num_intruders)
                    print("Number of Clusters:" + str(k))
                    pred_locations_splot_cluster_locations = centroid_to_cell_centers(pred_locations_splot_cluster)
                    pred_locations_splot_cluster_locations = list(set(pred_locations_splot_cluster_locations))
                    pred_locations_skewed_all[dictinory_ind].append(pred_locations_splot_cluster_locations)

                    try:
                        print('\nSkewed & SPLOT')
                        splot_time += time.time() - start
                        error, miss, false_alarm = selectsensor.compute_error2(true_locations,
                                                                               pred_locations_splot_cluster_locations)
                        if len(error) != 0:
                            errors_skewed.extend(error)
                        misses_skewed.append(miss)
                        false_alarms_skewed.append(false_alarm)
                        print('error/miss/false/time(s) = {}/{}/{}/{}'.format(np.array(error).mean(), miss, false_alarm,
                                                                              splot_time))

                        # visualize_localization(selectsensor.grid_len, true_locations, pred_locations_splot_cluster_locations,
                        #                        datetime.datetime.now().strftime('%Y%m_%d%H_%M%S_') +
                        #                        'skewed_' + "maxSkew" + str(max_skew) +
                        #                        '_intruders' + str(num_intruders) + '_iter' + str(repeat))
                    except Exception as e:
                        print(e)

                if 1 == 1:  # SPLOT with aligned signals
                    start = time.time()
                    length = len(aligned_sensors[0].signal)
                    for i in range(length): # pass observation vector for each observation point
                        for sensor_idx, sensor in enumerate(aligned_sensors):
                            sensor_outputs[sensor_idx] = sensor.signal[i]

                        pred_location = selectsensor.splot_localization(sensor_outputs, intruders_splot, fig=i)
                        pred_locations_algn_splot.append(pred_location)
                    k, pred_locations_algn_splot_cluster = clustering(pred_locations_algn_splot, 2 * num_intruders)
                    print("Number of Clusters:" + str(k))
                    pred_locations_algn_splot_cluster_locations = centroid_to_cell_centers(pred_locations_algn_splot_cluster)

                    pred_locations_algn_splot_cluster_locations = list(set(pred_locations_algn_splot_cluster_locations))
                    pred_locations_algn_all[dictinory_ind].append(pred_locations_algn_splot_cluster_locations)
                    try:
                        print('\nAligned & SPLOT')
                        aligned_splot_time += time.time() - start
                        error, miss, false_alarm = selectsensor.compute_error2(true_locations,
                                                                               pred_locations_algn_splot_cluster_locations)
                        if len(error) != 0:
                            errors_aligned.extend(error)
                        misses_aligned.append(miss)
                        false_alarms_aligned.append(false_alarm)
                        print('error/miss/false/time(s) = {}/{}/{}/{}'.format(np.array(error).mean(), miss, false_alarm,
                                                                              aligned_splot_time))

                        # visualize_localization(selectsensor.grid_len, true_locations,
                        #                        pred_locations_algn_splot_cluster_locations,
                        #                        datetime.datetime.now().strftime('%Y%m_%d%H_%M%S_') +
                        #                        'aligned_' + "maxSkew" + str(max_skew) +
                        #                        '_intruders' + str(num_intruders) + '_iter' + str(repeat))

                    except Exception as e:
                        print(e)

                # **************** Localizing & Synchronizing
                start = time.time()
                try:
                    synchronize = Synchronize(sensors, max_skew, 1.8*minimum_ni, accuracy_similarity)
                    synchronize.create_groups()
                    # draw_groups(synchronize, time_span)
                    print_nodes(synchronize, ground_truth_list_tx_sensors)
                    synchronize.create_graph()
                    synchronize.extract_distinct_intruders()
                    intruders = synchronize.get_intruders_list()
                    print_final_sets(intruders)
                    for intruder in intruders:
                        i += 1  # just for plotting and having different name
                        sensor_outputs = (noise_level - 50) * np.ones(number_of_sensors, dtype=float)
                        for sensor, sensor_idx in enumerate(intruder.group_list):
                            sensor_outputs[sensor_idx] = intruder.observation_vector[sensor]
                        pred_location = selectsensor.splot_localization(sensor_outputs, intruders_splot, fig=i)
                        pred_locations += pred_location
                    # pred_locations = synchronize.get_final_locations(pred_locations)
                    pred_locations = list(set(pred_locations))
                    pred_locations_sync_all[dictinory_ind].append(pred_locations)
                except Exception as e:
                    print(e)
                try:
                    print('\nSynchronized & SPLOT')
                    synchronize_time += time.time() - start
                    error, miss, false_alarm = selectsensor.compute_error2(true_locations,
                                                                           pred_locations)
                    if len(error) != 0:
                        errors_sync.extend(error)
                    misses_sync.append(miss)
                    false_alarms_sync.append(false_alarm)
                    print('error/miss/false/time(s) = '
                          '{}/{}/{}/{}'.format(np.array(error).mean(), miss, false_alarm, synchronize_time))
                    # visualize_localization(selectsensor.grid_len, true_locations, pred_locations,
                    #                        datetime.datetime.now().strftime('%Y%m_%d%H_%M%S_') +
                    #                        'synchronized_' + "maxSkew" + str(max_skew) +
                    #                        '_intruders' + str(num_intruders) + '_iter' + str(repeat))
                except Exception as e:
                    print(e)
            print("\n\n*** Calculation for number of intruders:" + str(num_intruders) +"\n")
            try:
                print("\t\t *** Skewed Signal")
                errors = np.array(errors_skewed)
                print('\t\t\t(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}),'
                      ' false_alarm=({}/{}/{})'.format(round(errors.mean(), 3), round(errors.max(), 3),
                                                       round(errors.min(), 3), round(sum(misses_skewed)/repeats, 3),
                                                       max(misses_skewed), min(misses_skewed),
                                                       round(sum(false_alarms_skewed)/repeats, 3), max(false_alarms_skewed),
                                                       min(false_alarms_skewed)))
                print('\t\t\t Average time = ', splot_time/repeats)

                print("\t\t *** Aligned Signal")
                errors = np.array(errors_aligned)
                print('\t\t\t(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}),'
                      ' false_alarm=({}/{}/{})'.format(round(errors.mean(), 3), round(errors.max(), 3),
                                                       round(errors.min(), 3), round(sum(misses_aligned)/repeats, 3),
                                                       max(misses_aligned), min(misses_aligned),
                                                       round(sum(false_alarms_aligned)/repeats, 3)/repeats, max(false_alarms_aligned),
                                                       min(false_alarms_aligned)))
                print('\t\t\t Average time = ', aligned_splot_time/repeats)

                print("\t\t *** Synchronized Signal")
                errors = np.array(errors_sync)
                print('\t\t\t(mean/max/min) error=({}/{}/{}), miss=({}/{}/{}),'
                      ' false_alarm=({}/{}/{})'.format(round(errors.mean(), 3), round(errors.max(), 3),
                                                       round(errors.min(), 3), round(sum(misses_sync)/repeats, 3),
                                                       max(misses_sync), min(misses_sync),
                                                       round(sum(false_alarms_sync)/repeats, 3), max(false_alarms_sync),
                                                       min(false_alarms_sync)))
                print('\t\t\t Average time = ', synchronize_time/repeats)
            except:
                print('Empty list!')
            print("\n\n\n")

            #var_f.close()

            errors_skewed_all.append(errors_skewed)
            misses_skewed_all.append(misses_skewed)
            false_alarms_skewed_all.append(false_alarms_skewed)
            errors_aligned_all.append(errors_aligned)
            misses_aligned_all.append(misses_aligned)
            false_alarms_aligned_all.append(false_alarms_aligned)
            errors_sync_all.append(errors_sync)
            misses_sync_all.append(misses_sync)
            false_alarms_sync_all.append(false_alarms_sync)
            splot_time_all.append(splot_time)
            aligned_splot_time_all.append(aligned_splot_time)
            synchronize_time_all.append(synchronize_time)

    pickle.dump([errors_skewed_all, misses_skewed_all, false_alarms_skewed_all, errors_aligned_all, misses_aligned_all,
                 false_alarms_aligned_all, errors_sync_all, misses_sync_all, false_alarms_sync_all, splot_time_all,
                 aligned_splot_time_all, synchronize_time_all, true_locations_all, pred_locations_skewed_all,
                 pred_locations_algn_all, pred_locations_sync_all], file=var_f)
    f.close()