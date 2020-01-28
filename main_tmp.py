import random as rd
from SensorSynchronization.Synchronize import *
import pickle
import datetime
import numpy as np
import time
from collections import defaultdict



def non_overlapping_txs():
    true_powers = np.random.choice(a=range(min_power, max_power), size=(num_intruders))
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

    return true_powers


def generating_transmitters():
    true_powers = np.random.choice(a=range(min_power, max_power), size=(num_intruders))
    max_length = 0
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
    return true_powers


def creating_sensors():
    # ****************** Creating Sensors with a random combination of intruders' signal ******
    # sensors = []
    # ground_truth_list_tx_sensors = []
    length_signal = len(transmitters_signal[0])
    for i in range(number_of_sensors):
        list_of_txs = rd.sample(range(0, num_intruders), max(1, rd.randrange(1, num_intruders + 1)))
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

def print_nodes(synchronize, ground_truth_list_tx_sensors):

    f = open("nodes.txt", "w")
    f.write("Nodes:")
    for group_idx, group in enumerate(synchronize.groups):
        if ground_truth_list_tx_sensors is None or len(ground_truth_list_tx_sensors)==0:
            f.write("\n\nnode " + str(group_idx) + ":\t\t ")
        else:
            f.write("\n\nnode " + str(group_idx) + ":\t\t Tx(s):{" +
                    str(ground_truth_list_tx_sensors[group.sensors_list[0]]) + str("}"))
        f.write("\n\t\tlist sensors:{" + str(group.sensors_list) + "}")
    f.close()


def print_final_sets(final_sets):
    if final_sets is None:
        return
    f = open("nodes.txt", "a+")
    f.write("\n\n\nFinal sets:")
    for set_idx, final_set in enumerate(final_sets):
        f.write("\n\nset " + str(set_idx) + ":\t\t Sensors:{" +
                str(final_set.sensors_list) + str("}"))
    f.close()


if __name__ == '__main__':
    # *************** Initializing Parameters ************************88
    accuracy_similarity = 0.8
    debug = False
    # *********** Intruders ********************
    lam = 8  # lambda of poisson distribution of arrival rate of pulses
    num_intruders = 4
    time_span = 0.000001
    t_on_prob = 0.5
    minimum_ni = 10  # minimum number of pulses
    maximum_ni = 15  # maximum number of pulses
    maximum_ton_length = 3
    minimum_ton_length = 2
    min_power = -36  # -2
    max_power = -32  # 2
    length_signal = 0
    true_powers = None
    true_indices = None

    # **************** Sensors *****************8
    noise_level = -80
    accuracy_parameter = 0.9
    number_of_sensors = 10

    t_of_signal_power = noise_level - 20

    max_skew = 5

    sensors = []
    aligned_sensors = []
    localize_sensors = []
    ground_truth_list_tx_sensors = []
    transmitters_signal = []
    transmitters_signal = [[] for i in range(num_intruders)]


    # true_powers = generating_transmitters()

    true_powers = non_overlapping_txs()
    creating_sensors()

    synchronize = Synchronize(sensors, max_skew, 1.8 * minimum_ni, accuracy_similarity)
    intruders = synchronize.synchronize()
    print_nodes(synchronize, ground_truth_list_tx_sensors)
    print_final_sets(intruders)