import sys
from os import path
import numpy as np
#print sys.path
# sys.path.append(path.abspath('..\\MovingTransmitter\\'))
from Commons.Point import *


class Sensor(object):
    def __init__(self, id, location=Point(), signal=[], noise_floor=-90, keep_interval=True, cost=0, std=1,
                 accuracy_parameter=1):
        self.location = location
        self.signal = signal
        self.size = len(signal)
        self.noise_floor = noise_floor
        self.number_of_pulse = 0
        self.number_of_non_noise_value = 0
        self.zero_one_signal = np.zeros( self.size, dtype=bool)
        self.keep_interval = keep_interval
        self.id = id
        self.number_of_one = 0
        self.cost = cost
        self.std = std
        self.accuracy_parameter = accuracy_parameter
        self.pre_process()

    def pre_process(self):
        self.get_number_pulse()
        self.zero_one_sequence(self.keep_interval)

    def get_number_pulse(self):
        [self.number_of_pulse, self.number_of_non_noise_value] = self.number_pulse(self.signal, self.noise_floor,
                                                                                   self.accuracy_parameter)

    @staticmethod
    def number_pulse(signal, noise_floor, accuracy_parameter=1):
        number_of_pulse = 0
        number_of_non_noise_value = 0
        decimal_signal = np.power(10, np.divide(signal, 10))
        threshold = math.pow(10, (noise_floor)/10)
        #if signal[0] > noise_floor:
        if decimal_signal[0] > accuracy_parameter * threshold:
            number_of_non_noise_value += 1
            number_of_pulse += 1
        for i in range(1, len(signal)):
            if decimal_signal[i] > accuracy_parameter * threshold:
                number_of_non_noise_value += 1
                #if decimal_signal[i] - decimal_signal [i-1] > accuracy_parameter * threshold:
                if decimal_signal[i - 1] <= accuracy_parameter * threshold:
                    number_of_pulse += 1
        return [number_of_pulse, number_of_non_noise_value]

    # def zero_one_sequence(self, keep_intervals = True):
    #     if keep_intervals:
    #         for i in range(self.size):
    #             if self.signal[i] > self.noise_floor:
    #                 self.zero_one_signal[i] = True
    #     else:
    #         if self.signal[0] > self.noise_floor:
    #             self.zero_one_signal[0] = True
    #
    #         for i in range(1, self.size):
    #             # if self.signal[i] > self.noise_floor and  self.signal[i - 1] <= self.noise_floor:
    #             if self.signal[i - 1] <= self.noise_floor < self.signal[i]:
    #                 self.zero_one_signal[i] = True
    #             elif self.signal[i - 1] > self.noise_floor >= self.signal[i]:
    #                 self.zero_one_signal[i - 1] = True
    #     self.number_of_one = sum(self.zero_one_signal)
    def zero_one_sequence(self, keep_intervals = True):
        decimal_signal = np.power(10, np.divide(self.signal, 10))
        threshold = math.pow(10, self.noise_floor/10)
        if keep_intervals:
            for i in range(self.size):
                if self.signal[i] > self.noise_floor:
                    self.zero_one_signal[i] = True
        else:
            if decimal_signal[0] > threshold:
                self.zero_one_signal[0] = True

            for i in range(1, self.size):
                # if self.signal[i] > self.noise_floor and  self.signal[i - 1] <= self.noise_floor:
                if decimal_signal[i] - decimal_signal[i-1] > self.accuracy_parameter * threshold:
                    self.zero_one_signal[i] = True
                elif decimal_signal[i] - decimal_signal[i-1] < -self.accuracy_parameter * threshold:
                    self.zero_one_signal[i - 1] = True
        self.number_of_one = sum(self.zero_one_signal)

    # compare will return a new sensor whose zero-one sequence is intersection of the two sensors s.t
    # maximum number of intervals can be aligned by shifting one of the sensors
    def compare(self, sensor, skew=None):
        if skew is None:
            skew = max(self.size, sensor.size)
        intersect_num = -1
        shift = 0
        for i in range(-skew-1, skew + 1):  ### <implement> consider shifting from -skew to skew
            [out_temp, intersect_temp] = self.intersect(self.zero_one_signal, sensor.zero_one_signal, i)
            if intersect_temp > intersect_num:
                intersect_num = intersect_temp
                out = out_temp
                shift = i
        #return [out, self.number_pulse(signal=out, noise_floor=0)[0]]
        return [out, intersect_num, shift]

    def subtract_signal(self, signal, skew=None):
        if skew is None:
            skew = max(self.size, len(signal))
        signal_number_of_one = sum(signal)
        similarity_max = 0
        i_max = None
        found = False
        for i in range(-skew-1, skew + 1): ### <implement> consider shifting from -skew to skew
            [_, intersect_temp] = self.intersect(self.zero_one_signal, signal, i)
            sensor_similarity = 1 - abs(intersect_temp - signal_number_of_one) / max(intersect_temp,
                                                                                     signal_number_of_one)
            if sensor_similarity == 1:
                found = True
                i_max = i
                break
            if sensor_similarity > similarity_max:
                similarity_max = sensor_similarity
                i_max = i
        if similarity_max > self.accuracy_parameter:
            found = True
        if not found:
            return
        [self.zero_one_signal, self.number_of_one] = self.intersect(self.zero_one_signal,
                                                                                np.logical_not(signal), i_max)
        [self.number_of_pulse, _] = self.number_pulse(signal=self.zero_one_signal, noise_floor=0)

    def observed_value(self, signal_mask, skew):
        max_value = -math.inf
        [_, _, shift] = self.compare(Sensor(id=None, signal=signal_mask, noise_floor=0), skew=skew)
        if shift > 0:
            signal_mask = np.concatenate((np.zeros(shift, dtype=signal_mask.dtype), signal_mask))
        else:
            signal_mask = signal_mask[-shift:]

        max_idx = min(len(signal_mask), len(self.signal))
        true_indices = np.where(signal_mask[:max_idx] == True)[0]
        return max(np.take(self.signal, true_indices))

    @staticmethod
    def intersect(signal1, signal2, shift):
        size1 = len(signal1)
        if shift > 0:
            b = np.zeros(shift, dtype=signal2.dtype)
            signal2 = np.concatenate((b, signal2))
        else:
            signal2 = signal2[-shift:]
        size2 = len(signal2)
        if size1 > size2:
            b = np.zeros(size1 - size2, dtype=signal2.dtype)
            signal2 = np.concatenate((signal2, b))
        out = np.multiply(signal1, signal2[:size1])
        #out = np.zeros((1, min(size1, size2)), dtype=bool)
        # out = np.zeros((1, size1), dtype=bool)
        # start = shift
        # if size1 < size2:
        #     end = min(size2, shift+size1)
        #     out = np.multiply(signal1[:end-start], signal2[start:end])
        # else:
        #     end = min(size1, shift+size2)
        #     out = np.multiply(signal2[:end-start], signal1[start:end])
        return out, sum(out)


class SensorGroup(object):
    def __init__(self, id, signal, sensors_list=[], accuracy_parameter=1):
        self.id = id
        self.sensor = Sensor(id=None, location=None, signal=signal, noise_floor=0, accuracy_parameter=accuracy_parameter)
        self.sensors_list = sensors_list  # a list that holds sensors id belonging to the group
        self.is_exist = True
        self.observation_vector = None

    def append_sensor(self, id):
        self.sensors_list.append(id)

    def append_group(self, list_sensor):
        self.sensors_list += list_sensor
        # self.group_list = set(self.group_list)
        self.sensors_list = list(set(self.sensors_list))  # omit duplicates
