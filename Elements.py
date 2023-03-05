import json
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c, h
from math import e, pi
from scipy.special import erfcinv
from random import shuffle

# set the number of channels per line:
number_of_channels = 10
# target Bit-Error-Rate:
BERt = 1e-3
# symbol rate (Baud)
Rs = 32e9
# noise bandwidth (Hz)
Bn = 12.5e9
# amplifier attributes (dB)
gain = 16
noise_figure = 3
# how often do we need an amplifier (m)
amplifier_distance = 80e3
# C-band center (Hz)
f = 193.414e12


class SignalInformation(object):
    def __init__(self, signal_power, path):
        self._noise_power = 0.0
        self._latency = 0.0
        self._signal_power = signal_power
        self._path = path

    # getters

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @property
    def noise_power(self):
        return self._noise_power

    @property
    def latency(self):
        return self._latency

    # setters

    @path.setter
    def path(self, path):
        self._path = path

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    # add the noise of a line to the noise experienced by the information

    def add_noise(self, noise):
        self._noise_power += noise

    # add the latency of a line to the latency of the information

    def add_latency(self, latency):
        self._latency += latency

    # update the path once one node is traversed

    def next(self):
        # deletes the first character of the path string thus moving to next node, e.g. ABCD -> BCD
        self._path = self.path[1:]


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict["label"]
        self._position = node_dict["position"]
        self._connected_nodes = node_dict["connected_nodes"]
        self._successive = {}
        self._transceiver = ""
        self._switching_matrix = None
        # check if the transceiver attribute is present within the node, else initialize to fixed rate
        if "transceiver" in node_dict.keys():
            if node_dict["transceiver"] in ["fixed-rate", "flex-rate", "shannon"]:
                self._transceiver = node_dict["transceiver"]
            else:
                print("Error: wrong value for transceiver. Value:", node_dict["transceiver"])
        else:
            self._transceiver = "fixed-rate"

    # getters

    @property
    def transceiver(self):
        return self._transceiver

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    # setters

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @switching_matrix.setter
    def switching_matrix(self, switching_matrix):
        self._switching_matrix = switching_matrix

    # propagate the signal from this node to the next line

    def propagate(self, signal_information, occupation=False):
        path = signal_information.path
        # if len=1 destination has been reached, return final info
        if len(path) > 1:
            # select the first two nodes of the path, i.e. the next line to traverse
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.signal_power = line.optimized_launch_power(signal_information)
            # update the path deleting the first character, i.e. move to the next node
            signal_information.next()
            # propagate the info on the line so that latency and noise can be updated
            signal_information = line.propagate(signal_information, occupation)
        return signal_information


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict["label"]
        self._length = line_dict["length"]
        self._successive = {}
        self._state = ["1"] * number_of_channels
        self._n_amplifiers = int(self.length / amplifier_distance)
        self._alpha_db = 0.2e-3
        self._beta2 = 2.13e-26
        self._gamma = 1.27e-3

    # getters

    @property
    def alpha_db(self):
        return self._alpha_db

    @property
    def beta2(self):
        return self._beta2

    @property
    def gamma(self):
        return self._gamma

    @property
    def state(self):
        return self._state

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    # setters

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @state.setter
    def state(self, state):
        for s in state:
            # turn the string to lower and remove all spaces at beginning and end
            s.strip()
            if s not in ["1", "0"]:
                print("Error: state not recognized. Value:", state)
                return
        self._state = state

    @n_amplifiers.setter
    def n_amplifiers(self, n):
        self._n_amplifiers = n

    # Total amount of Amplified Spontaneous Emissions ASE

    def ase_generation(self):
        # ASE = N(h*f*Bn*NF[G-1])
        ase = self.n_amplifiers * (h * f * Bn * pow(10, noise_figure / 20) * (pow(10, gain / 20) - 1))
        return ase

    # returns the value of the constant eta nli

    def get_eta_nli(self, light_path):
        alpha = self.alpha_db / (20 * np.log10(e))
        rs = light_path.symbol_rate
        delta_f = light_path.df
        return (16 / (27 * pi)) * np.log10(
            (pow(pi, 2) / 2) * np.abs(self.beta2) * pow(rs, 2) * pow(number_of_channels,
                                                                     2 * rs / delta_f) / alpha) * pow(self.gamma, 2) / (
                       4 * alpha * self.beta2 * pow(rs, 3))

    # calculate the non linear interference noise nli

    def nli_generation(self, light_path):
        # eta_nli = (16/(27pi))log[pi^2*|beta2|*Rs^2/alpha*Nch^(2Rs/df)]*gamma^2/(4alpha*beta3*Rs^3)
        eta_nli = self.get_eta_nli(light_path)
        nli = eta_nli * pow(light_path.signal_power, 3) * (self.n_amplifiers - 1) * Bn
        return nli

    # calculate the latency over the line

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    # calculate the noise over the line by summing the ASE and NLI power

    def noise_generation(self, light_path):
        noise = self.ase_generation() + self.nli_generation(light_path)
        return noise

    # find the optimum power to maximize GSNR:

    def optimized_launch_power(self, light_path):
        p_ase = self.ase_generation()
        eta_nli = self.get_eta_nli(light_path)
        n_span = self.n_amplifiers - 1
        return math.pow(p_ase / (2 * eta_nli * n_span * Bn), 1/3)

    # propagate the signal over the line

    def propagate(self, signal_information, occupation=False):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)
        # Update noise
        noise = self.noise_generation(signal_information)
        signal_information.add_noise(noise)
        # if the occupation flag is on, i.e. we need to track occupation, we set the channel of the line as busy
        if occupation:
            state = list(self.state)
            state[int(signal_information.channel)] = "0"
            self.state = state
        # find the next node in the path and propagate the info to it
        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information, occupation)
        return signal_information


class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, "r"))
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = None
        self._route_space = None
        self._connected = False
        for node_label in node_json:
            # create a new node
            node_dict = node_json[node_label]
            node_dict["label"] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # create a new line instance
            for connected_node_label in node_dict["connected_nodes"]:
                line_dict = {"label": node_label + connected_node_label}
                # calculating the length of a segment
                x2 = node_json[node_label]["position"][0]
                x1 = node_json[connected_node_label]["position"][0]
                y2 = node_json[node_label]["position"][1]
                y1 = node_json[connected_node_label]["position"][1]
                line_dict["length"] = np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
                line = Line(line_dict)
                self._lines[line.label] = line

    # getters

    @property
    def route_space(self):
        return self._route_space

    @property
    def connected(self):
        return self._connected

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    # setters

    @route_space.setter
    def route_space(self, route_space):
        self._route_space = route_space

    # calculate the bit rate given the used strategy (Gbps)

    def calculate_bit_rate(self, light_path, strategy):
        # convert snr to linear units
        path = ""
        length = len(light_path.path)
        for i in range(length):
            path += light_path.path[i]
            if i != (length - 1):
                path += "->"
        rs = light_path.symbol_rate
        gsnr_db = self.weighted_paths.loc[self.weighted_paths.path == path].snr.values[0]
        gsnr_linear = pow(10, gsnr_db / 10)
        # switch to the given strategy
        if strategy == "fixed-rate":
            threshold = 2 * (rs / Bn) * (erfcinv(2 * BERt) ** 2)
            if gsnr_linear >= threshold:
                # PM-QPSK
                return 100
            else:
                return 0
        elif strategy == "flex-rate":
            threshold1 = 2 * (rs / Bn) * (erfcinv(2 * BERt) ** 2)
            threshold2 = (14 / 13) * (rs / Bn) * (erfcinv((3 / 2) * BERt) ** 2)
            threshold3 = 10 * (rs / Bn) * (erfcinv((8 / 3) * BERt) ** 2)
            if gsnr_linear < threshold1:
                return 0
            elif gsnr_linear < threshold2:
                # PM-QPSK
                return 100
            elif gsnr_linear < threshold3:
                # PM-8-QAM
                return 200
            else:
                # PM-16-QAM
                return 400
        elif strategy == "shannon":
            return (2 * rs * math.log2(1 + gsnr_linear * Bn / rs)) * 10 ** (-9)
        else:
            print("Error, wrong strategy. Value:", strategy)

    # find all free paths for a given source and destination

    def available_paths(self, source, dest):
        if self.weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths = []
        # find all paths with the given source and destination
        for path in self.weighted_paths.path.values:
            if (path[0] == source) and (path[-1] == dest):
                all_paths.append(path)
        available_paths = []
        # for every candidate path check that there does not exist a line
        # within the path which is occupied, if there isn't add the path to available
        for path in all_paths:
            # find the row corresponding to the needed path
            # then transpose the df to get the values of the occupancy of the channels
            states = self.route_space.loc[self.route_space.path == path].T.values[1:]
            if "1" in states:
                available_paths.append(path)
        return available_paths

    # draw the network topology

    def draw(self):
        for node_label in self.nodes:
            n0 = self.nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            # plot the point x0,y0 for all nodes
            plt.plot(x0, y0, "go", markersize=10)
            plt.text(x0 + 50, y0 + 50, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = self.nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                # plot all edges coming out of n0
                plt.plot([x0, x1], [y0, y1], "b")
        plt.title("Network")
        plt.show()

    # find all paths connecting two nodes

    def find_paths(self, n1, n2):
        path_tmp = [n1]
        paths = []
        Network.find_paths_r(self, n1, n2, paths, path_tmp)
        return paths

    # static auxiliary method to find_paths

    @staticmethod
    def find_paths_r(network, n1, n2, paths, path_tmp):
        # if n1 == n2 the destination has been reached so the path can be added to the list of paths to n2
        if n1 == n2:
            path = ""
            for n in path_tmp:
                path += n
            paths.append(path)
            return
        # for every node connected to n1 try to follow the path through there to see if it gets to n2
        for n in network.nodes[n1].connected_nodes:
            if n not in path_tmp:
                path_tmp.append(n)
                Network.find_paths_r(network, n, n2, paths, path_tmp)
                path_tmp.remove(n)

        return

    # insert all successive lines in a node and all successive nodes in a line

    def connect(self):
        for node_label in self.nodes:
            switching_matrix = {}
            for node in list(self.nodes.keys()):
                # set the entry of the switching matrix for each connected node
                if node != node_label and (node_label in self.nodes[node].connected_nodes):
                    node_dict = {}
                    for node1 in self.nodes[node_label].connected_nodes:
                        if node1 != node_label:
                            if node == node1:
                                node_dict[node1] = np.zeros(number_of_channels)
                            else:
                                node_dict[node1] = np.ones(number_of_channels)
                    switching_matrix[node] = node_dict
            for connected_node in self.nodes[node_label].connected_nodes:
                line_label = node_label + connected_node
                # add a dictionary of node_label:node_object to every line containing the node at the end of the line
                self.lines[line_label].successive[connected_node] = self.nodes[connected_node]
                # add a dictionary of line_label:line_object to every node containing all lines from it
                self.nodes[node_label].successive[line_label] = self.lines[line_label]
            self.nodes[node_label].switching_matrix = switching_matrix
        self._connected = True

    # propagate a signal through the network

    def propagate(self, signal_information, occupation=False):
        path = signal_information.path
        source = self.nodes[path[0]]
        return source.propagate(signal_information, occupation)

    # propagate a signal of signal_power and save for every path the weight as snr and latency

    def set_weighted_paths(self, signal_power):
        if not self.connected:
            self.connect()
        pairs = []
        # select every pair of nodes
        for node1 in list(self.nodes.keys()):
            for node2 in list(self.nodes.keys()):
                if node1 != node2:
                    pairs.append(node1 + node2)
        # build the panda dataframe
        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in pairs:
            # find all paths for a given pair
            for path in self.find_paths(pair[0], pair[1]):
                tmp_str = ""
                for node in path:
                    tmp_str += node + "->"
                tmp_str = tmp_str[:int(len(tmp_str) - 2)]
                paths.append(tmp_str)
                # propagate to find snr, latency and noise for each path
                info = LightPath(signal_power, path, 1)
                info = self.propagate(info)
                latencies.append(info.latency)
                noises.append(info.noise_power)
                snrs.append(10 * np.log10(info.signal_power / info.noise_power))
        df["path"] = paths
        df["latency"] = latencies
        df["noise"] = noises
        df["snr"] = snrs
        self._weighted_paths = df

        # create the route space for the given weighted paths
        df1 = pd.DataFrame()
        df1["path"] = paths
        for channel_id in range(number_of_channels):
            # set channel_id to be free on every path
            df1[str(channel_id)] = ["1"] * len(paths)
        self.route_space = df1

    # find the path with the best snr between two nodes

    def find_best_snr(self, source, dest):
        available_paths = self.available_paths(source, dest)
        if not available_paths:
            return None
        # locate within all weighted paths those with the given source and destination and that are available
        paths_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
        # select the maximum value on the array of snr values provided
        best_snr = np.max(paths_df.snr.values)
        # locate the path with snr equal to the maximum value and replace the -> with "" to get the form ABCD
        best_path = paths_df.loc[paths_df.snr == best_snr].path.values[0]
        return best_path

    # find the path with the best latency between two nodes

    def find_best_latency(self, source, dest):
        available_paths = self.available_paths(source, dest)
        if not available_paths:
            return None
        # locate within all weighted paths those with the given source and destination and that are available
        paths_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
        # select the minimum value on the array of latency values provided
        best_latency = np.min(paths_df.latency.values)
        # locate the path with latency equal to the minimum value and replace the -> with "" to get the form ABCD
        best_path = paths_df.loc[paths_df.latency == best_latency].path.values[0]
        return best_path

    # manage traffic matrix streaming

    def traffic_matrix_stream(self, traffic_matrix, best="latency"):
        connections = []
        nodes = list(self.nodes.keys())
        finished = False
        while not finished:
            shuffle(nodes)
            source = nodes[0]
            dest = nodes[-1]
            if traffic_matrix[source][dest] > 0 or traffic_matrix[source][dest] == "inf":
                connection = Connection(source, dest, 1)
                path = ""
                if best == "latency":
                    path = self.find_best_latency(connection.input_node, connection.output_node)
                elif best == "snr":
                    path = self.find_best_snr(connection.input_node, connection.output_node)
                if path:
                    states = self.route_space.loc[self.route_space.path == path].T.values[1:]
                    channel = ""
                    for i in range(len(states)):
                        if states[i] == "1":
                            channel = str(i)
                            break
                    origin_node = self.nodes[path[0]]
                    p = path.replace("->", "")
                    info = LightPath(connection.signal_power, p, channel)
                    bit_rate = self.calculate_bit_rate(info, origin_node.transceiver)
                    connection.bit_rate = bit_rate
                    if traffic_matrix[source][dest] != "inf":
                        traffic_matrix[source][dest] -= bit_rate
                        if traffic_matrix[source][dest] < 0:
                            traffic_matrix[source][dest] = 0
                    if bit_rate != 0:
                        # propagate to find snr and latency
                        out_info = self.propagate(info, True)
                        connection.latency = out_info.latency
                        connection.snr = 10 * np.log10(out_info.signal_power / out_info.noise_power)
                        # update route space
                        self.route_space_update(path, channel)
                    else:
                        connection.latency = None
                        connection.snr = 0
                else:
                    connection.latency = None
                    connection.snr = 0
                connections.append(connection)
            # check if the matrix is empty
            if not finished:
                finished = True
                for n in nodes:
                    for n1 in nodes:
                        if traffic_matrix[n][n1] > 0:
                            finished = False
                            break
            if not finished:
                finished = True
                for node in list(self.nodes.keys()):
                    for node1 in list(self.nodes.keys()):
                        if traffic_matrix[node][node1] > 0 or traffic_matrix[source][dest] == "inf":
                            connection = Connection(node, node1, 1)
                            path = self.find_best_snr(connection.input_node, connection.output_node)
                            if path:
                                finished = False
        return connections

    # stream multiple connections through the network via the path with the best latency/snr and save the resulting data

    def stream(self, connections, best="latency"):
        streamed_connections = []
        for connection in connections:
            if best == "latency":
                path = self.find_best_latency(connection.input_node, connection.output_node)
            elif best == "snr":
                path = self.find_best_snr(connection.input_node, connection.output_node)
            else:
                print('ERROR: best input not recognized. Value:', best)
                continue
            if path:
                # if such path exists find a free channel to propagate:
                states = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = ""
                for i in range(len(states)):
                    if states[i] == "1":
                        channel = str(i)
                        break
                # find bit rate
                origin_node = self.nodes[path[0]]
                p = path.replace("->", "")
                info = LightPath(connection.signal_power, p, channel)
                bit_rate = self.calculate_bit_rate(info, origin_node.transceiver)
                connection.bit_rate = bit_rate
                if bit_rate == 0:
                    # connection rejected
                    print("connection rejected, path:", path)
                else:
                    # propagate to find snr and latency
                    out_info = self.propagate(info, True)
                    connection.latency = out_info.latency
                    connection.snr = 10 * np.log10(out_info.signal_power / out_info.noise_power)
                    # update route space
                    self.route_space_update(path, channel)
            else:
                connection.latency = None
                connection.snr = 0
            streamed_connections.append(connection)
        return streamed_connections

    # update a route space whenever a signal is propagated through the network on a given channel in a path

    def route_space_update(self, path, channel):
        # save in states_path all states related to the column of the channel
        states_path = self.route_space[str(channel)]
        lines_path = []
        path_standard_notation = path.replace("->", "")
        # generate a set of all lines of the given path
        for i in range(len(path_standard_notation) - 1):
            lines_path.append(path_standard_notation[i] + path_standard_notation[i + 1])
        lines_path_set = set(lines_path)
        for j, p in enumerate(self.route_space.path.values):
            p_standard = p.replace("->", "")
            # for every path find all lines, if there is an intersection , i.e. it shares one or more lines with path
            # then state the channel state corresponding to that path to occupied
            # i.e. 1 line occupied in a channel on a path -> that channel cannot be accessed
            lines_list = []
            for i in range(len(p_standard) - 1):
                lines_list.append(p_standard[i] + p_standard[i + 1])
            lines = set(lines_list)
            if lines.intersection(lines_path_set):
                result = np.ones(number_of_channels)
                # update the route space by multiplying by the switching matrix of all nodes of the path
                for i in range(1, len(p_standard) - 1):
                    # for each node except the first and last find the previous (origin) and following (arrival)
                    origin_node = self.nodes[p_standard[i - 1]]
                    arrival_node = self.nodes[p_standard[i + 1]]
                    # find the vector of channels corresponding to the origin and arrival
                    switching_mat = self.nodes[p_standard[i]].switching_matrix[origin_node.label][
                        arrival_node.label]
                    # multiply the result vector by each of the switching matrix vector
                    result = np.multiply(result, switching_mat)
                for i in range(number_of_channels):
                    # update each channel of the path by multiplying
                    # its current value to the result of the switching matrix multiplication
                    route_space_channel_status = int(self.route_space[str(i)][j])
                    switching_matrix_channel_value = int(result[i])
                    self.route_space[str(i)][j] = str(route_space_channel_status * switching_matrix_channel_value)
                # set the current channel to busy
                states_path[j] = "0"
        self.route_space[str(channel)] = states_path


class Connection(object):
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0
        self._bit_rate = 0

    # getters

    @property
    def bit_rate(self):
        return self._bit_rate

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @property
    def snr(self):
        return self._snr

    # setters

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate


class LightPath(SignalInformation):
    def __init__(self, signal_power, path, channel):
        super().__init__(signal_power, path)
        self._channel = channel
        self._symbol_rate = Rs
        self._df = 50e9

    # getters

    @property
    def channel(self):
        return self._channel

    @property
    def symbol_rate(self):
        return self._symbol_rate

    @property
    def df(self):
        return self._df

    # setters

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @symbol_rate.setter
    def symbol_rate(self, rs):
        self._symbol_rate = rs

    @df.setter
    def df(self, df):
        self._df = df
