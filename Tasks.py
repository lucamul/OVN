import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Network_Custom.Elements import Network, LightPath

if __name__ == "__main__":
    weighted_path_df = False
    comparison_latency_snr = False
    comparison_transceiver_types = False
    traffic_matrix_run = True
    if weighted_path_df:
        network = Network("/home/luca/256330.json")
        network.connect()
        network.draw()
        pairs = []
        # select every pair of nodes
        for node1 in list(network.nodes.keys()):
            for node2 in list(network.nodes.keys()):
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
            for path in network.find_paths(pair[0], pair[1]):
                tmp_str = ""
                for node in path:
                    tmp_str += node + "->"
                paths.append(path)

                # propagate to find snr, latency and noise for each path
                info = LightPath(1, path, 1)
                info = network.propagate(info)
                latencies.append(info.latency)
                noises.append(info.noise_power)
                snrs.append(10 * np.log10(info.signal_power / info.noise_power))
        df["path"] = paths
        df["latency"] = latencies
        df["noise"] = noises
        df["snr"] = snrs
        print(df)
    if comparison_latency_snr:
        M = 1
        network = Network("/home/luca/256330.json")
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix)
        # save every latency of the connections in a list to then plot it
        latencies = []
        snrs = []
        average_latency = 0
        average_snr = 0
        length = 0
        for connection in streamed_connections:
            if connection.latency:
                length += 1
                latencies.append(connection.latency)
                snrs.append(connection.snr)
                average_latency += connection.latency
                average_snr += connection.snr
        average_latency /= length
        average_snr /= length
        print("Average latency (Best Latency) = " + str(average_latency))
        print("Average SNR (Best Latency) = " + str(average_snr))
        # plot an histogram, bins is the number of equal sized bins
        plt.hist(latencies, bins=10)
        plt.title("Best latency, Latency Distribution")
        plt.show()
        plt.hist(snrs, bins=10)
        plt.title("Best latency, SNR Distribution")
        plt.show()
        # the same as before but for snr
        M = 1
        network = Network("/home/luca/256330.json")
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix, best="snr")
        snrs = []
        latencies = []
        length = 0
        for connection in streamed_connections:
            if connection.latency:
                length += 1
                latencies.append(connection.latency)
                snrs.append(connection.snr)
                average_latency += connection.latency
                average_snr += connection.snr
        average_latency /= length
        average_snr /= length
        print("Average latency (Best SNR) = " + str(average_latency))
        print("Average SNR (Best SNR) = " + str(average_snr))
        plt.hist(latencies, bins=10)
        plt.title("Best SNR, Latency Distribution")
        plt.show()
        plt.hist(snrs, bins=10)
        plt.title("Best SNR, SNR Distribution")
        plt.show()
    if comparison_transceiver_types:
        # FIXED
        network = Network("/home/luca/256330.json")
        M = 1
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix, best="snr")
        snrs = []
        bit_rates = []
        average_value = 0
        average_count = 0
        average_snr = 0
        # append the value of every snr and bit rate of every connection
        for connection in streamed_connections:
            snrs.append(connection.snr)
            bit_rates.append(connection.bit_rate)
            average_value += connection.bit_rate
            average_count += 1
            average_snr += connection.snr
        average_snr /= len(streamed_connections)
        print("Average SNR fixed = " + str(average_snr))
        # calculate the average bit rate and capacity
        average_value = average_value / average_count
        print("Average Rb fixed transceiver: " + str(average_value))
        print("Total capacity fixed transceiver: " + str(average_value * 10))
        # plot snr
        plot1 = plt.figure(1)
        plt.hist(snrs, bins=10)
        plt.title("SNR Distribution fixed-transceiver")
        # plot bit rate
        plot2 = plt.figure(2)
        plt.hist(bit_rates, bins=10)
        plt.title("bit rate Distribution fixed-transceiver")
        # FLEX
        network = Network("/home/luca/256330_flex.json")
        M = 1
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix, best="snr")
        snrs = []
        bit_rates = []
        average_value = 0
        average_count = 0
        average_snr = 0
        for connection in streamed_connections:
            snrs.append(connection.snr)
            bit_rates.append(connection.bit_rate)
            average_value += connection.bit_rate
            average_count += 1
            average_snr += connection.snr
        average_snr /= len(streamed_connections)
        print("Average SNR flex = " + str(average_snr))
        average_value = average_value / average_count
        print("Average Rb flex transceiver: " + str(average_value))
        print("Total capacity flex transceiver: " + str(average_value * 10))
        plot3 = plt.figure(3)
        plt.hist(snrs, bins=10)
        plt.title("SNR Distribution flex-transceiver")
        plot4 = plt.figure(4)
        plt.hist(bit_rates, bins=10)
        plt.title("bit rate Distribution flex-transceiver")
        # SHANNON
        network = Network("/home/luca/256330_shannon.json")
        M = 1
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix, best="snr")
        snrs = []
        bit_rates = []
        average_value = 0
        average_count = 0
        average_snr = 0
        for connection in streamed_connections:
            snrs.append(connection.snr)
            bit_rates.append(connection.bit_rate)
            average_value += connection.bit_rate
            average_count += 1
            average_snr += connection.snr
        average_snr /= len(streamed_connections)
        print("Average shannon fixed = " + str(average_snr))
        average_value = average_value / average_count
        print("Average Rb shannon transceiver: " + str(average_value))
        print("Total capacity shannon transceiver: " + str(average_value * 10))
        plot5 = plt.figure(5)
        plt.hist(snrs, bins=10)
        plt.title("SNR Distribution shannon transceiver")
        plot6 = plt.figure(6)
        plt.hist(bit_rates, bins=10)
        plt.title("bit rate Distribution shannon transceiver")
        # show all the plots at once so that you can compare
        plt.show()
    if traffic_matrix_run:
        M = 1
        network = Network("/home/luca/256330.json")
        network.connect()
        labels = list(network.nodes.keys())
        matrix = {}
        for label in labels:
            columns = {}
            for label1 in labels:
                if label == label1:
                    columns[label1] = 0
                else:
                    columns[label1] = 100 * M
            matrix[label] = columns
        streamed_connections = network.traffic_matrix_stream(matrix, best="snr")
        snrs = []
        bit_rates = []
        average_value = 0
        avg_snr = 0
        average_count = 0
        for connection in streamed_connections:
            snrs.append(connection.snr)
            bit_rates.append(connection.bit_rate)
            average_value += connection.bit_rate
            avg_snr += connection.snr
            average_count += 1
        # calculate the average bit rate and capacity
        average_value = average_value / average_count
        avg_snr /= len(snrs)
        print("Average Rb fixed transceiver: " + str(average_value))
        print("Total capacity fixed transceiver: " + str(average_value * 10))
        print("Average GSNR fixed transceiver: " + str(avg_snr))
        # plot snr
        plot1 = plt.figure(1)
        plt.hist(snrs, bins=10)
        plt.title("SNR Distribution fixed-transceiver")
        # plot bit rate
        plot2 = plt.figure(2)
        plt.hist(bit_rates, bins=10)
        plt.title("bit rate Distribution fixed-transceiver")
        plt.show()
