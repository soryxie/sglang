import matplotlib.pyplot as plt

def plot_throughput_over_time(series):
    times = [point[0] for point in series]
    values = [point[1] for point in series]
    plt.figure()
    plt.plot(times, values, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (tasks/sec)')
    plt.title('Throughput Over Time')
    plt.tight_layout()
    plt.savefig('throughput_over_time.png')
    plt.close()

def plot_latency_distribution(latencies):
    plt.figure()
    plt.hist(latencies, bins=20)
    plt.xlabel('Latency (s)')
    plt.ylabel('Number of Requests')
    plt.title('Latency Distribution')
    plt.tight_layout()
    plt.savefig('latency_distribution.png')
    plt.close()

def plot_queue_delay_distribution(queue_delays):
    plt.figure()
    plt.hist(queue_delays, bins=20)
    plt.xlabel('Queue Delay (s)')
    plt.ylabel('Number of Requests')
    plt.title('Queue Delay Distribution')
    plt.tight_layout()
    plt.savefig('queue_delay_distribution.png')
    plt.close()
