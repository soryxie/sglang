from workload_loader import generate_workload
from scheduler import PrefillPriorityScheduler
from simulator import Simulator
import metrics
import visualizer

if __name__ == "__main__":
    tasks = generate_workload(num_requests=100)
    scheduler = PrefillPriorityScheduler(max_batch_size=8)
    simulator = Simulator(scheduler)
    tasks_log = simulator.run(tasks)
    metrics_data = metrics.compute_metrics(tasks_log)
    print(f"Average Throughput (tasks/sec): {metrics_data['throughput']:.2f}")
    print(f"Average Latency (sec): {metrics_data['avg_latency']:.2f}")
    print(f"95th Percentile Latency (sec): {metrics_data['p95_latency']:.2f}")
    print(f"Average Queue Delay (sec): {metrics_data['avg_queue_delay']:.2f}")
    throughput_series = metrics.throughput_over_time(tasks_log, interval=1.0)
    visualizer.plot_throughput_over_time(throughput_series)
    visualizer.plot_latency_distribution(metrics_data['latencies'])
    visualizer.plot_queue_delay_distribution(metrics_data['queue_delays'])
    print("Simulation complete. Results saved to throughput_over_time.png, latency_distribution.png, queue_delay_distribution.png.")
