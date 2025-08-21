def compute_metrics(tasks_log):
    n = len(tasks_log)
    if n == 0:
        return {}
    first_arrival = min(t['arrival_time'] for t in tasks_log)
    last_completion = max(t['completion_time'] for t in tasks_log)
    total_time = last_completion - first_arrival
    throughput = n / total_time if total_time > 0 else 0
    latencies = [t['completion_time'] - t['arrival_time'] for t in tasks_log]
    latencies.sort()
    avg_latency = sum(latencies) / n
    p95_latency = latencies[int(0.95 * (n - 1))]
    queue_delays = [t['start_time'] - t['arrival_time'] for t in tasks_log]
    avg_queue_delay = sum(queue_delays) / n
    return {
        'throughput': throughput,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'avg_queue_delay': avg_queue_delay,
        'latencies': latencies,
        'queue_delays': queue_delays
    }

def throughput_over_time(tasks_log, interval=1.0):
    if not tasks_log:
        return []
    comp_times = [t['completion_time'] for t in tasks_log]
    comp_times.sort()
    series = []
    current = 0.0
    end_time = comp_times[-1]
    i = 0
    while current < end_time:
        window_end = current + interval
        if window_end > end_time:
            window_end = end_time
        count = 0
        while i < len(comp_times) and comp_times[i] < window_end:
            count += 1
            i += 1
        actual_interval = window_end - current
        throughput_val = count / actual_interval if actual_interval > 0 else 0
        series.append((current, throughput_val))
        current = window_end
    return series
