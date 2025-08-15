#!/usr/bin/env python3
import time
import signal
import atexit
import subprocess
import os, subprocess


OUTPUT_DIR = "/mnt/data/a100_verified"
INPUT_LEN = 1024
OUTPUT_LEN = 10
PORT = 30000
TP = 8
SERVER_ARGS = {
    "model_path": "Qwen/Qwen2.5-32B",
    "tp": 8,
    "trust_remote_code": True,
    "dtype": "auto",
    "kv_cache_dtype": "auto",
    "context_length": None,
    "device": "cuda",
    "mem_fraction_static": 0.5,
    "max_running_requests": None,
    "max_total_tokens": None,
    "chunked_prefill_size": 131072,
    "max_prefill_tokens": 16384,
    "schedule_policy": "fcfs",
    "schedule_conservativeness": 1.0,
    "disable_radix_cache": True,
    "cuda_graph_max_bs": None,
    "disable_cuda_graph": False,
    "disable_custom_all_reduce": False,
    "enable_mscclpp": False,
    "enable_torch_compile": False,
    "torch_compile_max_bs": 32,
}

SERVER = None
BENCH = None

def launch_server(concurrency, num_prompts, log_file):
    global SERVER
    env = os.environ.copy()
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model", SERVER_ARGS["model_path"],
        "--tp", str(TP),
        "--port", str(PORT),
        "--disable-radix-cache",
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--chunked-prefill-size", "131072",
        "--enable-torch-compile",
        "--torch-compile-max-bs", str(concurrency),
        "--enable-metrics",
    ]
    with open(log_file, "w") as log_file:
        SERVER = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)

def run_bench_serving(concurrency, num_prompts, short_num, output_file, log_file):
    global BENCH
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--output-file", output_file,
        "--num-prompts", str(num_prompts),
        "--dataset-name", "random",
        "--random-input", str(INPUT_LEN),
        "--random-short-num", str(OUTPUT_LEN),
        "--random-range-ratio", "0.3",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-concurrency", str(concurrency),
        "--output-details",
    ]
    with open(log_file, "w") as log_file:
        BENCH = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

def wait_for_server_to_be_ready(url: str, max_retries: int = 3600, delay: int = 1):
    retries = 0
    while retries < max_retries:
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                check=True,
                text=True,
                capture_output=True
            )

            if result.stdout.strip() == "200":
                print("Server is ready (200 OK).")
                break
        except subprocess.CalledProcessError:
            pass  
        retries += 1
        time.sleep(delay)
    else:
        print("Max retries reached, server is not responding with 200 OK.")

def wait_for_benchmark_done(output_file: str, current_len: int, max_retries: int = 3600, delay: int = 1):
    retries = 0
    while retries < max_retries:
        try:
            with open(output_file, "r") as f:
                lines = f.readlines()
                if len(lines) > current_len:
                    return True
        except FileNotFoundError:
            pass  
        retries += 1
        time.sleep(delay)
    print("Max retries reached, benchmark did not complete.")
    return False

def main():
    global OUTPUT_LEN
    global INPUT_LEN

    for concurrency in [16, 32, 64]:
        num_prompts = concurrency
        for input_len in [512, 1024, 2048, 4096]:
            INPUT_LEN = input_len
            log_file_prefix = os.path.join(OUTPUT_DIR, 
                f"TP_{TP}_CONC_{concurrency}_NUM_{num_prompts}_IN_{INPUT_LEN}_OUT_{OUTPUT_LEN}")

            server_log_file = log_file_prefix + "_server.log"
            print("Starting server...")
            launch_server(concurrency, num_prompts, server_log_file)
            wait_for_server_to_be_ready(f"http://localhost:{PORT}/metrics")
            print("Server ready. Starting benchmark...")

            for short_num in [0, 4, 8, 12, 16, 20, 24, 28]:
                output_file = log_file_prefix + f"_shortn_{short_num}.jsonl"
                for i in range(10):
                    client_log_file = log_file_prefix + f"_{i}_bench.log"
                    run_bench_serving(concurrency, num_prompts, short_num, output_file, client_log_file)
                    if wait_for_benchmark_done(output_file, i):
                        print(f"Benchmark {i} completed successfully.")
            
            print("All benchmarks completed. Cleaning up...")
            cleanup()

# while True:
#     with open(os.path.join(OUTPUT_DIR, f"TP_{tp}_CONC_{concurrency}_NUM_{num_prompts}_IN_{INPUT_LEN}_OUT_{OUTPUT_LEN}_server.log"), "r") as f:
#         raw = f.read()
#         if "Application startup complete." in raw:
#             break
#     time.sleep(1)

def cleanup():
    if SERVER:
        print("Stopping server...")
        SERVER.terminate()
        SERVER.wait()
    if BENCH:
        print("Stopping benchmark...")
        BENCH.terminate()
        BENCH.wait()
    print("Cleanup completed.")

def signal_handler(signum, frame):
    print(f"Received signal {signum}, cleaning up...")
    cleanup()
    exit(0)

if __name__ == "__main__":
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
