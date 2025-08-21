from collections import deque
from predictor_interface import predict_model_ttft, predict_model_tpot

class Simulator:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def run(self, tasks):
        tasks_sorted = sorted(tasks, key=lambda x: x['arrival_time'])
        prefill_queue = deque()
        decode_queue = deque()
        current_time = 0.0
        i = 0
        if i < len(tasks_sorted):
            current_time = tasks_sorted[i]['arrival_time']
        while True:
            while i < len(tasks_sorted) and tasks_sorted[i]['arrival_time'] <= current_time:
                prefill_queue.append(tasks_sorted[i])
                i += 1
            if prefill_queue:
                batch = self.scheduler.select_prefill_batch(prefill_queue)
                for task in batch:
                    task['start_time'] = current_time
                batch_time = predict_model_ttft(sum(task['prompt_length'] for task in batch))
                finish_time = current_time + batch_time
                for task in batch:
                    remaining = task['output_length'] - 1
                    task['remaining_tokens'] = remaining
                    if remaining == 0:
                        task['completion_time'] = finish_time
                    else:
                        decode_queue.append(task)
                current_time = finish_time
            elif decode_queue:
                batch = self.scheduler.select_decode_batch(decode_queue)
                decode_time = predict_model_tpot(sum(task['remaining_tokens'] for task in batch))
                finish_time = current_time + decode_time
                for task in batch:
                    if 'decode_start_time' not in task:
                        task['decode_start_time'] = current_time
                    task['remaining_tokens'] -= 1
                    if task['remaining_tokens'] == 0:
                        task['completion_time'] = finish_time
                    else:
                        decode_queue.append(task)
                current_time = finish_time
            else:
                if i < len(tasks_sorted):
                    current_time = tasks_sorted[i]['arrival_time']
                    continue
                else:
                    break
        return tasks_sorted
