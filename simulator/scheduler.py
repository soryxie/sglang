class PrefillPriorityScheduler:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size

    def select_prefill_batch(self, prefill_queue):
        batch = []
        while prefill_queue and len(batch) < self.max_batch_size:
            batch.append(prefill_queue.popleft())
        return batch

    def select_decode_batch(self, decode_queue):
        batch = []
        while decode_queue and len(batch) < self.max_batch_size:
            batch.append(decode_queue.popleft())
        return batch
