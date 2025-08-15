# generate list of input lengths for benchmarking
import numpy as np
import time

def sample_input_lengths(num_prompts: int, total_tokens: int, min_len: int = 1, ):
    eps: float = 1e-6
    assert min_len >= 1, "min_len must be >= 1"
    base = np.full(num_prompts, min_len)
    remaining_tokens = total_tokens - base.sum()
    if remaining_tokens < 0:
        raise ValueError(f"min_len={min_len} 太大，无法满足总 token 数为 {total_tokens}")

    np.random.seed(int(time.time() * 1000) % (2**32 - 1))
    weights = np.random.rand(num_prompts)
    weights /= (weights.sum() + eps)  
    floats = weights * remaining_tokens
    extra = np.floor(floats).astype(int)

    diff = remaining_tokens - extra.sum()
    for i in range(int(diff)):
        extra[i % num_prompts] += 1

    input_lens = base + extra
    return input_lens.astype(int)

def gen_len(num_prompts, num_short_len, avg_input_len, short=50):
    short_len = [short for _ in range(num_short_len)]

    remaining_len = num_prompts * avg_input_len - num_short_len * short
    num_long_len = num_prompts - num_short_len
    long_len = sample_input_lengths(
        num_prompts=num_long_len,
        total_tokens=remaining_len,
        min_len=int(avg_input_len * 0.2),
    )
    input_lens = np.concatenate([short_len, long_len])
    np.random.shuffle(input_lens)
    return input_lens


avg_input_len = 4096
num_prompts = 32
for num_short_len in [0, 4, 8, 12, 16, 20, 24, 28]:
    print(f"# avg_input_len={avg_input_len}, num_short_len={num_short_len}, num_prompts={num_prompts}")
    input_lens = gen_len(num_prompts, num_short_len, avg_input_len)
    # print as code
    print(f"input_lens = {input_lens.tolist()}")

