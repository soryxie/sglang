def predict_model_ttft(prompt_length):
    # Placeholder: time to first token depends on prompt length
    return 50 + 2 * prompt_length

def predict_model_tpot(prompt_length):
    # Placeholder: constant time per output token
    return 10 * prompt_length
