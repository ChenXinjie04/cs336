vocab_size = 50_257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4_288


def final_linear():
    out = 2 * context_length * d_model * vocab_size
    print(f"final_linear {out / 1e9:.2f}")
    return out


def transformer():
    proj = 4 * context_length * d_model * d_model
    print(f"proj {proj / 1e9:.2f}")
    atten = 2 * context_length * context_length * d_model  # q@k
    atten += 2 * context_length * context_length * d_model  # attention@v
    print(f"atten {atten / 1e9: .2f}")
    ff = 2 * 2 * context_length * d_model * d_ff
    ff += 2 * context_length * d_ff * d_model
    print(f"ff {ff / 1e9:.2f}")
    return proj + ff


print(f"transformer: {transformer() * num_layers / 1e9: .2f}")
final_linear()
