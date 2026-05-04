vocab_size: int = 50_257
context_length: int = 2_024
layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288


def embedding():
    out = vocab_size * d_model
    print(f"embedding:{out}")
    return out


def ln():
    out = d_model
    print(f"layerNorm:{out}")
    return out


def linear():
    out = d_model * vocab_size
    print(f"linear:{out}")
    return out


def transformer():
    out = 0
    out += 2 * d_model  # 2 RMSNorm
    out += 4 * d_model * d_model  # q/k/v/o proj
    out += 2 * d_model * d_ff  # w1 and w3
    out += d_ff * d_model  # w2
    return out


ans = 0
ans += embedding()
ans += ln()
ans += linear()
for _ in range(layers):
    ans += transformer()
print(ans)
