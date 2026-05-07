vocab_size = 10000
context_length = 256
num_layers = 4
d_model = 512
d_ff = 1344
num_heads = 16


DTYPE_SIZE = 2  # float32
ADAMW_PARAM = 2 * 4  # size of m and v for AdamW. m and v are single precision


def rms_norm(context_length, d_model):
    weight_size = 4 * d_model
    activation_size = context_length * d_model
    return activation_size, weight_size


def embedding(vocab_size, context_length, d_model):
    weight_size = 4 * vocab_size * d_model
    activation_size = context_length * d_model
    return activation_size, weight_size


def linear(context_length, vocab_size, d_model):
    weight_size = 4 * vocab_size * d_model
    activation_size = context_length * vocab_size
    return activation_size, weight_size


def attn(num_heads, context_length, d_model):
    # 4 proj q, k, v, o respectly.
    # 4 params for each weight in matrix weight, grad, m and v.
    # matrix size is d_model * d_model
    weight_size = 1 * 4 * d_model
    weight_size += 4 * 4 * d_model * d_model  # 4 proj matrix
    activation_size = context_length * d_model  # RMSNorm
    activation_size += 5 * context_length * d_model  # 4 proj, 1 score@V
    activation_size += 2 * num_heads * context_length * context_length  # Q@K and softmax
    return activation_size, weight_size


def ffn(d_model, d_ff):
    weight_size = 3 * 4 * d_model * d_ff
    weight_size += 1 * 4 * d_model  # RMSNorm
    activation_size = context_length * d_model  # RMSNorm
    activation_size += 4 * context_length * d_ff  # w1, w3, gate * down and SiLU
    activation_size += context_length * d_model  # w2
    return activation_size, weight_size


def transformer_language_model():
    a_emb, w_emb = embedding(vocab_size, context_length, d_model)
    a_attn, w_attn = attn(num_heads, context_length, d_model)
    a_ffn, w_ffn = ffn(d_model, d_ff)
    a_norm, w_norm = rms_norm(context_length, d_model)
    a_lm, w_lm = linear(context_length, vocab_size, d_model)
    weight_size = w_emb + num_layers * (w_attn + w_ffn) + w_norm + w_lm
    activation_size = a_emb + num_layers * (a_attn + a_ffn) + a_norm + a_lm
    gb = 1000**3
    print(f"final linear {4 * w_lm / 16 / gb: .2f}")
    print(f"({4 * activation_size / gb: .2f} * BatchSize + {4 * weight_size / gb: .2f} ) GB")
    print(f"param size {4 * weight_size / 16 / gb: .2f}")


if __name__ == "__main__":
    transformer_language_model()
