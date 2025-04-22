from transformers import GPT2LMHeadModel

def split_model(model, mode="device0"):
    if not isinstance(model, GPT2LMHeadModel):
        raise ValueError("Model must be GPT2LMHeadModel.")

    wte = model.transformer.wte
    wpe = model.transformer.wpe
    drop = model.transformer.drop
    blocks = model.transformer.h
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    if mode == "device0":
        return wte, wpe, drop, blocks[0:2]
    elif mode == "device1":
        return blocks[2:4]
    elif mode == "device2":
        return blocks[4:6]
    elif mode == "device3":
        return blocks[6:8]
    elif mode == "device4":
        return blocks[8:12], ln_f, lm_head
    else:
        raise ValueError(f"Unknown split mode: {mode}")