import torch
import io
from transformers import AutoTokenizer
import time
import socket

def send_to_device(device_id, tensor):
    port = 9000 + device_id
    host = "localhost" if device_id < 4 else "192.168.1.44"  # <-- adjust Pi IP here

    print(f"Send to device >> [Device {device_id}] ({host}, {port})")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        out_buffer = io.BytesIO()
        torch.save(tensor, out_buffer)
        data = out_buffer.getvalue()

        s.sendall(len(data).to_bytes(4, 'big'))
        s.sendall(data)

        size_bytes = s.recv(4)
        expected_size = int.from_bytes(size_bytes, 'big')

        recv_data = b''
        while len(recv_data) < expected_size:
            packet = s.recv(min(4096, expected_size - len(recv_data)))
            if not packet:
                break
            recv_data += packet

        buffer = io.BytesIO(recv_data)
        output_tensor = torch.load(buffer, map_location='cpu')
    return output_tensor

def sample_next_token(logits, top_k=50):
    top_k = min(top_k, logits.size(-1))
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    top_probs = top_probs / torch.sum(top_probs, dim=-1, keepdim=True)
    next_token = torch.multinomial(top_probs, 1)
    next_token_id = top_indices.gather(-1, next_token)
    return next_token_id.squeeze(0)

if __name__ == "__main__":
    start_time = time.time()
    prompt = "Once upon a time"
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    max_new_tokens = 20
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        tensor = generated_ids
        for i in range(5):
            tensor = send_to_device(i, tensor)

        next_token = sample_next_token(tensor[:, -1, :], top_k=50).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        print('\n',tokenizer.decode(generated_ids[0], skip_special_tokens=True),'\n')

    print("\n=== Generated Text ===")
    print(f"[{time.time()-start_time} sec]", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

