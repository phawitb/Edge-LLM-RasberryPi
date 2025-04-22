import sys
import socket
import threading
import torch
import io
from transformers import AutoModelForCausalLM
from shard_utils import split_model

def handle_client(conn, addr, model_parts, device_id):
    # print(f"[Device {device_id}] Connected by {addr}")
    while True:
        size_bytes = conn.recv(4)
        if not size_bytes:
            break
        expected_size = int.from_bytes(size_bytes, 'big')

        data = b''
        while len(data) < expected_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if not data:
            break

        buffer = io.BytesIO(data)
        input_data = torch.load(buffer, map_location='cpu')

        with torch.no_grad():
            if device_id == 0:
                input_ids = input_data
                position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
                inputs_embeds = model_parts['wte'](input_ids) + model_parts['wpe'](position_ids)
                hidden_states = model_parts['drop'](inputs_embeds)
                for block in model_parts['blocks']:
                    hidden_states = block(hidden_states)[0]
            elif device_id in [1, 2, 3]:
                hidden_states = input_data
                for block in model_parts['blocks']:
                    hidden_states = block(hidden_states)[0]
            elif device_id == 4:
                hidden_states = input_data
                for block in model_parts['blocks']:
                    hidden_states = block(hidden_states)[0]
                hidden_states = model_parts['ln_f'](hidden_states)
                hidden_states = model_parts['lm_head'](hidden_states)

        out_buffer = io.BytesIO()
        torch.save(hidden_states, out_buffer)
        response_data = out_buffer.getvalue()
        conn.sendall(len(response_data).to_bytes(4, 'big'))
        conn.sendall(response_data)

    conn.close()

def start_server(device_id, bind_ip="localhost", bind_port=None):
    print(f"[Device {device_id}] Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    model_parts = {}
    if device_id == 0:
        wte, wpe, drop, blocks = split_model(model, mode="device0")
        model_parts.update({'wte': wte, 'wpe': wpe, 'drop': drop, 'blocks': blocks})
    elif device_id == 1:
        blocks = split_model(model, mode="device1")
        model_parts['blocks'] = blocks
    elif device_id == 2:
        blocks = split_model(model, mode="device2")
        model_parts['blocks'] = blocks
    elif device_id == 3:
        blocks = split_model(model, mode="device3")
        model_parts['blocks'] = blocks
    elif device_id == 4:
        blocks, ln_f, lm_head = split_model(model, mode="device4")
        model_parts.update({'blocks': blocks, 'ln_f': ln_f, 'lm_head': lm_head})
    else:
        raise ValueError("Invalid device id")

    print(f"[Device {device_id}] Ready.")
    if bind_port is None:
        bind_port = 9000 + device_id

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((bind_ip, bind_port))
    server.listen()

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr, model_parts, device_id))
        thread.start()

if __name__ == "__main__":
    device_id = int(sys.argv[1])
    bind_ip = sys.argv[2]
    bind_port = int(sys.argv[3])
    start_server(device_id, bind_ip, bind_port)
