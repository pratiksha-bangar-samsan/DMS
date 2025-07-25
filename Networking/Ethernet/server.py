import socket
import os
print("Readable?", os.access('r./ecording.py', os.R_OK))


# Save the file in the same folder as this script
save_path = './received_recording.py'

s = socket.socket()
s.bind(('0.0.0.0', 5001))  # Listen on all interfaces
s.listen(1)
print("Waiting for connection...")

conn, addr = s.accept()
print(f"Connected by {addr}")

with open(save_path, 'wb') as f:
    print("Receiving file...")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        f.write(data)

conn.close()
print(f"File received and saved to {save_path}")
