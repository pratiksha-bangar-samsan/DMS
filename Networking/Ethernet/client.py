# save as client.py
'''
import socket
import os

file_path = '/home/pratiksha/documents/DMS/recording.py'

s = socket.socket()
s.connect(('192.168.1.92', 5001))  # Jetson IP

with open('file_to_send.txt', 'rb') as f:
    data = f.read(1024)
    while data:
        s.send(data)
        data = f.read(1024)
s.close()
print("File sent.")


'''


# # client.py
# import socket
# import os

# # Path to the file you want to send
# file_path = 'r./ecording.py'

# # Jetson Nano IP address
# # jetson_ip = '192.168.1.92'  # <-- Replace this with Jetson's actual IP
# jetson_ip = '192.168.1.90'

# s = socket.socket()
# s.connect((jetson_ip, 5001))

# with open(file_path, 'rb') as f:
#     print(f"Sending {file_path}...")
#     while True:
#         data = f.read(1024)
#         if not data:
#             break
#         s.send(data)

# s.close()
# print("File sent successfully.")

# print("Readable?", os.access('recording.py', os.R_OK))

import socket
import os

file_path = './recording.py'
jetson_ip = '192.168.1.92'  # Replace with actual IP

print("Current directory:", os.getcwd())
print("File exists:", os.path.isfile(file_path))
print("Readable by Python:", os.access(file_path, os.R_OK))

try:
    with open(file_path, 'rb') as f:
        print(f"Successfully opened {file_path}")
        s = socket.socket()
        s.connect((jetson_ip, 5001))

        while True:
            data = f.read(1024)
            if not data:
                break
            s.send(data)

        s.close()
        print("File sent successfully.")
except Exception as e:
    print("❌ ERROR:", e)


