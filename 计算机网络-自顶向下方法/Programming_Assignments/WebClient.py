from socket import *
import sys

clientSocket = socket(AF_INET, SOCK_STREAM)

if len(sys.argv) != 4:
    print('Format: client.py server_host server_port filename')
    sys.exit(1)


host = sys.argv[1]
port = int(sys.argv[2])
filename = sys.argv[3]


def generateRequest() -> str:
    request = f"GET {filename} HTTP/1.1\r\n"
    request += f"Host: {host}\r\n"
    request += "Connection: close\r\n\r\n"
    print(request)
    return request


clientSocket.connect((host, port))
clientSocket.send(generateRequest().encode())
# 接收完整的响应
response = b''
while True:
    recv = clientSocket.recv(1024)
    if len(recv) == 0:
        break
    response += recv
print(response.decode())
