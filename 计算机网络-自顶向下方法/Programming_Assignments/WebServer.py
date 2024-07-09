from socket import *

serverSocket = socket(AF_INET, SOCK_STREAM)
host = "127.0.0.1"
port = 7890
serverSocket.bind((host, port))
print(f'Listenning on {host} {port}')
serverSocket.listen(5)


def generateResponse(status_code: int, status_msg: str, content_type: str, content: str) -> str:
    response = f"HTTP/1.1 {status_code} {status_msg}\r\n"
    response += f"Content-Type: {content_type}\r\n"
    response += f"Content-Length: {len(content)}\r\n"
    response += "\r\n"
    response += content
    return response


while True:
    # Establish the connection
    connectionSocket, addr = serverSocket.accept()
    print(f"Got connection from {addr}...")
    try:
        message = connectionSocket.recv(1024)
        filename = message.split()[1]
        print(f'Receive {message}', filename)
        f = open(filename, encoding='utf-8')
        outputdata = generateResponse(200, "OK", "text", f.read())
        f.close()
        connectionSocket.send(outputdata.encode())
        print(outputdata)
        connectionSocket.close()
    except OSError:
        # Send response message for file not found
        # Close client socket
        outputdata = generateResponse(404, "NOT FOUND", "text", "")
        connectionSocket.send(outputdata.encode())
        connectionSocket.close()
