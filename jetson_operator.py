import socket
import threading
from time import sleep

def send_capture_command():
    while True:
        input("Press Enter to send capture command to all clients: ")
        for i in range(10):
            print(10-i)
            sleep(1)
        for client in clients:
            client.sendall(b'capture')

def accept_clients():
    while True:
        client, addr = server.accept()
        clients.append(client)
        print(f'Connected by {addr}')

clients = []
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 65432))
server.listen()

threading.Thread(target=accept_clients, daemon=True).start()

send_capture_command()
