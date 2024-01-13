import socket
import threading
from time import sleep

def send_capture_command():
    while True:
        input("Press Enter to send capture command to all clients: \n")

        for client in clients:
            client.sendall(b'capture')
        sleep(1)
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
