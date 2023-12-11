import socket
import threading
import os

def receive_and_save_image(client_socket):
    while True:
        filename = ""
        while True:
            chunk = client_socket.recv(1024)
            filename += chunk.decode('utf-8', 'ignore')
            if '\n' in filename:
                filename = filename.split('\n')[0]
                break

        if not os.path.exists('images'):
            os.makedirs('images')

        with open(os.path.join('images', filename), 'wb') as file:
            while True:
                data = client_socket.recv(4096)
                if b'ENDOFIMAGE' in data:
                    end_index = data.index(b'ENDOFIMAGE')
                    file.write(data[:end_index])
                    break
                file.write(data)

def send_capture_command():
    while True:
        input("Press Enter to send capture : ")
        for client in clients:
            client.sendall(b'capture')

def accept_clients():
    while True:
        client, addr = server.accept()
        clients.append(client)
        print(f'Connected by {addr}')
        threading.Thread(target=receive_and_save_image, args=(client,), daemon=True).start()

clients = []
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 65432))
server.listen()

threading.Thread(target=accept_clients, daemon=True).start()
send_capture_command()
