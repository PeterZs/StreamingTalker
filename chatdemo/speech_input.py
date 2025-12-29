import socket
import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please talk...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_vosk(audio, language="en-US")
        print(f"Recognized test: {text}")
        return text
    except sr.UnknownValueError:
        print("Unable to recognize")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None

def send_to_server(text, server_ip, server_port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((server_ip, server_port))
            client_socket.sendall(text.encode('utf-8'))
            print(f"Send success: {text}")
        except Exception as e:
            print(f"Send fail: {e}")

if __name__ == "__main__":
    SERVER_IP = "10.76.2.109"
    SERVER_PORT = 12543
    text = recognize_speech()
    if text:
        send_to_server(text, SERVER_IP, SERVER_PORT)
