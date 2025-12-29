import os 
import librosa
import cv2
import threading
import socket
import torch
import pickle
import trimesh
import pyrender
import psutil
import numpy as np
import simpleaudio as sa

from gtts import gTTS
from tqdm import tqdm
from pydub import AudioSegment
from multiprocessing import Pool, Process, cpu_count, Queue, Value, Event

from time import time
from algorithms.models import get_model
from utils.demo_utils import render_mesh_helper, render_mesh_with_opengl, render_mesh_with_blender

try:
    from psbody.mesh import Mesh
except:
    Mesh = None

from transformers import AutoTokenizer, AutoModelForCausalLM

train_subjects = {
        'FaceTalk_170728_03272_TA': 0,
        'FaceTalk_170904_00128_TA': 1,
        'FaceTalk_170725_00137_TA': 2,
        'FaceTalk_170915_00223_TA': 3,
        'FaceTalk_170811_03274_TA': 4,
        'FaceTalk_170913_03279_TA': 5,
        'FaceTalk_170904_03276_TA': 6,
        'FaceTalk_170912_03278_TA': 7,
    }

def load_audio(processor, audio_path):        
        speech_array, sampling_rate = librosa.load(
                os.path.join(audio_path), 
                sr=16000
            )

        audio_feature = np.squeeze(
            processor(
                speech_array,
                sampling_rate = sampling_rate
            ).input_values
        )

        audio_feature = np.reshape(
            audio_feature,
            (-1,audio_feature.shape[0])
        )

        return torch.FloatTensor(audio_feature)

def convert_mp3_to_wav(mp3_path="tmp_audio.mp3", wav_path="output.wav", sample_rate=16000):
    audio = AudioSegment.from_mp3(mp3_path)
    
    audio = audio.set_frame_rate(sample_rate)
    
    audio.export(wav_path, format="wav")

def process_with_llm(llm, tokenizer, text, flag=None, device='cuda'):
    """
    Get answers from Large Language model
    """               
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = llm.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error while calling LLM: {e}")
        return "Sorry, I'm unable to process your request right now."

def socket_listener(args, cfg, host, port, audio_income_start, audio_income_end, vertices_list, flag=None, audio_path='tmp_audio.mp3'):
    '''
    Listen to socket connections and process received text
    '''
    pid = os.getpid()
    p = psutil.Process(pid)
    print(f"socket process{pid}")
    # p.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])

    template_file = args.template
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
        subject_id = args.id
        assert subject_id in template, f'{subject_id} is not a subject included'
        template = torch.Tensor(template[subject_id].reshape(-1))

    if subject_id in train_subjects:
        id_idx = train_subjects[subject_id]
        id = torch.zeros((1,args.id_dim))
        id[0, id_idx] = 1
    else:
        id = torch.zeros((1,args.id_dim))
        id[0, 0] = 1
    print("Template, Id loaded..")

    device = args.device
    model = get_model(cfg=cfg)
    state_dict = torch.load(args.checkpoint,pickle_module=torch.serialization.pickle,map_location="cpu")["state_dict"]
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    llm = AutoModelForCausalLM.from_pretrained("gpt2")
    llm.to(device)
    llm.eval()

    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    print("Model loaded..")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Listening {host}:{port}")
    
    while True:
        client_socket, client_address = server_socket.accept()
        with client_socket:
            print(f"Receive connections from {client_address}")
            data = client_socket.recv(1024)
            if data:
                # get text from client
                received_text = data.decode('utf-8')
                print(f"Received text:{received_text}")

                # get answer from LLM
                response = process_with_llm(llm, tokenizer, received_text, flag)
                print(f"Get response from LLM: {response}")

                # tts
                tts = gTTS(text=response, lang='en')
                tts.save(audio_path)
                print(f"Audio saved at {audio_path}")
                audio_income_start.value = True

                audio_income_end.value = True

                
                # load audio 1s
                start_time = time()
                # inference first, display when it's ready
                audio = load_audio(processor, audio_path)
                input_data = {
                    'id': id.to(device),
                    'audio': audio.to(device),
                    'template': template.to(device)
                }
                end_time = time()
                print(f"data time:{end_time-start_time}")

                
                model.streaming_inference(input_data, vertices_list)
                # audio_income_end.value = True
                
                print("vertices ready!!")

def render_frame(args, mesh, frames, vertices_list, render_inprocess, flag, template_type = "flame",rot=np.zeros(3), z_offset=0, rgb_per_v = None):
    pid = os.getpid()
    p = psutil.Process(pid)
    print(f"render process{pid}")
    # p.cpu_affinity([10, 11, 12, 13, 14, 15, 16, 17])

    assert template_type in ["flame", "biwi"], "template_type should be one of ['flame', 'biwi'],but got {}".format(template_type)


    if template_type == "flame":
        camera_params = {'c': np.array([400, 400]),
                            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                            'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
    elif template_type == "biwi":
        camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
        
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    intensity = 0.5
    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                # baseColorFactor=[110/255, 190/255, 220/255, 1.0], 
                baseColorFactor=[50/255, 120/255, 150/255, 1.0], 
                metallicFactor=0.0,
                roughnessFactor=0.9,
                emissiveFactor=[0.0, 0.0, 0.0]
            )

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                    fy=camera_params['f'][1],
                                    cx=camera_params['c'][0],
                                    cy=camera_params['c'][1],
                                    znear=frustum['near'],
                                    zfar=frustum['far'])
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])

    count = 0 
    while True:
        if not vertices_list.empty():
            render_inprocess.value = True

            vertice = vertices_list.get_nowait()
            center = np.mean(vertice, axis=0)
            render_mesh = Mesh(vertice, mesh.f)

            mesh_copy = Mesh(render_mesh.v, render_mesh.f)
            mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-center).T).T+center

            tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
            render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
            
            # scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
            scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[255, 255, 255])
            camera_pose = np.eye(4)
            camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
            scene.add(camera, pose=[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 1],
                                    [0, 0, 0, 1]])
            scene.add(render_mesh, pose=np.eye(4))

            angle = np.pi / 6.0
            pos = camera_pose[:3,3]
            light_color = np.array([1., 1., 1.])
            light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

            light_pose = np.eye(4)
            light_pose[:3,3] = pos
            scene.add(light, pose=light_pose.copy())
            
            light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
            scene.add(light, pose=light_pose.copy())

            light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
            scene.add(light, pose=light_pose.copy())

            light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
            scene.add(light, pose=light_pose.copy())

            light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
            scene.add(light, pose=light_pose.copy())

            color, _ = r.render(scene, flags=flags)
            pred_img = color[..., ::-1].astype(np.uint8)
            frames.put_nowait(pred_img)
            
            # print(f"rendering frame {count}")
            count+=1
            print(f"render frames size:{frames.qsize()}")

        render_inprocess.value = False

def video_output(frames, static_frame, output_start, audio_income_end, render_inprocess, flag, window_name = "real time demo", fps = 30):
    pid = os.getpid()
    p = psutil.Process(pid)
    print(f"video process{pid}")

    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    while True:
        try:
            if audio_income_end.value and not render_inprocess.value:
                print("show results")
                print(f"video frames size:{frames.qsize()}")
                while not frames.empty():
                    frame = frames.get_nowait()
                    output_start.value = True
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break
                audio_income_end.value = False
                
            # display inference results
            else:
                cv2.imshow(window_name, static_frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    print("Animation stopped.")

def audio_play(audio_income_start, output_start ,audio_path = 'output.wav'):
    """
    Play the audio file using simpleaudio.
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    print(f"audio process{pid}")
    
    while True:
        if audio_income_start.value:
            convert_mp3_to_wav()
            wave_obj = sa.WaveObject.from_wave_file(audio_path)
            print("audio convertion done")
            audio_income_start.value = False

            output_start.wait()
            
            print("play audio")
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until playback is finished

            output_start.clear()
            # if output_start.value:
            #     print("play audio")
            #     play_obj = wave_obj.play()
            #     play_obj.wait_done()  # Wait until playback is finished
            #     output_start.value = False

class ChatAvatarServer:
    def __init__(self,cfg = None, args = None):
        self.cfg = cfg
        self.args = args
        print("Server initialized..")

        self.mesh = Mesh(filename=args.ply)
        if "FLAME" in args.ply:
            self.mesh_type = "flame"
        elif "BIWI" in args.ply:
            self.mesh_type = "biwi"
        else:
            raise ValueError("Template type not recognized, please use either BIWI or FLAME")
        
        template_file = args.template
        with open(template_file, 'rb') as fin:
            template = pickle.load(fin,encoding='latin1')
            subject_id = args.id
            assert subject_id in template, f'{subject_id} is not a subject included'
            self.template = torch.Tensor(template[subject_id].reshape(-1))

        self.vertices_list = Queue(maxsize=800)
        self.render_inprocess = Value('b', False)
        self.audio_income_end = Value('b', False)
        self.audio_income_start = Value('b', False)
        # self.output_start = Value('b', False)
        self.output_start = Event()
        self.frames = Queue(maxsize=800)
        self.flag = Value('i', 0)
        print("Global variable set..")

    def start_render(
                    self,
                    host='192.168.1.101', 
                    port=6020, 
                    fps=30,  
                    window_name='real time demo',):
        pid = os.getpid()
        p = psutil.Process(pid)
        print(f"main process{pid}")
        # p.cpu_affinity([29, 30 ,31, 32])

        template_vertices = self.template[None, ...].numpy()
        template_vertices = template_vertices.reshape(-1, template_vertices.shape[1]//3, 3)
        static_center = np.mean(template_vertices[0], axis=0)
        static_mesh = Mesh(template_vertices[0], self.mesh.f)
        
        static_frame = render_mesh_helper(static_mesh, static_center, template_type=self.mesh_type)
        static_frame = static_frame.astype(np.uint8)

        socket_process = Process(target=socket_listener, args=(self.args, self.cfg, host, port, self.audio_income_start, self.audio_income_end, self.vertices_list, self.flag,), daemon=True)
        render_process = Process(target=render_frame, args=(self.args, self.mesh, self.frames, self.vertices_list, self.render_inprocess, self.flag), daemon=True)
        video_process = Process(target=video_output, args=(self.frames, static_frame, self.output_start, self.audio_income_end, self.render_inprocess, self.flag), daemon=True)
        audio_process = Process(target=audio_play, args=(self.audio_income_start, self.output_start,), daemon=True)
        
        socket_process.start()
        render_process.start()
        video_process.start()
        audio_process.start()
        
        socket_process.join()
        render_process.join()
        video_process.join()       
        audio_process.join()
        