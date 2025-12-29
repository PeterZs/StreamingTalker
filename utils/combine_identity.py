import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.io import wavfile
from multiprocessing import Pool

subjects = [ # for VOCASET test
            'FaceTalk_170809_00138_TA',
            'FaceTalk_170731_00024_TA'
            ]

# subjects = [ # for BIWI test B
#                 "F1",
#                 "F5",
#                 "F6",
#                 "F7",
#                 "F8",
#                 "M1",
#                 "M2",
#                 "M6"
#             ]

def get_args_parser():
    parser = ArgumentParser('combine long seqs', add_help=False)
    parser.add_argument('--vertice_dir', 
                        default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/vertices_npy', 
                        type=str)
    parser.add_argument('--audio_dir', 
                        default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/wav', 
                        type=str)
    parser.add_argument('--vertice_output_dir', 
                        default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/longseq_npy', 
                        type=str)
    parser.add_argument('--audio_output_dir', 
                        default='/nas/home/yangyifan/Code/3dFacialAnimation/DiffSpeaker/datasets/vocaset/longseq_wav', 
                        type=str)
    parser.add_argument('--multiprocessing', 
                        action='store_true')
    parser.add_argument('--cpu_count', 
                        default=8, 
                        type=int)
    parser.add_argument('--rate', 
                        default=44100, 
                        type=int)
    parser.add_argument('--length', 
                        default=10, 
                        type=int)
    return parser

def read_audio(file):
    _, audio_data = wavfile.read(file)
    return audio_data

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    os.makedirs(args.vertice_output_dir, exist_ok=True)
    os.makedirs(args.audio_output_dir, exist_ok=True)

    for prefix in tqdm(subjects):
        vertice_files = []
        audio_files = []

        for file_name in os.listdir(args.vertice_dir):
            if file_name.startswith(prefix + '_') and file_name.endswith('.npy'):
                vertice_files.append(os.path.join(args.vertice_dir, file_name))
        vertice_files = sorted(vertice_files)[:args.length] 
        for file_name in os.listdir(args.audio_dir):
            if file_name.startswith(prefix + '_') and file_name.endswith('.wav'):
                audio_files.append(os.path.join(args.audio_dir, file_name))
        audio_files = sorted(audio_files)[:args.length] 

        print(len(vertice_files),len(audio_files))
        assert len(vertice_files) == len(audio_files), "Number of vertex files and audio files must match"

        if args.multiprocessing: 
            with Pool(processes = args.cpu_count) as pool:
                all_vertices = pool.map(np.load,  vertice_files)
            combined_vertices = np.concatenate(all_vertices, axis=0)

            with Pool(processes = args.cpu_count) as pool:
                all_audios = pool.map(read_audio,  audio_files)
            combined_audio = np.concatenate(all_audios, axis=0)

        else:
            all_vertices = []
            for vertice_file in vertice_files:
                vertices = np.load(vertice_file)
                all_vertices.append(vertices)
                print(vertices.shape)
            combined_vertices = np.concatenate(all_vertices, axis=0)

            all_audios = []
            for audio_file in audio_files:
                rate, audio_data = wavfile.read(audio_file)
                all_audios.append(audio_data)
                print(audio_data.shape)
            combined_audio = np.concatenate(all_audios, axis=0)

        print(combined_vertices.shape, combined_audio.shape)

        output_vertice_file = os.path.join(args.vertice_output_dir, f'{prefix}_e00.npy')
        output_audio_file = os.path.join(args.audio_output_dir, f'{prefix}_e00.wav')
        np.save(output_vertice_file, combined_vertices)
        wavfile.write(output_audio_file, args.rate, combined_audio)