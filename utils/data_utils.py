import os
import numpy as np
import librosa

def load_data(args):
    file, root_dir, processor, templates, audio_dir, vertice_dir, name = args
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        key = file.replace("wav", "npy")

        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        
        result = {}

        result["name"] = file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)
        result["audio"] = input_values

        subject_id = "_".join(key.split("_")[:-1])
        temp = templates[subject_id]
        result["template"] = temp.reshape((-1)) 

        vertice_path = os.path.join(root_dir, vertice_dir, file.replace("wav", "npy"))

        if not os.path.exists(vertice_path):
            return None
        else:
            if name == 'VOCASET':
                result["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:] #due to the memory limit
            elif name == 'BIWI':
                result["vertice"] = np.load(vertice_path,allow_pickle=True)
            return (key, result)


### This is borrowed from DiffSpeaker: https://github.com/theEricMa/DiffSpeaker
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]

    adapted_batch = {
        'audio': collate_tensors([b['audio'].float() for b in notnone_batches]),
        'vertice': collate_tensors([b['vertice'].float() for b in notnone_batches]),
        'template': collate_tensors([b['template'].float() for b in notnone_batches]),
        'id': collate_tensors([b['id'].float() for b in notnone_batches]),
        'file_name': [b['file_name'] for b in notnone_batches],
        'file_path': [b['file_path'] for b in notnone_batches],
    }
    return adapted_batch