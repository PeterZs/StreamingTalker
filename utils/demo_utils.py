import os
import cv2
import torch
import librosa
import pyrender
import trimesh
import numpy as np
import tempfile
import imageio

from tqdm import tqdm
from transformers import Wav2Vec2Processor
try:
    from psbody.mesh import Mesh
except:
    Mesh = None

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import bpy

import platform
if platform.system() == "Linux":
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

blink_exp_betas = np.array(
    [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
        0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
        0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
        -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
        3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
        -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
        -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
        -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
        -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
        1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
        2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
        0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
        -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
        1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
        0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
        -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
        -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
        -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
        1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
        -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])

import bpy
import numpy as np

def render_mesh_with_blender(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, template_type="flame"):
    scene = bpy.context.scene

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_data = bpy.data.meshes.new("mesh")
    mesh_data.from_pydata(mesh.v.tolist(), [], mesh.f.tolist())
    mesh_obj = bpy.data.objects.new("Mesh", mesh_data)

    mesh_obj.location = t_center
    mesh_obj.rotation_euler = np.deg2rad(rot)
    scene.collection.objects.link(mesh_obj)

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    camera_obj.location = (0, 0, 1.0 - z_offset)
    scene.camera = camera_obj

    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_obj = bpy.data.objects.new("Light", light_data)
    light_obj.location = (0, 0, 3)
    scene.collection.objects.link(light_obj)

    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.film_transparent = True

    bpy.ops.render.render(write_still=False)
    bpy.context.view_layer.update()

    image = bpy.data.images['Render Result']
    image_data = np.array(image.pixels[:])
    print(image_data)
    image_data = image_data.reshape((scene.render.resolution_y, scene.render.resolution_x, 4))
    image_data = image_data[:, :, :3]

    return image_data


def render_mesh_with_opengl(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"OpenGL Rendering")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glTranslatef(*t_center)
    glRotatef(rot[0], 1, 0, 0)
    glRotatef(rot[1], 0, 1, 0)
    glRotatef(rot[2], 0, 0, 1)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 800, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    depth_buffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 800)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)

    framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if framebuffer_status != GL_FRAMEBUFFER_COMPLETE:
        print(f"Framebuffer not complete, status code: {framebuffer_status}")
        return

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    for i in range(len(mesh.f)):
        for j in range(3):
            vertex = mesh.v[mesh.f[i][j]]
            glVertex3fv(vertex)
    glEnd()

    pixels = glReadPixels(0, 0, 800, 800, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((800, 800, 3))[::-1, :, :]


    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [fbo])
    glDeleteTextures(1, [texture])
    glDeleteRenderbuffers(1, [depth_buffer])

    return image

def load_audio(audio_path, processor = None):
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
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

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, template_type: str = "flame", rgb_per_v = None):
    

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

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center

    if rgb_per_v is None:
        intensity = 2.0
        primitive_material = pyrender.material.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                    metallicFactor=0.8, 
                    roughnessFactor=0.8 
                )

        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
    else:
        intensity = 0.5
        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    # scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    # if rgb_per_v is None:
    #     intensity = 0.5
    #     primitive_material = pyrender.material.MetallicRoughnessMaterial(
    #                 alphaMode='BLEND',
    #                 baseColorFactor=[220/255, 190/255, 110/255, 1.0], 
    #                 # baseColorFactor=[110/255, 190/255, 220/255, 1.0], 
    #                 # baseColorFactor=[50/255, 120/255, 150/255, 1.0], 
    #                 metallicFactor=0.0,                   # 半金属
    #                 roughnessFactor=0.9,                  # 光滑表面
    #                 emissiveFactor=[0.0, 0.0, 0.0]        # 微弱自发光
    #             )

    #     tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    #     render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
    # else:
    #     intensity = 0.5
    #     tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    #     render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    # scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[255, 255, 255])

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

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

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    # try:
    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=flags)
    # except:
    #     print('pyrender: Failed rendering frame')
    #     color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_frame(args):
    predicted_vertice, f, center,  template_type = args
    render_mesh = Mesh(predicted_vertice, f)
    pred_img = render_mesh_helper(render_mesh, center, template_type=template_type)
    pred_img = pred_img.astype(np.uint8)
    return pred_img

def animate(vertices: np.array, wav_path, file_name: str, ply: str, fps: int = 25, vertice_gt: np.array = None, use_tqdm: bool = False, multi_process = False):
    """
    Animate the predicted vertices with the synchronized audio and save the video to the output directory.
    Args:
        vertices: (num_frames, num_vertices*3)
        wav_path: path to wav file
        file_name: name of the output file
        ply: path to the ply file
        fps: frames per second
        use_tqdm: whether to use tqdm to show the progress
        vertice_gt: (num_frames, num_vertices*3)
        template: template to use, can be "flame" or "biwi"
    """
    # make output dir
    output_dir = os.path.dirname(file_name)
    os.makedirs(output_dir, exist_ok=True)

    template = Mesh(filename=ply)
    # determine biwi or flame
    if "FLAME" in ply:
        template_type = "flame"
    elif "BIWI" in ply:
        template_type = "biwi"
    else:
        raise ValueError("Template type not recognized, please use either BIWI or FLAME")

    # reshape vertices
    predicted_vertices = vertices.reshape(-1, vertices.shape[1]//3, 3) if vertices.ndim < 3 else vertices

    num_frames = predicted_vertices.shape[0]
    if vertice_gt is not None:
        vertice_gt = vertice_gt.reshape(-1, vertice_gt.shape[1]//3, 3) if vertice_gt.ndim < 3 else vertice_gt
        num_frames = np.where(np.sum(vertice_gt, axis=(1, 2)) != 0)[0][-1] + 1 # find the number of frames where the vertices are not all zeros

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_dir)
    center = np.mean(predicted_vertices[0], axis=0)


    # make animation
    if multi_process:

        from multiprocessing import Pool, cpu_count
        from itertools import cycle
        # get maximum num of process
        frames = []
        max_processes = cpu_count()
        with Pool(processes=max_processes) as pool:
            args = [(
                predicted_vertice,
                template.f,
                center,
                template_type
            ) for predicted_vertice in predicted_vertices]

            for pred_img in pool.imap(render_frame, tqdm(args)):
                frames.append(pred_img)

        if vertice_gt is not None:
            frames_gt = []
            with Pool(processes=max_processes) as pool:
                args = [(
                    gt_vertice,
                    template.f,
                    center,
                    template_type
                ) for gt_vertice in vertice_gt]
                
                for gt_img in pool.imap(render_frame, tqdm(args)):
                    frames_gt.append(gt_img)

            # concat two videos
            frames_final = []
            for i in range(num_frames):
                frames_final.append(np.concatenate([frames_gt[i], frames[i]], axis=1))
            frames = frames_final

    else:
        frames = []
        for i_frame in tqdm(range(num_frames)) if use_tqdm else range(num_frames):
            render_mesh = Mesh(predicted_vertices[i_frame], template.f)
            pred_img = render_mesh_helper(render_mesh, center, template_type=template_type)
            pred_img = pred_img.astype(np.uint8)
            frames.append(pred_img)

        if vertice_gt is not None:
            frames_gt = []
            for i_frame in tqdm(range(num_frames)) if use_tqdm else range(num_frames):
                render_mesh = Mesh(vertice_gt[i_frame], template.f)
                pred_img = render_mesh_helper(render_mesh, center)
                pred_img = pred_img.astype(np.uint8)
                frames_gt.append(pred_img)
        
            # concat two videos
            frames_final = []
            for i in range(num_frames):
                frames_final.append(np.concatenate([frames_gt[i], frames[i]], axis=1))
            frames = frames_final

    imageio.mimsave(tmp_video_file.name, frames, fps = fps)

    if wav_path is not None:
        # cmd = " ".join(['ffmpeg', '-hide_banner -loglevel error', '-y', '-i', tmp_video_file.name, '-i', wav_path, '-c:v copy -c:a aac', '-pix_fmt yuv420p -qscale 0',file_name, ])
        cmd = " ".join(['ffmpeg', '-i', tmp_video_file.name, '-i', wav_path, '-c:v copy -c:a aac', '-pix_fmt yuv420p -qscale 0',file_name, ])
    else:
        cmd = " ".join(['ffmpeg', '-i', tmp_video_file.name, '-c:v copy', '-pix_fmt yuv420p -qscale 0',file_name, ])

    os.system(cmd)
    if wav_path is not None:
        tmp_dir = tempfile.gettempdir() # check if the wav file is in the tmp dir
        if os.path.exists(wav_path) and tmp_dir in wav_path: 
            os.remove(wav_path)

    print(f"Video saved to {file_name}")
