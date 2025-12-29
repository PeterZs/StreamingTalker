import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频文件
audio_file = 'speech_20241111081057475.wav'  # 替换为你的音频文件路径
y, sr = librosa.load(audio_file, sr=44100)  # y是音频数据，sr是采样率

# 创建波形图（无坐标轴）
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.8, color='purple')
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()

# 保存波形图
output_path = 'audio_waveform_no_axes1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"波形图已保存为: {output_path}")
