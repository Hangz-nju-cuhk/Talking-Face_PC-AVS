import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.AudioConfig import AudioConfig  # noqa: E402


def proc_frames(src_path, dst_path):
    # -qp 0 is the lossless flag for mp4. For avi, it should be -q:v 1.
    cmd = f'ffmpeg -i \"{src_path}\" -start_number 0 -qp 0 \"{dst_path}\"/%06d.jpg -loglevel error -y'
    os.system(cmd)
    return len(glob.glob(os.path.join(dst_path, '*.jpg')))


def proc_audio(src_mouth_path, dst_audio_path):
    audio_command = 'ffmpeg -i \"{}\" -loglevel error -y -f wav -acodec pcm_s16le ' \
                    '-ar 16000 \"{}\"'.format(src_mouth_path, dst_audio_path)
    os.system(audio_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dst_dir_path', default='/mnt/lustre/DATAshare3/VoxCeleb2',
    #                     help="dst file position")
    parser.add_argument('--dir_path', default='./misc',
                        help="dst file position")
    parser.add_argument('--src_pose_path', default='./misc/Pose_Source/00473.mp4',
                        help="pose source file position, this could be an mp4 or a folder")
    parser.add_argument('--src_audio_path', default='./misc/Audio_Source/00015.mp4',
                        help="audio source file position, it could be an mp3 file or an mp4 video with audio")
    parser.add_argument('--src_mouth_frame_path', default=None,
                        help="mouth frame file position, the video frames synced with audios")
    parser.add_argument('--src_input_path', default='./misc/Input/00098.mp4',
                        help="input file position, it could be a folder with frames, a jpg or an mp4")
    parser.add_argument('--csv_path', default='./misc/demo2.csv',
                        help="path to output index files")
    parser.add_argument('--convert_spectrogram', action='store_true', help='whether to convert audio to spectrogram')

    args = parser.parse_args()
    dir_path = args.dir_path
    os.makedirs(dir_path, exist_ok=True)

    # ===================== process input =======================================================
    input_save_path = os.path.join(dir_path, 'Input')
    os.makedirs(input_save_path, exist_ok=True)
    input_name = args.src_input_path.split('/')[-1].split('.')[0]
    num_inputs = 1
    dst_input_path = os.path.join(input_save_path, input_name)
    os.makedirs(dst_input_path, exist_ok=True)
    if args.src_input_path.split('/')[-1].split('.')[-1] == 'mp4':
        num_inputs = proc_frames(args.src_input_path, dst_input_path)
    elif os.path.isdir(args.src_input_path):
        dst_input_path = args.src_input_path
    else:
        os.system(f'cp {args.src_input_path} {os.path.join(dst_input_path, args.src_input_path.split("/")[-1])}')

    # ===================== process audio =======================================================
    audio_source_save_path = os.path.join(dir_path, 'Audio_Source')
    os.makedirs(audio_source_save_path, exist_ok=True)
    audio_name = args.src_audio_path.split('/')[-1].split('.')[0]
    spec_dir = 'None'
    dst_audio_path = os.path.join(audio_source_save_path, audio_name + '.mp3')

    if args.src_audio_path.split('/')[-1].split('.')[-1] == 'mp3':
        os.system('cp {} {}'.format(args.src_audio_path, dst_audio_path))
        if args.src_mouth_frame_path and os.path.isdir(args.src_mouth_frame_path):
            dst_mouth_frame_path = args.src_mouth_frame_path
            src_mouth_frames = glob.glob(os.path.join(args.src_mouth_frame_path, '*.jpg')) + \
                glob.glob(os.path.join(args.src_mouth_frame_path, '*.png'))
            num_mouth_frames = len(src_mouth_frames)
        else:
            dst_mouth_frame_path = 'None'
            num_mouth_frames = 0
    else:
        mouth_source_save_path = os.path.join(dir_path, 'Mouth_Source')
        os.makedirs(mouth_source_save_path, exist_ok=True)
        dst_mouth_frame_path = os.path.join(mouth_source_save_path, audio_name)
        os.makedirs(dst_mouth_frame_path, exist_ok=True)
        proc_audio(args.src_audio_path, dst_audio_path)
        num_mouth_frames = proc_frames(args.src_audio_path, dst_mouth_frame_path)

    if args.convert_spectrogram:
        audio = AudioConfig(fft_size=1280, hop_size=160)
        wav = audio.read_audio(dst_audio_path)
        spectrogram = audio.audio_to_spectrogram(wav)
        spec_dir = os.path.join(audio_source_save_path, audio_name + '.npy')
        np.save(spec_dir, spectrogram.astype(np.float32), allow_pickle=False)

    # ===================== process pose =======================================================
    if os.path.isdir(args.src_pose_path):
        pose_frames = glob.glob(os.path.join(args.src_pose_path, '*.jpg')) + \
            glob.glob(os.path.join(args.src_pose_path, '*.png'))
        num_pose_frames = len(pose_frames)
        dst_pose_frame_path = args.src_pose_path
    else:
        pose_source_save_path = os.path.join(dir_path, 'Pose_Source')
        os.makedirs(pose_source_save_path, exist_ok=True)
        pose_name = args.src_pose_path.split('/')[-1].split('.')[0]
        dst_pose_frame_path = os.path.join(pose_source_save_path, pose_name)
        os.makedirs(dst_pose_frame_path, exist_ok=True)
        num_pose_frames = proc_frames(args.src_pose_path, dst_pose_frame_path)

    # ===================== form csv =======================================================

    with open(args.csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerows([[dst_input_path, str(num_inputs), dst_pose_frame_path, str(num_pose_frames),
                           dst_audio_path, dst_mouth_frame_path, str(num_mouth_frames), spec_dir]])
        print('meta-info saved at ' + args.csv_path)

    csvfile.close()
