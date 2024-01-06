import os

import torch
from tqdm import tqdm

import data
import util.util as util
from models import create_model
from options.test_options import TestOptions


def video_concat(root_dir: str, filename: str, video_names: list, audio_path: str):
    output_path: str = os.path.join(root_dir, filename + '.mp4')
    cmd = [f'-i \"{os.path.join(root_dir, video_name)}.mp4\" ' for video_name in video_names]
    cmd.append(f'-filter_complex hstack=inputs={len(video_names)}')
    cmd.append('-qp 0')                             # Lossless flag for mp4.
    cmd.append(f'\"{output_path}\"')
    cmd.append('-y')
    cmd.append('-loglevel error')

    os.system(f'ffmpeg {" ".join(cmd)}')
    video_add_audio(filename, audio_path, root_dir)


def video_add_audio(video_filename: str, audio_path: str, root_dir: str):
    os.system(f'cp {audio_path} {root_dir}')
    video_input_path = os.path.join(root_dir, video_filename + '.mp4')
    audio_input_path = audio_path
    output_path = os.path.join(root_dir, 'av' + video_filename + '.mp4')
    # No need to re-encode again, just copy.
    os.system(f'ffmpeg -i \"{video_input_path}\" -i \"{audio_input_path}\" -c copy \"{output_path}\" -y')


def img2video(dst_path: str, prefix: str, video_path: str):
    # -qp 0 is a lossless flag for mp4. For avi, it should be -q:v 1.
    # Specify framerate to 25 in case ffmpeg changes it in the future.
    cmd = f'ffmpeg -framerate 25 -i \"{video_path}/{prefix}%d.jpg\" -qp 0 -r 25 \"{dst_path}/{prefix}.mp4\" -y'
    os.system(cmd)


def inference_single_audio(opt, path_label, model):
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()

    if opt.driving_pose:
        video_names = ['Input_', 'G_Pose_Driven_', 'Pose_Source_', 'Mouth_Source_']
    else:
        video_names = ['Input_', 'G_Fix_Pose_', 'Mouth_Source_']
    is_mouth_frame = os.path.isdir(dataloader.dataset.mouth_frame_path)
    if not is_mouth_frame:
        video_names.pop()
    save_paths = []
    for name in video_names:
        save_path = os.path.join(processed_file_savepath, name)
        util.mkdir(save_path)
        save_paths.append(save_path)

    idx = 0
    for data_i in tqdm(dataloader):
        fake_image_original_pose_a, fake_image_driven_pose_a = model.forward(data_i, mode='inference')

        for num in range(len(fake_image_driven_pose_a)):
            util.save_torch_img(data_i['input'][num], os.path.join(save_paths[0], video_names[0] + str(idx) + '.jpg'))
            if opt.driving_pose:
                util.save_torch_img(fake_image_driven_pose_a[num],
                                    os.path.join(save_paths[1], video_names[1] + str(idx) + '.jpg'))
                util.save_torch_img(data_i['driving_pose_frames'][num],
                                    os.path.join(save_paths[2], video_names[2] + str(idx) + '.jpg'))
            else:
                util.save_torch_img(fake_image_original_pose_a[num],
                                    os.path.join(save_paths[1], video_names[1] + str(idx) + '.jpg'))
            if is_mouth_frame:
                util.save_torch_img(data_i['target'][num],
                                    os.path.join(save_paths[-1], video_names[-1] + str(idx) + '.jpg'))
            idx += 1

    if opt.gen_video:
        for i, video_name in enumerate(video_names):
            img2video(processed_file_savepath, video_name, save_paths[i])
        video_concat(processed_file_savepath, 'concat', video_names, dataloader.dataset.audio_path)

    print('Results saved under:', processed_file_savepath)
    del dataloader
    return


def main():
    opt = TestOptions().parse()
    opt.isTrain = False
    torch.manual_seed(0)
    model = create_model(opt).cuda()
    model.eval()

    with open(opt.meta_path_vox, 'r') as f:
        lines = f.read().splitlines()

    for clip_idx, path_label in enumerate(lines):
        try:
            assert len(path_label.split()) == 8, path_label

            inference_single_audio(opt, path_label, model)

        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(path_label + '\n')
            print(str(ex))


if __name__ == '__main__':
    main()
