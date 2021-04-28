import os
import sys
sys.path.append('..')
from options.test_options import TestOptions
import torch
from models import create_model
import data
import util.util as util
from tqdm import tqdm


def video_concat(processed_file_savepath, name, video_names, audio_path):
    cmd = ['ffmpeg']
    num_inputs = len(video_names)
    for video_name in video_names:
        cmd += ['-i', '\'' + str(os.path.join(processed_file_savepath, video_name + '.mp4'))+'\'',]

    cmd += ['-filter_complex hstack=inputs=' + str(num_inputs),
            '\'' + str(os.path.join(processed_file_savepath, name+'.mp4')) + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

    video_add_audio(name, audio_path, processed_file_savepath)


def video_add_audio(name, audio_path, processed_file_savepath):
    os.system('cp {} {}'.format(audio_path, processed_file_savepath))
    cmd = ['ffmpeg', '-i', '\'' + os.path.join(processed_file_savepath, name + '.mp4') + '\'',
                     '-i', audio_path,
                     '-q:v 0',
                     '-strict -2',
                     '\'' + os.path.join(processed_file_savepath, 'av' + name + '.mp4') + '\'',
                     '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)


def img2video(dst_path, prefix, video_path):
    cmd = ['ffmpeg', '-i', '\'' + video_path + '/' + prefix + '%d.jpg'
           + '\'', '-q:v 0', '\'' + dst_path + '/' + prefix + '.mp4' + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)


def inference_single_audio(opt, path_label, model):
    #
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()

    idx = 0
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
    for data_i in tqdm(dataloader):
        # print('==============', i, '===============')
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
                util.save_torch_img(data_i['target'][num], os.path.join(save_paths[-1], video_names[-1] + str(idx) + '.jpg'))
            idx += 1

    if opt.gen_video:
        for i, video_name in enumerate(video_names):
            img2video(processed_file_savepath, video_name, save_paths[i])
        video_concat(processed_file_savepath, 'concat', video_names, dataloader.dataset.audio_path)

    print('results saved...' + processed_file_savepath)
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
