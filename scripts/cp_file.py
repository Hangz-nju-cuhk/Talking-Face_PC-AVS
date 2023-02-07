import os
import shutil


def main():
    results_dir = './results'
    collections_dir = './results/collections_audio_id'
    os.makedirs(collections_dir, exist_ok=True)
    for dir_name in os.listdir(results_dir):
        if not dir_name.startswith('id'): continue
        av_concat_path = os.path.join(results_dir, dir_name, 'avconcat.mp4')
        tgt_concat_path = os.path.join(collections_dir, dir_name+'.mp4')
        shutil.copyfile(av_concat_path, tgt_concat_path)


if __name__ == '__main__':
    main()
