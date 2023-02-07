meta_path_vox='./misc/demo_id.csv'

jobname=demo_id

python3.8 -u inference.py  \
        --name ${jobname} \
        --meta_path_vox ${meta_path_vox} \
        --dataset_mode voxtest \
        --netG modulate \
        --netA resseaudio \
        --netA_sync ressesync \
        --netD multiscale \
        --netV resnext \
        --netE fan \
        --model av \
        --gpu_ids 0 \
        --clip_len 1 \
        --batchSize 16 \
        --style_dim 2560 \
        --nThreads 4 \
        --generate_interval 1 \
        --style_feature_loss \
        --use_audio 1 \
        --noise_pose \
        --driving_pose \
        --gen_video \
        --generate_from_audio_only \
        --use_audio_id 1 \
        --n_mel_T 4
#        --input_id_feature \
