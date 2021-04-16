meta_path_vox='./misc/demo.csv'


python -u inference.py  \
        --name demo \
        --list_start 0 \
        --list_end 1 \
        --dataset_mode voxtest \
        --netG modulate \
        --netA resseaudio \
        --netA_sync ressesync \
        --netD multiscale \
        --netV resnext \
        --netE fan \
        --num_inputs 1 \
        --gpu_ids 0 \
        --clip_len 1 \
        --batchSize 8 \
        --model av \
        --style_dim 2560 \
        --nThreads 4 \
        --norm_D spectralsyncbatch \
        --input_id_feature \
        --generate_interval 1 \
        --style_feature_loss \
        --use_audio 1 \
        --noise_pose \
        --driving_pose \
        --gen_video \
        --meta_path_vox ${meta_path_vox} \
        --generate_from_audio_only \
