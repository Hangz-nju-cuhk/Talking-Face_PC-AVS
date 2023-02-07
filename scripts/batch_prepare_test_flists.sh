#
#pose_dir='./misc/Pose_Source'
#audio_dir='./misc/Audio_Source'
#input_dir='./misc/Input'
#
#for input_mp4 in `ls ${input_dir}`
#do
#  for pose_mp4 in `ls ${pose_dir} | shuf`
#  do
#    echo ${pose_mp4}
#    for audio_mp4 in `ls ${audio_dir}`
#    do
#      src_pose_path=${pose_dir}/${pose_mp4}
#      src_audio_path=${audio_dir}/${audio_mp4}
#      src_input_path=${input_dir}/${input_mp4}
#      echo ${input_mp4} ${pose_mp4} ${audio_mp4}
#      cmd="python scripts/prepare_testing_files.py \
#                    --src_input_path ${src_input_path} \
#                    --src_pose_path ${src_pose_path} \
#                    --src_audio_path ${src_audio_path} \
#                    --csv_path ./misc/demo_finetune_AVG.csv"
#      echo ${cmd}
#      $cmd
#    done
#    break
#    break
#  done
#done

# teaser
#src_input_path=./misc/teaser/00131.mp4
#src_pose_path=./misc/teaser/00099.mp4
#src_audio_path=./misc/teaser/00418.mp4
#
#cmd="python scripts/prepare_testing_files.py \
#              --src_input_path ${src_input_path} \
#              --src_pose_path ${src_pose_path} \
#              --src_audio_path ${src_audio_path} \
#              --csv_path ./misc/teaser.csv"
#echo $cmd
#
#$cmd

# qualitative 1
src_input_path=./misc/Input/00098.mp4
src_pose_path=./misc/Pose_Source/00473.mp4
src_audio_path=./misc/Audio_Source/00015.mp3

cmd="python scripts/prepare_testing_files.py \
              --src_input_path ${src_input_path} \
              --src_pose_path ${src_pose_path} \
              --src_audio_path ${src_audio_path} \
              --csv_path ./misc/qual1.csv"
echo $cmd
$cmd

# qualitative 2
src_input_path=./misc/Input/00002.mp4
src_pose_path=./misc/Pose_Source/012345.mp4
src_audio_path=./misc/Audio_Source/00086.mp4

cmd="python scripts/prepare_testing_files.py \
              --src_input_path ${src_input_path} \
              --src_pose_path ${src_pose_path} \
              --src_audio_path ${src_audio_path} \
              --csv_path ./misc/qual2.csv"
echo $cmd

$cmd

