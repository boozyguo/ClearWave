#!/bin/bash

CMD="main_dnn.py"


MODEL_FILE="./models/pretrained/base_dnn_model.h5"
INPUT_NOISY=1

WORKSPACE="./demo_workspace"
mkdir $WORKSPACE
DEMO_SPEECH_DIR="./demo_data/speech"
DEMO_NOISE_DIR="./demo_data/noise"
DEMO_NOISY_DIR="./demo_data/noisy"
echo "Denoise Demo. "


TR_SNR=5
TE_SNR=5
N_CONCAT=7
N_HOP=2
CALC_LOG=0
#EPOCHS=10000
ITERATION=10000
#LEARNING_RATE=1e-3

CALC_DATA=1
if [ $CALC_DATA -eq 1 ]; then

  if [ $INPUT_NOISY -eq 0 ]; then
      # Create mixture csv.
      echo "Go:Create mixture csv. "
      python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --noise_dir=$DEMO_NOISE_DIR --data_type=test  --speechratio=1

      # Calculate mixture features.
      echo "Go:Calculate mixture features. "
      python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$DEMO_SPEECH_DIR --noise_dir=$DEMO_NOISE_DIR --data_type=test --snr=$TE_SNR
  else
      # Calculate noisy features.
      TE_SNR=1000
      echo "Go:Calculate noisy features. "
      python prepare_data.py calculate_noisy_features --workspace=$WORKSPACE --noisy_dir=$DEMO_NOISY_DIR --data_type=test --snr=$TE_SNR
  fi

  echo "Data finish!"
  #exit

fi


# Inference, enhanced wavs will be created.
echo "Inference, enhanced wavs will be created. "
CUDA_VISIBLE_DEVICES=0 python $CMD inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION --calc_log=$CALC_LOG --model_file=$MODEL_FILE


