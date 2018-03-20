#!/bin/bash

CMD="main_dnn.py"

WORKSPACE="./workspace"
mkdir $WORKSPACE
TR_SPEECH_DIR="./data/train_speech"
TR_NOISE_DIR="./data/train_noise"
TE_SPEECH_DIR="./data/test_speech"
TE_NOISE_DIR="./data/test_noise"

MODEL_FILE="null"

TR_SNR=5
TE_SNR=5
N_CONCAT=7
N_HOP=2
CALC_LOG=0
EPOCHS=100000
ITERATION=90000
LEARNING_RATE=1e-4

CALC_DATA=1
if [ $CALC_DATA -eq 1 ]; then
  # Create mixture csv.
  echo "Go:Create mixture csv. "
  python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --speechratio=1 --magnification=2
  python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test  --speechratio=1


  #echo "Calculate mixture features. "
  TR_SNR=5
  TE_SNR=5
  python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
  python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR


  #echo "finish!"
  #exit

  # Calculate PESQ of all noisy speech.
  echo "Calculate PESQ of all noisy speech. "
  python evaluate.py calculate_noisy_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

  # Calculate noisy overall stats.
  echo "Calculate noisy overall stats. "
  python evaluate.py get_stats


  # Pack features.
  echo "Pack features. "
  N_CONCAT=7
  N_HOP=2
  python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --calc_log=$CALC_LOG
  python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --calc_log=$CALC_LOG

  # Compute scaler.
  echo "Compute scaler. "
  python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR

fi



# Train. 
echo "Train. "
CUDA_VISIBLE_DEVICES=0 python $CMD train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE --epoch=$EPOCHS --calc_log=$CALC_LOG


# Inference, enhanced wavs will be created. 
echo "Inference, enhanced wavs will be created. "
CUDA_VISIBLE_DEVICES=0 python $CMD inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION --calc_log=$CALC_LOG --model_file=$MODEL_FILE

# Calculate PESQ of all enhanced speech. 
echo "Calculate PESQ of all enhanced speech. "
python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

# Calculate overall stats. 
echo "Calculate overall stats. "
python evaluate.py get_stats

