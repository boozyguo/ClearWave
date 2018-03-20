# ClearWave
Denoise Speech by Deep Learning (Using Keras and Tensorflow) 

This project is modified from deep neural network (DNN) to do speech enhancement by yongxuUSTC(https://github.com/yongxuUSTC/sednn). 

Also, the project uses ffmpeg, webrtc and pesq to deal with speech data.

Before try the project, you should download the base dnn model from https
Please copy the .h5 file to ./models/pretrained/base_dnn_model.h5.

## Inference Usage: Denoise on noisy data. 
If you have noisy speech, you can edit and run "demo.sh" and denoised the noisy file. 

1. Put the noisy file in path "./demo_data/noisy/*.wav"

2. Edit the demo.sh file with "INPUT_NOISY=1" 

3. Run ./demo.sh

4. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/1000db/*.wav"



## Inference Usage: Denoise on speech data and noise data. 
If you have clear speech and noise: 

1. Put the noise file in path "./demo_data/noise/*.wav"

2. Put the clear speech file in path "./demo_data/clear/*.wav"

3. Edit the demo.sh file with "INPUT_NOISY=0". Also, you can modify the SNR in parameter "TE_SNR", for example "TE_SNR=5" is 5db.

4. Run ./demo.sh

5. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/5db/*.wav" (if "TE_SNR=5") 


## Training Usage: Training model on speech data and noise data. 
If you have clear speech and noise: 



It is suggest to use mini data for a quick run before using full data. We already prepared mini data in this repo. You may run the code as follows, 


## Run on mini data. 
It is suggest to use mini data for a quick run before using full data. We already prepared mini data in this repo. You may run the code as follows, 

1. pip install -r requirements.txt

2. Download the PESQ evaluation tool (https://www.itu.int/rec/T-REC-P.862-200102-I/en) and compile the code by $ gcc -o pesq pesq_tool/*.c -lm

Copy the compiled executable pesq to mixture2clean_dnn/

3. Run ./runme.sh, then mixing data, training, inference and evaluation will be executed. You may also run the commands in runme.sh line by line to ensure every steps are correctly runned. 

If all the steps are successful, you may get results printed on the screen. Notice only mini data is used for training. Better results can be obtained using more data for training. 

<pre>
Noise(0dB)   PESQ
----------------------
n64     1.36 +- 0.05
n71     1.35 +- 0.18
----------------------
Avg.    1.35 +- 0.12
</pre>

## Run on TIMIT and 115 noises
You may replace the mini data with your own data. We listed the data need to be prepared in meta_data/ to re-run the experiments in [1]. The data contains:

Training:
Speech: TIMIT 4620 training sentences. 
Noise: 115 kinds of noises (http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/115noises.html)

Testing:
Speech: TIMIT 168 testing sentences (selected 10% from 1680 testing sentences)
Noise: Noise 92 (http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html)

Some of the dataset are not published. Instead, you could collect your own data. 

1. Download and prepare data. 

2. Set MINIDATA=0 in runme.sh. Modify WORKSPACE, TR_SPEECH_DIR, TR_NOISE_DIR, TE_SPEECH_DIR, TE_NOISE_DIR in runme.sh file. 

3. Run ./runme.sh

If all the steps are successful, you may get results printed on the screen. The training takes a few miniutes to train 10,000 iterations on a TitanX GPU. The training and testing loss looks like:

<pre>
Iteration: 0, tr_loss: 1.228049, te_loss: 1.252313
Iteration: 1000, tr_loss: 0.533825, te_loss: 0.677872
Iteration: 2000, tr_loss: 0.505751, te_loss: 0.678816
Iteration: 3000, tr_loss: 0.483631, te_loss: 0.666576
Iteration: 4000, tr_loss: 0.480287, te_loss: 0.675403
Iteration: 5000, tr_loss: 0.457020, te_loss: 0.676319
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_5000iters.h5
Iteration: 6000, tr_loss: 0.461330, te_loss: 0.673847
Iteration: 7000, tr_loss: 0.445159, te_loss: 0.668545
Iteration: 8000, tr_loss: 0.447244, te_loss: 0.680740
Iteration: 9000, tr_loss: 0.427652, te_loss: 0.678236
Iteration: 10000, tr_loss: 0.421219, te_loss: 0.663294
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_10000iters.h5
Training time: 202.551192045 s
</pre>

The final PESQ looks like:

<pre>
Noise(0dB)            PESQ
---------------------------------
pink             2.01 +- 0.23
buccaneer1       1.88 +- 0.25
factory2         2.21 +- 0.21
hfchannel        1.63 +- 0.24
factory1         1.93 +- 0.23
babble           1.81 +- 0.28
m109             2.13 +- 0.25
leopard          2.49 +- 0.23
volvo            2.83 +- 0.23
buccaneer2       2.03 +- 0.25
white            2.00 +- 0.21
f16              1.86 +- 0.24
destroyerops     1.99 +- 0.23
destroyerengine  1.86 +- 0.23
machinegun       2.55 +- 0.27
---------------------------------
Avg.             2.08 +- 0.24
</pre>


## Visualization
In the inference step, you may add --visualize to the arguments to plot the mixture, clean and enhanced speech log magnitude spectrogram. 

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)

