# ClearWave
Denoise Speech by Deep Learning (Using Keras and Tensorflow) 

------------------

This project is modified from deep neural network (DNN) to do speech enhancement by yongxuUSTC(https://github.com/yongxuUSTC/sednn). 

Also, the project uses ffmpeg, webrtc and pesq to deal with speech data.

Before try the project, you should download the base dnn model from https://pan.baidu.com/s/1eVnRkNb5xIn96aYOV8C-Gg

Please copy the .h5 file to ./models/pretrained/base_dnn_model.h5.

------------------

## Inference Usage: Denoise on noisy data. 
If you have noisy speech, you can edit and run "./demo.sh" to denoise the noisy file. 

1. Put the noisy file in path "./demo_data/noisy/*.wav"

2. Edit the demo.sh file with "INPUT_NOISY=1" 

3. Run ./demo.sh

4. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/1000db/*.wav"

------------------

## Inference Usage: Denoise on speech data and noise data. 
If you have clear speech and noise: 

1. Put the noise file in path "./demo_data/noise/*.wav"

2. Put the clear speech file in path "./demo_data/clear/*.wav"

3. Edit the demo.sh file with "INPUT_NOISY=0". Also, you can modify the SNR in parameter "TE_SNR", for example "TE_SNR=5" is 5db.

4. Run ./demo.sh

5. Check the denoised speech in "demo_workspace/ns_enh_wavs/test/5db/*.wav" (if "TE_SNR=5") 

------------------

## Training Usage: Training model on speech data and noise data. 
If you want to train yourself model, just prepare your data, then run "./runme.sh": 

1. Put the train noise file in path "./data/train_noise/*.wav"

2. Put the train clear speech file in path "./data/train_speech/*.wav"

3. Put the validtaion noise file in path "./data/test_noise/*.wav"

4. Put the train validtaion speech file in path "./data/test_speech/*.wav"

5. Edit the runme.sh file, set parameters: TR_SNR, TE_SNR, EPOCHS, LEARNING_RATE

6. Run ./runme.sh

7. Check the new model in "./workspace/models/5db/*.h5" (if "TE_SNR=5") 

------------------


## Models:

ClearWave model based on simple DNN in keras:

```python
    n_concat = 7
    n_freq = 257
    n_hid = 2048
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dropout(0.1))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='relu'))
    model.summary()
```

------------------

## Run on THCH and 5 noises

Training:

Speech: THCHS30(http://cslt.org) 2178 training sentences. (selected 20% from 10893 testing sentences)

Noise: 5 kinds of noises

Testing:

Speech: THCHS30 499 testing sentences (selected 20% from 2495 testing sentences)

Noise: same to training


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

------------------

## Samples:
There are some speech files in "./notes". 

The clear speech is "./notes/THCH_test_D8_770-.wav", which figures showed below:

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)


The noisy file are  "./notes/THCH_test_D8_770.noise1.wav" and "./notes/THCH_test_D8_770.noise2.wav", which figures showed below:

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)
![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)


The denoised file are  "./notes/THCH_test_D8_770.noise1.ns_enh.wav" and "./notes/THCH_test_D8_770.noise2.ns_enh.wav", which figures showed below:

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)
![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)



------------------


## Donate:

If the project could help you, please star it and give us some donations. Donations will be used to fund expenses related to development (e.g. to cover equipment and server maintenance costs), to sponsor bug fixing, feature development.


WeChat Payment
![WeChat Payment](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)


