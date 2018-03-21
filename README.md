# ClearWave
Denoise Speech by Deep Learning (Using Keras and Tensorflow) 

------------------

This project is modified from deep neural network (DNN) by yongxuUSTC(https://github.com/yongxuUSTC/sednn). 

Also, the project uses ffmpeg, webrtc and pesq to deal with speech data.

Before try the project, please download the base dnn model from https://pan.baidu.com/s/1eVnRkNb5xIn96aYOV8C-Gg
 and copy the .h5 file to ./models/pretrained/base_dnn_model.h5.

------------------

# Speech Samples
You could download and listen the clear, noisy and denoised Speech:

The Clear Speech ------- https://github.com/boozyguo/ClearWave/blob/master/notes/THCH_test_D8_770-.wav



The noisy Speech -------https://github.com/boozyguo/ClearWave/blob/master/notes/THCH_test_D8_770.noise1.wav

The Denoised Speech ----https://github.com/boozyguo/ClearWave/blob/master/notes/THCH_test_D8_770.noise1.ns_enh.wav



The noisy Speech -------https://github.com/boozyguo/ClearWave/blob/master/notes/THCH_test_D8_770.noise2.wav

The Denoised Speech ----https://github.com/boozyguo/ClearWave/blob/master/notes/THCH_test_D8_770.noise2.ns_enh.wav


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


The denoised PESQ is(SNR=5db):

<pre>
Calculate overall stats. 
Noise            PESQ            
---------------------------------
Cafeteria_Noise_16s_26s 2.81 +- 0.09    
Fullsize_Car1_16s_26s 3.13 +- 0.09    
Pub_Noise_16s_26s 2.51 +- 0.09    
Outside_Traffic_2s_12s 2.89 +- 0.11    
RockMusic01m48k_16s_26s 3.04 +- 0.09    
---------------------------------
Avg.             2.87 +- 0.10
</pre>

------------------

## Samples:
There are some speech files in "./notes". 

The clear speech is "./notes/THCH_test_D8_770-.wav", which figures showed below:

![Clear Speech](https://github.com/boozyguo/ClearWave/blob/master/notes/clear-d8-770.jpg)


The noisy file are  "./notes/THCH_test_D8_770.noise1.wav" and "./notes/THCH_test_D8_770.noise2.wav", which figures showed below:

![Noisy1](https://github.com/boozyguo/ClearWave/blob/master/notes/noise1-d8-770.jpg)
![Noisy2](https://github.com/boozyguo/ClearWave/blob/master/notes/noise2-d8-770.jpg)


The denoised file are  "./notes/THCH_test_D8_770.noise1.ns_enh.wav" and "./notes/THCH_test_D8_770.noise2.ns_enh.wav", which figures showed below:

![Denoised1](https://github.com/boozyguo/ClearWave/blob/master/notes/denoised-noise1-d8-770.jpg)
![Denoised2](https://github.com/boozyguo/ClearWave/blob/master/notes/denoised-noise2-d8-770.jpg)



------------------


## Donate:

If the project could help you, please star it and give us some donations. Donations will be used to fund expenses related to development (e.g. to cover equipment and server maintenance costs), to sponsor bug fixing, feature development.


WeChat Payment
![WeChat Payment](https://github.com/boozyguo/ClearWave/blob/master/notes/wechat.jpg)


[Paypal Payment](http://paypal.me/githubClearWave)
[![Paypal Payment](https://github.com/boozyguo/ClearWave/blob/master/notes/paypal.jpg)](http://paypal.me/githubClearWave)

------------------

## Ref:

 https://github.com/yongxuUSTC/sednn
 
 http://arxiv.org/abs/1512.01882
