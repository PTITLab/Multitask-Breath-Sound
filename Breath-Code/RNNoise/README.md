### RNNoise filter
Load data in folder ```Breath-Data/```

```
Breath-Data
└───01_male_23_BQuyen.wav
	|02_male_22_PTuan.wav
	│03_male_21_BDuong.wav
	|...
└───label
	└───01_male_23_BQuyen.txt
        |02_male_22_PTuan.txt
        │03_male_21_BDuong.txt
        |...
```
---

### Getting Started

##### Requirements

- python=3.7
- autoconf
- libtool

### Check out the code and install: 
```sh
git clone https://github.com/Desklop/RNNoise_Wrapper
cd RNNoise_Wrapper
chmod +x ./compile_rnnoise.sh
./compile_rnnoise.sh
```

### Convert model RNNoise to lib RNNoise: 
```sh
cd RNNoise_Wrapper
python3 rnnoise-master/training/dump_rnn_mod.py train_logs/test_training_set/weights_test_b_500k.hdf5 rnnoise-master/src/rnn_data.c rnnoise-master/src/rnn_data.h
./compile_rnnoise.sh
cd rnnoise-master && make clean && ./autogen.sh && ./configure && make && cd -
cp rnnoise-master/.libs/librnnoise.so.0.4.1 train_logs/test_training_set/librnnoise_test_b_500k.so.0.4.1
```

### Run RNNoise filter to convert data from raw data to data filter: 
```sh
cd RNNoise
python RNNoise_filter.py
```