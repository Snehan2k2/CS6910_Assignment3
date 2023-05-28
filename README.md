# CS6910-Assignment3

Use recurrent neural networks to build a character transliteration system. The goal of this assignment is fourfold: (i) learn how to model sequence to sequence learning problems using Recurrent Neural Networks (ii) compare different cells such as simple RNN, LSTM and GRU (iii) Usage of attention networks

Running the code:

train.py code can be used to run the code, to check if it works. (sweep.py contains the sweep code and attention.py contains the attention code).


Firstly the dataset zip file has to be present in the google drive, as the drive is mounted and later it's unzipped in the code.

The code can be run by the following command:

python train.py --hidden_size hidden_size --embedding_size embedding_size --num_layers num_layers --dropout dropout --cell_type cell_type
