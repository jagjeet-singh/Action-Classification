# Action Classification

This repository contains the code for Action Classification on [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#introduction) dataset. It uses the ResNet18 features available at the following links: ([TrainSet](https://www.dropbox.com/s/y23pdfngf7uu4xn/annotated_train_set.p?dl=0), [TestSet](https://www.dropbox.com/s/2zc1vystx0161cr/randomized_annotated_test_set_no_name_no_num.p?dl=0)). There are two networks used:

1. Fully Connected Network that tries to classify individual frames and then pools the results
2. LSTM based network that keeps track of the temporal information as well


Requirements -

tensorboard_logger (https://github.com/TeamHG-Memex/tensorboard_logger) \
PyTorch\
TensorFlow
