# MCV 2020 - M5 Visual Recognition - Group 6

## Project: Object Detection and Segmentation

## Group Members:
- Dhananjay Nahata: nahatadhananjay33@gmail.com
- Siddhant Bhambri: siddhantbhambri@gmail.com
- Yu Pang: yupang2020@gmail.com
- Khanh Nguyen: khanhnguyen21006@gmail.com

## Project Structure:
We organize the project according to the works of each week. The source code is pushed to the corresponding folder for the week, named WeekX.

### Week 1
- We use the script main.py to train the model.
- The presentation for the work done this week can be found [here](https://docs.google.com/presentation/d/1WBQWrhNQ5ybHa9XmfLkyJsPHMqfX2ilstxTsrWRO5X0/edit?usp=sharing).

### Week 2
- The script inference.py is used to obtain inference results by changing the model configuration.
- train.py is used to train Faster R-CNN on KITTI dataset. Hyper-parameters need to be fixed in the script.
- The presentation for the work done this week can be found [here](https://docs.google.com/presentation/d/1Cxh8sIgiQTOaXbjc6ygp3B-W5Z3nnfGUZHmHK0JMc9c/edit?usp=sharing).
- A short report in statr-of-the-art Object Detection approaches applying deep models [Overleaf project](https://www.overleaf.com/read/vgdjrgsjdfqw).

### Week 3
- Run the inference_evaluate.py to obtain the results (AP metric, resulting images) for task b - c, the configuration has to be specified by modifying the script.
- For the training task d, go the train_evaluate.py, play with some hyper-parameters (learning rate, batch size, number of ROIs etc.) and run.
- The train-validation loss curve can be plotted by running plot.py, pointing to the folder containing metrics.json. An alternative way to visualize the training process is to use tensorboard. Run: tensorboard --logdir /path/to/metrics.json.
- The presentation for the work done this week can be found [here](https://docs.google.com/presentation/d/1wvgrYZm9FmR1pt6ufvdRt9kGv2YG0dZ4DiQmTdigVZ4/edit?usp=sharing).
- Overleaf report with Experiment section updated [Overleaf project](https://www.overleaf.com/read/vgdjrgsjdfqw).

### Week 4