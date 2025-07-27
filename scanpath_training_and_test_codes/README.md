
## Test Codes
## Test Codes
* To perform Artificially Generated Visual Scanpath Prediction model.
* Store your test images inside the 'new_test' folder in the [dataset](https://github.com/ashishverma03/SDC/tree/main/scanpath_training_and_test_codes/dataset) folder in the scanpath_training_and_test_codes folder. 
* To run the code, you need to download the pre-trained scanpath prediction model from [here](https://drive.google.com/drive/folders/16-olj4gRIRdJ7iUKgGKKaNZ5h9y6wGMj), and extract files to the [models](https://github.com/ashishverma03/SDC/tree/main/scanpath_training_and_test_codes/models) folder in the scanpath_training_and_test_codes folder.
```
python new_sample_lstm.py
```
* Artificially generated scanpaths will be stored at 'result_scanpaths' folder.
## Train Codes

### Train Scanpath Prediction Code

* The pre-processed REFLACX dataset used in training is provided [here](https://drive.google.com/drive/folders/1yHONIc_4RMtzQFvR-qhHiNmlN8mEN71h). The images are resized to 256x256 and the corresponding scanpaths are scaled down to match the image size. Download it and keep extracted train, val and test folders in the [dataset](https://github.com/ashishverma03/SDC/tree/main/scanpath_training_and_test_codes/dataset) folder in the in the scanpath_training_and_test_codes folder. 

  ``` run the codes
  python lstm_train.py
  ```
* Check the [models](https://github.com/ashishverma03/SDC/tree/main/scanpath_training_and_test_codes/models) folder in the scanpath_training_and_test_codes folder for trained models.
* The 'sample_lstm.py' is code to generate artificial scanpaths on test images of the REFLACX dataset and compute SCANMATCH scores to select best model. 
```
python sample_lstm.py
```



## Contact
Ashish Verma: verma.bao[at]gmail[dot]com
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
