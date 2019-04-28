# beautiful sign language (Turkish version) 

this repository has been built to be a start point hoping to continue further in building full sign language system leveraging deep learning techniques to be able to help hear-impaired people, in this moment our system contains Deep learning model trained on some Turkish sign language [words](https://www.youtube.com/playlist?list=PLxVilKcX9J7Siru_n8Dy1NajiH0CR8EwP) collected with the help of friends.

<p align="center">
<img width="600" src="https://media.giphy.com/media/XoVtquOIFanZEg1cHq/giphy.gif"/> 
</p>


## Features
- **Functionality**
	- **word-base system**
		- for one sign language sentence be translated should be fed to the system as **words**, each word considered as independent Input.
		- accurate, simple and fast
		- Input space is discrete, means each video(input) is a word. 
	- **sentence-base system**
		- system can accept any kind of input word or sentence.
		- comparing with word-base system this system is more slower and less accurate.
		- Input space is continuous.
- **Input**
	system accepts a video as input with any known formats(flv,mp4,wbem, ...) and with any kind of camera device (webcam, mobile camera ...)
- **Output**
	system outputs 'n' sign language word ex:(n==3) sorted according to the top predicted probabilities(likelihoods) which is associated with each word.
- **Deployment**
	- **Rest API**
		serve the system as an API interface that listens to HTTP requests(GET) to get predictions of sign language words.
	- **WampServer**
		for testing purposes on local machine, wamp server stack(Apache, MySQL, PHP) is used with simple web interface demonstrating some useful use cases of the system. It could be developed to serve online but some addition work need to be done on both server and UI side.
- **Multiprocessing**
	the hole system pipeline is multi-processed for performance gain, but sometimes for some OS mutliprocessing low level libraries like windows it may cause bottleneck problems for some reasons.
- **CPU/GPU** 
	supports cpu and gpu, but to run it on GPU you need to install CUDA(Nivaida GPU) and cuDNN, head to [Tensorflow](https://www.tensorflow.org/install/gpu) website for more information
- **OS**: Ubuntu(18),Windows(10)


## Results

- **word-base system**

<p align="center">
<img width="600" src="https://media.giphy.com/media/fRaH9Pmox7r0UxevpH/giphy.gif"/> 
</p>

- **sentence-base system**

<p align="center">
<img width="600" src="https://media.giphy.com/media/kDHJVj8t55VwQoSllu/giphy.gif"/> 
</p>


## system sign language words

Right now the system can recognize these sign language [words](https://www.youtube.com/playlist?list=PLxVilKcX9J7Siru_n8Dy1NajiH0CR8EwP)

## Installation

the system is tested on ubuntu 18.4 and windows 10, but it should work fine for other OS. If you encountered any problem feel free to open an issue.

### Prerequisites

- python3
- Tensorflow >= 1.11.0
- Keras >= 2.2.4
- in addition(**optional**):
	- wamp(windows) or Xampp(linux)
	- CUDA 8 and cuDNN 7.4


clone the repo (or download it as .zip file) and install the requirements.

```bash
git clone https://github.com/AtaaEddin/alpha
cd alpha
pip install -r requirements
```
## Quick start

for quick testing on your webcam, run the following command. with 'run' flag webcam and 'download' flag True to download the model's weights you can turn it back to False after the first run.

While it's running you will do one of [these](https://www.youtube.com/playlist?list=PLxVilKcX9J7Siru_n8Dy1NajiH0CR8EwP) sign language words, after you do the word you will press 'q' to end recording and to start processing your video.

```bash
python main.py -run webcam -pred_type word -download True
```

Take a look to [Flags.md](https://github.com/AtaaEddin/beautiful-sign-language-tr/blob/master/Flags.md) file to see all the flags and what it do.

## serve the system as REST API

it is the most convenient way to use the system because it separates this system from your already existing system where you are trying to integrate sign language features to your system, and you don't want to mix up your dependences with this system's dependences or you don't want to code hole of your system in python then it's the way to go.

you can make your own machine serve as Rest API serves or you can head to one of the cloud computing services like amazon ([AWS sageMaker](https://aws.amazon.com/sagemaker/) - [lambda function](https://aws.amazon.com/lambda/) or [google-cloud-platform](https://console.cloud.google.com) and install the system on it.

After you install the system simply run:

```bash
python main.py -run REST_API -pred_type word
```

learn more about the flags [Flags.md](https://github.com/AtaaEddin/beautiful-sign-language-tr/blob/master/Flags.md)

## run the system on WampServer

[WampServer]((http://www.wampserver.com/en/)) is a Windows web development environment. So if you are on Ubuntu or linux you have to install [Xampp](https://www.apachefriends.org/tr/index.html).

- **On Windows** 
	- install wamp from [WampServer](http://www.wampserver.com/en/) site and download the version that suits your OS.
	- download necessary files from [here](https://drive.google.com/file/d/12a2i-jLCLGlHXKTFR2jd1PZFJv1DOpiW/view?usp=sharing) the folder named 'combine'. after that move the folder to 'path/to/wampdir/www/'
	- After you run WampServer, you should be able to go to  http://127.0.0.1/combine/. 
	- Go to http://127.0.0.1/phpmyadmin/. In the left there are Database icons press on 'New', name the database 'isaret' and from the drop-down menu select 'utf8_general_ci' then press create.
	- In the main.py change 'wamp_folder' to your path directory where you put combine folder.
	- Run the python server and start testing by running:
		`$ python main.py -run wamp`

- **On Ubuntu**
	- install [Xampp](https://www.apachefriends.org/tr/index.html).
	- download necessary files from [here](https://drive.google.com/file/d/12a2i-jLCLGlHXKTFR2jd1PZFJv1DOpiW/view?usp=sharing) the folder named 'combine'. after that move the folder to /opt/lampp/htdocs/
	- you can run Xampp with `$ sudo opt/lampp/lampp start` if you faced any troubles try to run Xampp through the XAMPP-Launcher, after you run it you should be able to go to  http://127.0.0.1/combine/ 
	- Go to http://127.0.0.1/phpmyadmin/. In the left there are Database icons press on 'New', name the database 'isaret' and from the drop-down menu select 'utf8_general_ci' then press create.
	-  run `$ sudo find /opt/lampp/htdocs -type d -exec chmod 755 {} \;` to allow read and write to somefolder like 'uploads'.
	- In the main.py make sure that 'wamp_folder' variable has '/opt/lampp/htdocs/combine/' as value.
	- Run the python server and start testing by running:
		`$ python main.py -run wamp`

learn more about the flags [Flags.md](https://github.com/AtaaEddin/beautiful-sign-language-tr/blob/master/Flags.md)

## Run on GPU

First install Tensorflow-gpu

```bahs
pip install tensorflow-gpu == 1.11.0
```

Then head to [Tensorflow](https://www.tensorflow.org/install/gpu) website under 'Software requirements' install all the requirements.



## Training

If you are interested in the training process and/or data preprocessing, just raise an issue and we'll discuss it there.

## How the system works

I'm publishing a new block on medium very soon, so stay tuned!!.

