#beautiful sign language (Turkish version) 

this repository has been built to be a start point hoping to continue further in building full sign language system leveraging deep learning techniques to be able to help hear-impaired people, in this moment our system conatins Deep learning model trained on only ten Turkish sign language words collected with the help of friends.


## Features
- **Functionality**
	- **word-base system**
		- one sign language sentence in order to be translated should be fed to the system as **words**, each word considered as independent Input.
		- accurate, simple and fast
		- Input space is descret, means each video(input) is a word. 
	- **sentence-base system**
		- system can accpet any kind of input word or sentence.
		- comparing with word-base system this system is more slower and less accurate.
		- Input space is continuous.
- **Deployment**
	- **Rest API**
		serve the system as an API interface thats listens to HTTP requessts(GET) to get predictions of sign language words.
	- **WampServer**
		for testing purposes on local machine, wamp server stack(Apache, MySQL, PHP) is used with simple web interface demostrating some usefull use cases of the system. It could be developed to serve online but some addition work need to be done on both server and UI side.
- **Multiprocessing**
	the hole system pipeline is multiprocessed for preformance gain, but sometimes for some OS mutliprocessing low level libraries like windows it may cause bottleneck probelms for some reasons.
- **CPU/GPU** 
	supports cpu and gpu, but to run it on GPU you need to install CUDA(Nivaida GPU) and cuDNN, head to [Tensorflow](https://www.tensorflow.org/install/gpu) website for more information
- **OS**: Ubuntu(18),Windows(10)

for further details, check my new block on mideuam explaining how it work in details. 

## Results

## Installation

the system is tested on ubuntun 18.4 and windows 10, but it should work fine for other OS. If you encountered any problem feel free to open an issue.

### Prerequisites

- Tensorflow >= 1.11.0
- Keras >= 2.2.4
- in addition(**optional**):
	- wamp(windows) or lamp(linux)
	- CUDA 8 and cuDNN 7.4

### Quick start





