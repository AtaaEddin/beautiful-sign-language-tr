| Flag | Values | Defualt | Description |
| -- | -- | -- | -- |
| run | REST_API - wamp - webcam | REST_API | options on how to test or deploy the system |
| sys | any folder inside 'checkpoints' conatins weights and classes | turkish_10_word | which trained weights and classes to use |
| download | True - False | False | download weights and classes to checkpoints directory |
| on_cpu | True - False | True | False => to run the system on the GPU |
| pred_type | word - sentence | word | choose between 'word-base system' and sentence-base system |
| nTop | a number between 1 --> 10 | 3 | number of predicted word (outputs) sorted |
| mul_oflow | True - False | True | faster optical flow calculation with multiprocessing |
| oflow_pnum | int number | 4 | number of processes to parallelize the optical flow algorithm|
| mul_2stream | True - False | True | Parallelizing the nerual network model |
| rgb | True - False | False | True means just use rgb nerual network and disable the rest of nerual network pipeline False means use rgb with the rest of the nerual network pipeline |
| oflow | True - False | False | True means just use oflow nerual network and disable the rest of nerual network pipeline False means use oflow with the rest of the nerual network pipeline |
| use_lstm | True - False | False | add lstm nerual network architecture on top of the nerual network pipeline | 