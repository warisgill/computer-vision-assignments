# Comp451 HW3

This assignment is adapted from [Stanford Course CS231n](http://cs231n.stanford.edu/).

In this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO.

The goals of this assignment are as follows:

- Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
- Understand and implement Vanilla RNN.

## Setup Instructions

**Installing Anaconda:** If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.7. We are no longer supporting Python 2.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n comp451 python=3.7 anaconda`

to create an environment called comp451.

Then, to activate and enter the environment, run

`source activate comp451`

To exit, you can simply close the window, or run

`source deactivate comp451`

Note that every time you want to work on the assignment, you should run `source activate comp451` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

In order to use the correct version of `scipy` for this assignment,
run the following after your first activation of the environment:

`conda install scipy=1.1.0`

## Download data:

Once you have the starter code (regardless of which method you choose above), you will need to download the COCO captioning dataset. Make sure `wget` is installed on your machine before running the commands below. Run the following from the assignment3 directory:

```
cd comp451/datasets
./get_assignment3_data.sh
```

## Start IPython:

After you have the COCO captioning data, you should start the IPython notebook server from the assignment3 directory, with the jupyter notebook command.

If you are unfamiliar with IPython, you can also refer to our [IPython tutorial](http://cs231n.github.io/ipython-tutorial/).

## Grading

Q1: Vanilla RNN (100 points)

## Submission

Zip (do not use RAR) the assignment folder using the format `username_studentid_assignment3.zip`.
Email the zip file to the instructor and TA. Do not include large files in the submission (for
instance data files under `./comp451/datasets/*`).

## Notes

NOTE 1: Make sure that your homework runs successfully. Otherwise, you may get a zero grade from the assignment.

NOTE 2: There are # *****START OF YOUR CODE/# *****END OF YOUR CODE tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

NOTE 3: The assignment1 code has been tested to be compatible with python version 3.7 (it may work with other versions of 3.x, but we won’t be officially supporting them). You will need to make sure that during your virtual environment setup that the correct version of python is used. You can confirm your python version by (1) activating your virtualenv and (2) running which python.

NOTE 4: If you are working in a virtual environment on OSX, you may potentially encounter errors with matplotlib due to the [issues described here](https://matplotlib.org/faq/virtualenv_faq.html). In our testing, it seems that this issue is no longer present with the most recent version of matplotlib, but if you do end up running into this issue you may have to use the start_ipython_osx.sh script from the assignment1 directory (instead of jupyter notebook above) to launch your IPython notebook server. Note that you may have to modify some variables within the script to match your version of python/installation directory. The script assumes that your virtual environment is named .env.

## Troubleshooting

**macOS**

If you are having problems with matplotlib (e.g. imshow), try running this:

`conda install python.app`
