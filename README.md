Python files for augmenting data and training machine learning models on simulations of the ESA SWARM satellites. Right now, only the data augmentation script is available.

# Installation
1. Create a Python virtual environment using `$ python3 -m venv venv`.

2. Activate the virtual environment using `$ source venv/bin/activate`. After installation, this will need to be done each session before using the script.

3. Install all the required packages with
`$ pip install -r requirements.txt`.

4. Run the script using the following format
`$ venv/bin/python ./src/<script name> <arguments>`.
This assures that we are using the right version of python with the right packages installed. To see the specific input format format for each script, run
`$ venv/bin/python ./src/augment -h`.

5. To exit the virtual environment when you are done
`$ deactivate`.


