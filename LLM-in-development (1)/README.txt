This directory is designed to help a novice readily spin up spaCy, QDRANT, openai API, and other tools useful in many parsing related problems. The goal is to minimize setup and illustrate how to use these technologies. 

You will need the following:

Docker -- Installed on your machine. 

OpenAI Key -- set as the environment variable "OPENAI_API_KEY" for your local machine.

Update pip if necessary: (> python.exe -m pip install --upgrade pip)

python virtual environment (bash: pip install virtualenv)

If you want to run this in a virtual environment using a jupyter notebook you can do the following:

First download the "LLMs_in_development" directory and save it in a location where you prefer to run virtual environments.

Open bash terminal. (Windows Powershell or any other linux based command line interpreter.)

Navigate to the "LLMs_in_development" folder. (> cd ./path/to/LLMs_in_development/directory)

Create virtual environment in the folder. (> python -m venv myenv)

Activate virtual environment. For Windows: (> myenv\Scripts\activate)  For Mac/Linux: (> source myenv/bin/activate)

If you have not done so already, install jupyter to work with jupyter notebooks (> pip install jupyter)

Launch Jupyter Notebook that has access to this folder (> jupyter notebook) or (> jupyter lab) if the notebook call does not work. 

It should automatically bring you to a locally hosted jupyter notebook with the directory's contents. 
You should now be able to open "LLMs_in_development.ipynb". 

The dependencies are at the top. Ensure you follow all steps listed at the beginning to avoid errors.