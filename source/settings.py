'''
global settings that should be available to any module in the project.
'''


import os


# project level configs - gets the top level directory for the whole project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# filter file paths
GIVEN_NAMES_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data/configuration/given_names.txt'))
FAMILY_NAMES_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data/configuration/family_names.txt'))

# data path - use a sample of 1,000 trasncriptsfor prototyping.
#             post pipeline-prototyping, we'll scale up to the full 25K+ available transcripts.
INPUT_FILE_PATH_SAMPLE = os.path.abspath(os.path.join(PROJECT_ROOT, 'data/inputs/voice_data_v3_sample.txt'))
INPUT_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'data/inputs/voice_data_v3.txt'))


# note: the large transcript file doesn't live in the repo.
#       you'll need to downlaod it and update the definition of 'INPUT_FILE_PATH'
#       to match the downloaded file's path.


