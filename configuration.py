###############################################################################
#                            Net Structure Configuration                      #
###############################################################################
# Noise size
NOISE_SIZE = 100
BASE_CHANNAL_DIM = 64
OUTPUT_DIM = 3

###############################################################################
#                              Training Configuration                         #
###############################################################################
TRAIN_BATCH_SIZE = 64
NUM_EPOCHS = 100000
MAX_ITERATIONS = 200000
LEARNING_RATE = 0.0002

###############################################################################
#                             Loss Weight Configuration                       #
###############################################################################


###############################################################################
#                            Dataset Configuration                            #
###############################################################################
# Dataset configuration
DATASET_DIR = './dataset'

# Dataset Root for two domain data
DATA_ROOT = './dataset/Animals_with_Attributes2'

# Experiment Ouput Directory
IMAGE_SAVING_DIRECTORY = './checkpoint/outputs'
MODEL_SAVEING_DIRECTORY = './checkpoint/models'
