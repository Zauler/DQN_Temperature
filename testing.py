#TESTING PHASE 
#IMPORT PYTHON LIBRARIES AND OTHER PYTHON FILES
from keras.models import load_model

import os
import numpy as np 
import random as rn 
import environment

#Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#CONFIGURATION OF THE PARAMETERS 
number_actions = 5 
direction_boundary = (number_actions-1)/2
temperture_step = 1.5


#CONSTRUCTION OF THE ENVIRONMENT BY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users=20, initial_rate_data = 30)
#LOADING OF A PRE-TRAINED MODEL
model = load_model("model.h5")
# CHOICE OF TRAINING MODE
train = False

#ONE YEAR'S SIMULATION RUN IN INFERENCE MODE
env.train = train
current_state, _ , _ = env.observe()
for timestep in range(0, 30*12*24*60): #minutes of the year
    
    #EXECUTE THE FOLLOWING ACTION BY INFERENCE
    q_values = model.predict(current_state, verbose= False)
    action = np.argmax(q_values[0])
    
    #calculation of direction of action and energy used.
    if action < direction_boundary :
            direction =-1
    else:
        direction = 1
    
    energy_ai = abs(action - direction_boundary) * temperture_step
    next_state, reward, game_over = env.update_env(direction=direction, energy_ai=energy_ai, month= int(timestep / (30*24*60)) )
    current_state = next_state
            
#PRINT TRAINING RESULTS AT THE END OF THE EPOCH
print("\n")
print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(env.total_energy_ai))
print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
print(" - Energía ahorrada: {:.1f} %".format( ( (env.total_energy_noai - env.total_energy_ai) / (env.total_energy_noai) )*100 ) )
