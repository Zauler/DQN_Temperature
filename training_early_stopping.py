
# AI training phase

# Importing libraries and other python files
import os
import numpy as np
import random as rn

import environment
import brain
import dqn

# Configure seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(22)
rn.seed(54321)

# PARAMETER SETTINGS 
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions -1)/2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CONSTRUCTION OF THE ENVIRONMENT BY CREATING AN OBJECT OF THE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CONSTRUCTION OF THE BRAIN BY CREATING AN OBJECT OF CLASS BRAIN
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CONSTRUCTION OF THE DQN MODEL BY CREATING AN OBJECT OF THE DQN CLASS
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

# CHOICE OF TRAINING MODE
train = True

# TRAINING THE IA
env.train = train
model = brain.model

early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if (env.train):
    # BEGIN ALL EPOCHES (1 Epoch = 5 months)
    for epoch in range(1, number_epochs):
         # INITIALISATION OF ENVIRONMENT AND TRAINING LOOP VARIABLES
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)  
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # INITIALISATION OF TIMESTEPS (Timestep = 1 minute) IN ONE EPOCA
        while ((not game_over) and (timestep <= 5*30*24*60)):
            # EXECUTE THE FOLLOWING ACTION BY SCANNING
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                   
            # EXECUTE THE FOLLOWING ACTION BY INFERENCE
            else: 
                q_values = model.predict(current_state,verbose=0)
                action = np.argmax(q_values[0])
            
            if (action < direction_boundary):
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step
            
            # UPDATE THE ENVIRONMENT AND REACH THE FOLLOWING STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            
            # STORING THE NEW TRANSITION IN MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            # OBTAIN THE TWO SEPARATE BLOCKS OF INPUTS AND TARGETS
            inputs, targets = dqn.get_batch(model, batch_size)
            
            # CALCULATE THE LOSS FUNCTION USING THE ENTIRE INPUT BLOCK AND OBJECTIVES
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
        # PRINT THE TRAINING RESULTS AT THE END OF THE EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print("Total energy expended by the system with AI: {:.0f} J.".format(env.total_energy_ai))
        print("Total energy expended by the system without AI: {:.0f} J.".format(env.total_energy_noai))
        
        # EARLY DETENTION
        if early_stopping:
            if (total_reward <= best_total_reward):
                patience_count += 1
            else:
                best_total_reward = total_reward
                patience_count = 0
                
            if patience_count >= patience:
                print("Premature execution of the method")
                break
        
        # SAVE THE MODEL FOR FUTURE USE
        model.save("model.h5")













