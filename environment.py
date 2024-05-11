# Creating the Environment

# Importing libraries
import numpy as np

#  BUILDING THE ENVIRONMENT IN A CLASSROOM
class Environment(object):
    # ENTER AND INITIALISE ENVIRONMENT PARAMETERS AND VARIABLES
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
    
    # CREATE A METHOD THAT UPDATES THE ENVIRONMENT RIGHT AFTER THE IA EXECUTES AN ACTION
    def update_env(self, direction, energy_ai, month):
        # OBTAINING THE REWARD
        
        # Calculate the energy expended by the server's cooling system without AI
        energy_noai = 0
        if(self.temperature_noai  < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif(self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        # Calculate the reward
        self.reward = energy_noai - energy_ai
        # Scaling the reward
        self.reward = 1e-3*self.reward
        
        # OBTAINING THE FOLLOWING STATUS
        
        # updating the atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        # Update the number of users
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if(self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        elif(self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        # Update the data transfer rate
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if(self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data
        elif(self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        # Calculate the intrinsic temperature variation
        past_intrinsic_temperature =  self.intrinsec_temperature
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        delta_intrinsec_temperaure = self.intrinsec_temperature - past_intrinsic_temperature
        # Calculate the temperature variation caused by AI
        if(direction==-1):
            delta_temperature_ai = -energy_ai
        elif(direction == 1):
            delta_temperature_ai = energy_ai
        # Calculate the new server temperature when AI is online
        self.temperature_ai += delta_intrinsec_temperaure + delta_temperature_ai
        # Calculate the new temperature of the server when no AI is connected
        self.temperature_noai += delta_intrinsec_temperaure
        
        # OBTAINING THE GAME OVER
        if(self.temperature_ai < self.min_temperature):
            if(self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]
        if(self.temperature_ai > self.max_temperature):
            if(self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                self.temperature_ai = self.optimal_temperature[1]
                
        # UPDATE THE SCORES
        
        # calculate total energy expended by the AI
        self.total_energy_ai += energy_ai
        # Calculate the total energy expended by the server's cooling system without AI.
        self.total_energy_noai += energy_noai
        
        # SCALE THE FOLLOWING STATE
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        # RETURN NEXT STATE, REWARD AND GAME OVER
        return next_state, self.reward, self.game_over
    
    # CREATE A METHOD THAT RESTARTS THE ENVIRONMENT
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25*self.current_number_users+1.25*self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

   # CREATE A METHOD THAT GIVES US AT ANY INSTANT THE CURRENT STATUS, THE LAST REWARD AND THE GAME OVER VALUE.
    def observe(self):
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/(self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users)/(self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/(self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        
        return current_state, self.reward, self.game_over
        
        
        
        
        
        
        
        
        
    
