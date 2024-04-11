#CREATING THE ENVIRONMENT 
#IMPORT LIBRARIES

import numpy as np 


#CONSTRUCTING THE ENVIRONMENT IN A CLASSROOM
class Environment(object):
    
    #INPUT AND INITIALISE ENVIRONMENT PARAMETERS AND VARIABLES
    def __init__(self,optimal_temperature = (18.0,24.0), initial_month = 0, initial_number_users=10, initial_rate_data = 60 ):
        self.monthly_atmospheric_temperature = [1.0,5.0,7.0,10.0,11.0,20.0,23.0,24.0,22.0,10.0,5.0,1.0]
        self.initial_month = initial_month 
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.optimal_temperature = optimal_temperature 
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_date = 20
        self.max_rate_data = 300
        self.max_update_rate = 10
        self.initial_number_users = initial_number_users
        self.current_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data 
        self.temperature_ai = self.intrinsec_temperature 
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0 
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0 
        self.reward = 0.0
        self.game_over = 0 
        self.train = 1 # 1 = TRAINING, 0 = TESTING
    
    #CREATE A METHOD THAT WILL UPDATE THE ENVELOPE RIGHT AFTER AN ACTION IS EXECUTED
    def update_env(self, direction, energy_ai, month):
        #GETTING THE REWARD
        
        #calculate the energy expended by the cooling system without IA
        energy_noai = 0
        if self.temperature_noai < self.optimal_temperature[0]:
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif self.temperature_noai > self.optimal_temperature[1]:
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        #calculate the reward
        self.reward = energy_noai - energy_ai
        #Scaling the reward 
        self.reward = 1e-3 * self.reward  #scale factor, we normalise between -20 to 80, difference of 100 then between 1000 that's why we are cool.

        
        #OBTAINING THE FOLLOWING STATE
        
        #Update atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        #Update the number of users
        self.current_number_users += np.random.randint(-self.max_number_users,self.max_number_users)
        if (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users
        elif (self.current_number_users > self.max_number_users):
            self.current_number_users = self.max_number_users
        
        #Updating the data transfer rate
        self.current_rate_data += np.random.randint(-self.max_update_rate,self.max_update_rate)
        if (self.current_rate_data < self.min_rate_date):
            self.current_rate_data = self.min_rate_date
        elif (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        
        #Calculate intrinsic temperature
        past_intrinsic = self.intrinsec_temperature
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data 
        delta_intrinsec_temperature = self.intrinsec_temperature - past_intrinsic
        #Calculate the temperature of the AI system
        if direction == -1:
            delta_temperature_ai = -energy_ai 
        elif direction == 1:
            delta_temperature_ai = energy_ai
        #Calculate the new server temperature with AI connected.
        self.temperature_ai += delta_intrinsec_temperature + delta_temperature_ai
        #New server tempoerature when no AI connected
        self.temperature_noai += delta_intrinsec_temperature
        
        #OBTAINING THE GAME OVER VARIABLE 
        if self.temperature_ai < self.min_temperature:
            if self.train ==1:
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]

        
        if self.temperature_ai > self.max_temperature:
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1] 
                self.temperature_ai = self.optimal_temperature[1]
                
        #UPDATE SCORES
        #calculate the total energy expended by the AI
        self.total_energy_ai += energy_ai
        #calculate the total energy expended by the cooling system without IA
        self.total_energy_noai += energy_noai
        
        #SCALE THE NEXT STATE
        scaled_temperature_ai  = (self.temperature_ai - self.min_temperature) / ( self.max_temperature - self.min_temperature )
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_date) / (self.max_rate_data - self.min_rate_date)
        next_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])
        
        #RETURNING THE NEXT STATE, THE REWARD AND THE GAME OVER
        return next_state, self.reward, self.game_over
    
    #CREATE A METHOD THAT RESTARTS THE ENVIRONMENT
    def reset(self, new_month=0):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data 
        self.temperature_ai = self.intrinsec_temperature 
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0 
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0 
        self.reward = 0.0
        self.game_over = 0 
        self.train = 1 # 1 = TRAINING, 0 = TESTING
        
    #CREATE A METHOD THAT GIVES US AT ANY TIME THE CURRENT STATUS, THE LAST REWARD AND WHETHER THE GAME IS OVER OR NOT.
    def observe(self):
        scaled_temperature_ai  = (self.temperature_ai - self.min_temperature) / ( self.max_temperature - self.min_temperature )
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_date) / (self.max_rate_data - self.min_rate_date)
        current_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])
        
        return current_state, self.reward, self.game_over
