# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + \
            list(permutations([i for i in range(1,m+1)], 2))
        self.state_space = [[x, y, z]
                            for x in range(1,m+1) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]-1] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])

        return possible_actions_index+[0],actions   


    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # Get the next state and the various time durations
        next_state, wait_time, transit_time, ride_time = self.next_state_func(
            state, action, Time_matrix)

        # Calculate the reward based on the different time durations
        rewards = self.reward_func(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time
        
        return rewards, next_state, total_time   


    def reward_func(self, wait_time, transit_time, ride_time):
        """Takes in state, action and Time-matrix and returns the reward"""
        # transit and wait time yield no revenue, only battery costs, so they are idle times.
        passenger_time = ride_time
        idle_time      = wait_time + transit_time
        
        reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current  location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        curr_time = state[1]
        curr_day = state[2]
        """
         3 Scenarios: 
           a) Refuse all requests
           b) Driver is already at pick up point
           c) Driver is not at the pickup point.
        """    
        if ((pickup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            next_loc = curr_loc
        elif (curr_loc == pickup_loc):
            # means driver is already at pickup point, wait and transit are both 0 then.
            ride_time = Time_matrix[curr_loc-1][drop_loc-1][curr_time][curr_day]
            
            # next location is the drop location
            next_loc = drop_loc
        else:
            # Driver is not at the pickup point, he needs to travel to pickup point first
            # time take to reach pickup point
            transit_time      = Time_matrix[curr_loc-1][pickup_loc-1][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The driver is now at the pickup point
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc-1][drop_loc-1][new_time][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        
        # Construct next_state using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state, wait_time, transit_time, ride_time


    def update_time_day(self, time, day, ride_duration):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            time = time + ride_duration
            # day is unchanged
        else:
            # duration taken spreads over to subsequent days
            # convert the time to 0-23 range
            time = (time + ride_duration) % 24 
            
            # Get the number of days
            num_days = (time + ride_duration) // 24
            
            # Convert the day to 0-6 range
            day = (day + num_days ) % 7

        return time, day


    def reset(self):
        return self.action_space, self.state_space, self.state_init
