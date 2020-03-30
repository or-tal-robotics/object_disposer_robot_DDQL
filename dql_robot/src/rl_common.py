#!/usr/bin/env python
import numpy as np
class ReplayMemory():
    def __init__(self, size = 50000, frame_height = 84, fram_width = 84, agent_history_lenth = 4, batch_size = 32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = fram_width
        self.agent_history_lenth = agent_history_lenth
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype = np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        
        self.frames = np.empty((self.size,self.frame_height, self.frame_width), dtype = np.uint8)
        self.frames_lsr = np.empty((self.size,self.frame_height, self.frame_width), dtype = np.uint8)

        
        self.terminal_flags = np.empty(self.size, dtype = np.bool)
        
        self.states = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)

        self.states_lsr = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)
        self.new_states_lsr = np.empty((self.batch_size,self.agent_history_lenth,self.frame_height,self.frame_width), dtype=np.uint8)
        
        
        self.indices = np.empty(self.batch_size, dtype = np.int32)
        
    def add_experience(self, action, frame, frame_lsr, reward, terminal):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Frames dimansions are wrong!')
        
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.frames_lsr[self.current, ...] = frame_lsr
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        
        self.count = max(self.count , self.current + 1)
        self.current = (self.current + 1) % self.size
        
    def get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_lenth-1:
            raise ValueError("Index must be over 3!")
        return self.frames[index-self.agent_history_lenth+1:index+1, ...],self.frames_lsr[index-self.agent_history_lenth+1:index+1, ...]
    
    def get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = np.random.randint(self.agent_history_lenth, self.count - 1)
                if index < self.agent_history_lenth:
                    continue
                if index >= self.current and index - self.agent_history_lenth <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_lenth].any():
                    continue
                break
            self.indices[i] = index
    
    def get_minibatch(self):
        if self.count < self.agent_history_lenth:
            raise ValueError('Not enough history to get a minibatch!')
        
        self.get_valid_indices()
        for i, idx in enumerate(self.indices):
            self.states[i],self.states_lsr[i] = self.get_state(idx - 1)
            self.new_states[i],self.new_states_lsr[i] = self.get_state(idx)
        
        return np.transpose(self.states, axes=(0,2,3,1)), \
            np.transpose(self.states_lsr, axes=(0,2,3,1)), \
            self.actions[self.indices], \
            self.rewards[self.indices], \
            np.transpose(self.new_states, axes=(0,2,3,1)), \
            np.transpose(self.new_states_lsr, axes=(0,2,3,1)), \
            self.terminal_flags[self.indices]
            

def update_state(state, obs_small,lsr_state,lsr_small):
    return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis = 2) ,np.append(lsr_state[:,:,1:], np.expand_dims(lsr_small, 2), axis = 2)

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    states,states_lsr, actions, rewards, next_states,next_states_lsr, dones = experience_replay_buffer.get_minibatch()
    next_Qs = target_model.predict([next_states,next_states_lsr])
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
    loss = model.train_step(states.astype(np.float32),states_lsr.astype(np.float32), actions, targets)
    return loss