import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from collections import deque
import random
import json
import time
class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x




class Simple2DEnvironment(gym.Env):
    def __init__(self, size=10, n_ob=10):
        super(Simple2DEnvironment, self).__init__()

        self.size = size
        self.n_ob = n_ob
        # define state space
        self.action_space = spaces.Discrete(4)
        # observation space
        self.observation_space = spaces.Box(low=-1, high=self.size, shape=(10,), dtype=np.int32)
        # define state 
        self.state = None
        # define obstacles
        self.obstacles = None
        # obstacles target
        self.target = None

        self.maps = []
        
        
    def save_maps(self, filename):
        with open(filename, 'w') as f:
            maps = [[list(obstacle) for obstacle in map_data["obstacles"]] + [list(map_data["target"])] for map_data in self.maps]
            json.dump(maps, f)

    def load_maps(self, filename):
        with open(filename, 'r') as f:
            maps = json.load(f)
            self.maps = [{"obstacles": [tuple(obstacle) for obstacle in map_data[:-1]], "target": tuple(map_data[-1])} for map_data in maps]


    def step(self, action):
        x, y = self.state
        old_state = self.state

        # update state
        if action == 0:  # up
            new_y = min(y+1, self.size-1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y-1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x-1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x+1, self.size-1)
            x = new_x

        self.state = (x, y)

        done = self.state == self.target  # done if arrive target
        if self.state == self.target or self.state[1] == 0:
            arrive_target = True
            arrive_land = True
            done = True
        else:
            arrive_target = False
            arrive_land = False
            done = False
            

        reward = -10 if self.state in self.obstacles else 100 if arrive_target else -100 if arrive_land else -5
        # If hit an obstacle, the state returns to the state before the collision
        if self.state in self.obstacles:
            self.state = old_state

        state = self.get_surrounding_cells()
        state = np.insert(state, 0, self.state)

        #self.trajectory.append(self.state)
        return state, reward, done, {}
    
    def step_t(self, action):
        x, y = self.state
        old_state = self.state

        # update state
        if action == 0:  # up
            new_y = min(y+1, self.size-1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y-1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x-1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x+1, self.size-1)
            x = new_x

        self.state = (x, y)

        done = self.state == self.target  # done if arrive target
        if self.state == self.target or self.state[1] == 0:
            arrive_target = True
            arrive_land = True
            done = True
        else:
            arrive_target = False
            arrive_land = False
            done = False
            

        reward = -10 if self.state in self.obstacles else 100 if arrive_target else -100 if arrive_land else -5
        # If hit an obstacle, the state returns to the state before the collision
        if self.state in self.obstacles:
            self.state = old_state

        state = self.get_surrounding_cells()
        state = np.insert(state, 0, self.state)
        
        self.trajectory.append(self.state)
        return state, reward, done, {}

    def reset(self, map_index=None):
        if map_index is None:

            self.state = (int(self.size/2), self.size-1)

            self.target = (int(self.size/2), 0) 

            self.trajectory = [self.state]
            # create or load your own map and environment

            self.obstacles=[(np.random.randint(1,self.size-1),np.random.randint(1,self.size-1))for _ in range(10)]

            self.obstacles += [(i, -1) for i in range(self.size+2)]
            self.obstacles += [(i, self.size) for i in range(self.size+2)]
            self.obstacles += [(-1, i) for i in range(-1, self.size+1)]
            self.obstacles += [(self.size, i) for i in range(-1, self.size+1)]

            # save maps
            map_data = {"obstacles": self.obstacles, "target": self.target}
            self.maps.append(map_data)
        else:
            # if map index is provided, load it
            map_data = self.maps[map_index]
            self.state = (int(self.size/2), self.size-1)
            self.target = map_data['target']
            self.obstacles = map_data['obstacles']

        state = self.get_surrounding_cells()
        state = np.insert(state, 0, self.state)
        return state

    
      
    
    def get_surrounding_cells(self):
        surrounding = np.full(8, 0)  # initialize with -1
        x, y = self.state
       
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)] 
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) == self.state:
                    surrounding[i] = 1  # agent
                elif (new_x, new_y) == self.target:
                    surrounding[i] = 2  # target
                elif (new_x, new_y) in self.obstacles:
                    surrounding[i] = 3  # obstacle
            else:
                surrounding[i] = 3  # edge is considered as obstacle
        return surrounding




    def render(self, mode='human'):
        for y in range(self.size+1, -2, -1):  
            for x in range(-1, self.size+1):  
                if self.state == (x, y):
                    print('X', end='')
                elif self.target == (x, y):
                    print('T', end='')
                elif (x, y) in self.obstacles:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()


    def render_with_trajectory(self):
        print("\nMap with Trajectory:")
        output=''
        for y in range(self.size+1, -2, -1):  
            for x in range(-1, self.size+1):
                if (x, y) == self.state:
                    output += 'A'
                    print('A', end='')  # A for Agent's current position
                elif (x, y) == self.trajectory[0]:
                    output += 'S'
                    print('S', end='')  # S for Start
                elif (x, y) in self.trajectory:
                    output += '*'
                    print('*', end='')  # * for Trajectory
                elif (x, y) == self.target:
                    output += 'T'
                    print('T', end='')  # T for Target
                elif (x, y) in self.obstacles:
                    output += '#'
                    print('#', end='')  # # for Obstacle
                else:
                    output += ' '
                    print(' ', end='')

        #     print()
        # file_path = "map_with_trajectory.txt"  # Specify the file path
        # with open(file_path, 'w') as file:
        #     file.write(output)
        #     print(f"Map with trajectory saved to {file_path}")
class DQNAgent():
    def __init__(self, state_dim, action_dim, replay_buffer):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.model = DQN(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.replay_buffer = replay_buffer
        self.loss_fn = nn.MSELoss(reduction='mean')

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)
        done       = torch.FloatTensor(done).to(device)

        q_values      = self.model(state)
        next_q_values = self.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        expected_q_value1 = expected_q_value-5
        loss = self.loss_fn(q_value, expected_q_value.detach())
        loss1= self.loss_fn(q_value, expected_q_value1.detach())
        #print(loss)
        #print(loss1)
        # Define the output you want to save
       
    

        # file_path = "loss_values.txt"  # Specify the file path
        # with open(file_path, 'a') as file:  # Open the file in append mode ('a')
        #     # Write the loss values to the file
        #     file.write(f"Loss: {loss.item()}, Loss1: {loss1.item()}\n")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state   = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_value = self.model.forward(state)
        return q_value.max(1)[1].data[0]

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
env = Simple2DEnvironment(10,10)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n
buffer = ReplayBuffer(1000)
agent = DQNAgent(state_dim, action_dim, buffer)

episodes = 1000
batch_size = 32
epsilon = 1.0
rewards = []    


start_time = time.time()  
map_id = -1
all_step = 0
steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
for i_episode in range(n_episodes):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        state = env.reset()
        #state = env.reset(map_index=0)
        map_id = map_id + 1
        #print(all_step/each_map)
       
        all_step = 0
        epsilon = 1.0
        total_reward = 0
        #env.render()
        
    else:
        if map_id==1:
             state = env.reset(map_index=0)
        else:
        #state = env.reset(map_index=0)
            state = env.reset(map_index=map_id)
    done = False
    steps = 0  
    #print(i_episode)
    while not done and steps < 200:  
        #action = agent.get_action(state, epsilon)
        #print(state)
        action = agent.get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        if map_id != 0:
            buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
        total_reward += reward
        state = next_state
        steps += 1  
        #env.render()
    
    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
    rewards.append(total_reward)
    epsilon *= 0.995
end_time = time.time() 

print(f'Map {21} Reward: {total_reward/each_map} steps: {all_step/each_map}')
print(f'train time: {end_time-start_time} ')



start_time = time.time()  
map_id = -1
all_step = 0
# test
steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
epsilon = 0
rewards = []
print('test')
for i_episode in range(n_episodes):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        map_id = map_id + 1
        state = env.reset(map_index=map_id)

        all_step = 0
        epsilon = 1.0
        total_reward = 0
        #env.render()
    else:
        state = env.reset(map_index=map_id)
        
    done = False
    steps = 0 
    #print(i_episode)
    while not done and steps < 100:  

        action = agent.get_action(state, epsilon=0.05) 
        next_state, reward, done, _ = env.step_t(action)
        total_reward += reward
        state = next_state
        steps += 1
        #env.render()
       
    rewards.append(total_reward)
    #print(steps)
    # if map_id==0:
    #     env.render_with_trajectory()
    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
end_time = time.time() 
print(f'test time: {end_time-start_time} ')

def shuffle_three_parts(lst,change):
    indices = [i for i, x in enumerate(lst[2:], start=2) if x >-5]

    chosen_indices = np.random.choice(indices, change, replace=False)

    np.random.shuffle(chosen_indices)

    lst_new = lst.copy()
    for i, index in enumerate(sorted(chosen_indices)):
        lst_new[index] = lst[chosen_indices[i]]

    return lst_new    


start_time = time.time()  
map_id = -1
all_step = 0
# unlearning
steps = 0 
n_episodes = 200
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
print('unlearning')
for i_episode in range(n_episodes):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        state = env.reset()
        map_id = map_id + 1
        #print(all_step/each_map)
        
        all_step = 0
        epsilon = 1.0
        total_reward = 0
        #env.render()
    else:

        state = env.reset(map_index=map_id)
    done = False
    steps = 0 
    while not done and steps < 100: 

        if map_id == 0:
            action = agent.get_action(state, 1)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
        else:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            agent.update(batch_size)
        total_reward += reward
        state = next_state
        steps += 1  

    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
    rewards.append(total_reward)
    epsilon *= 0.995
end_time = time.time() 
print(f'unlearn time: {end_time-start_time} ')


start_time = time.time()     
map_id = -1
all_step = 0
steps = 0 
n_episodes = 1000
n_maps = 20
each_map = n_episodes/n_maps
total_reward = 0
epsilon = 0
rewards = []
print('test')
for i_episode in range(n_episodes):
    all_step = all_step + steps
    if i_episode%each_map ==0 :
        map_id = map_id + 1
        state = env.reset(map_index=map_id)
        
        all_step = 0
        epsilon = 1.0
        total_reward = 0
        #env.render()
    else:
        state = env.reset(map_index=map_id)
        
    done = False
    steps = 0  

    while not done and steps < 100:  
        action = agent.get_action(state, epsilon=0.05) 
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1  
    rewards.append(total_reward)
    # if map_id==0:
    #      env.render_with_trajectory()
    if (i_episode+1)%each_map ==0 :
        print(f'Map {map_id} Reward: {total_reward/each_map} steps: {all_step/each_map}')
end_time = time.time() 
print(f'unlearn test time: {end_time-start_time} ')
