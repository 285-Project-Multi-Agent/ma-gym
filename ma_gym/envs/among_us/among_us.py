import copy
import logging

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

logger = logging.getLogger(__name__)


class AmongUs(gym.Env):
    """

    Among Us is an online multiplayer social deduction game developed and published by American game studio Innersloth 
    and released on June 15, 2018. The game takes place in a space-themed setting, in which players each take on one 
    of two roles, most being Crewmates, and a predetermined number being Impostors.

    Predator-prey involves a grid world, in which multiple predators attempt to capture randomly moving prey.
    Agents have a 5 × 5 view and select one of five actions ∈ {Left, Right, Up, Down, Stop} at each time step.
    Prey move according to selecting a uniformly random action at each time step.

    We define the “catching” of a prey as when the prey is within the cardinal direction of at least one predator.
    Each agent’s observation includes its own coordinates, agent ID, and the coordinates of the prey relative
    to itself, if observed. The agents can separate roles even if the parameters of the neural networks are
    shared by agent ID. We test with two different grid worlds: (i) a 5 × 5 grid world with two predators and one prey,
    and (ii) a 7 × 7 grid world with four predators and two prey.

    We modify the general predator-prey, such that a positive reward is given only if multiple predators catch a prey
    simultaneously, requiring a higher degree of cooperation. The predators get a team reward of 1 if two or more
    catch a prey at the same time, but they are given negative reward −P.We experimented with three varying P vales,
    where P = 0.5, 1.0, 1.5.

    The terminating condition of this task is when all preys are caught by more than one predator.
    For every new episodes , preys are initialized into random locations. Also, preys never move by themself into
    predator's neighbourhood
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(10, 10), n_agents=4, n_preys=3, n_imposter=1,
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, max_steps=100,
                 crewmate_capture_reward=10, persistent_rewards=False, alive_reward=10, 
                 enable_kill_cooldown=True, kill_cooldown=10, scenario=0
                 ):
        # project params
        self.persistent_kill_reward = persistent_rewards
        self.persistent_task_reward = persistent_rewards
        self.enable_kill_cooldown = enable_kill_cooldown
        self.kill_cooldown = kill_cooldown

        self._alive_reward = alive_reward
        self._crewmate_capture_reward = crewmate_capture_reward
        self._prey_capture_reward = prey_capture_reward

        self.scenario = scenario
        if self.scenario == 2:
            n_agents, n_preys, n_imposter = 4, 3, 1
            
        self._grid_shape = grid_shape

        self.n_agents = n_agents
        self.n_imposter = n_imposter
        self.n_crew = n_agents - n_imposter
        self.n_preys = n_preys

        self._num_crewmate_dead = 0
        self._num_tasks_finished = 0
        
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        
        self._agent_view_mask = (5, 5)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(6) for _ in range(self.n_agents)])
        
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}

        self._prey_alive = None
        self._agent_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        
        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        # self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0])
        # self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0])
        self._obs_high = np.array([1., 1.] + [3.] * mask_size + [1.0])
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0])

        if self.enable_kill_cooldown:
            self._imp_obs_high = np.array([1., 1.] + [4.] * mask_size + [1.0] + [self.kill_cooldown])
            self._imp_obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0] + [0.0])
            self._imp_cooldowns = np.array([self.kill_cooldown] * self.n_imposter)
        else:
            self._imp_obs_high = np.array([1., 1.] + [4.] * mask_size + [1.0])
            self._imp_obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0])
        
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)

        # obs_space_list = [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)]
        obs_space_list = []
        for i in range(self.n_imposter):
            obs_space_list.append(spaces.Box(self._imp_obs_low, self._imp_obs_high))
        for i in range(self.n_imposter, self.n_agents):
            obs_space_list.append(spaces.Box(self._obs_low, self._obs_high))
        self.observation_space = MultiAgentObservationSpace(obs_space_list)

        self._total_episode_reward = None
        self.seed()



    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR)


    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid
    
    def __create_medium_grid(self):
        _grid = [[PRE_IDS['wall']  if cell==1 else PRE_IDS['empty'] for cell in row] for row in MEDIUM_GRID]
        self._grid_shape = (len(_grid), len(_grid[1]))
        return _grid

    def __init_rnd_starting(self):
        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1), self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)


        for prey_i in range(self.n_preys):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1), self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)
    
    def __init_fixed_medium(self):
        # 3 tasks, 3 crewmates, 1 imposter
        # imposter
        self.agent_pos[0] = [0, 3]
        
        # crewmates
        self.agent_pos[1] = [1, 5]
        self.agent_pos[2] = [8, 5]
        self.agent_pos[3] = [4, 1]

        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)
        
        # tasks
        self.prey_pos[0] = [0, 9]
        self.prey_pos[1] = [9, 9]
        self.prey_pos[2] = [3, 4]

        for prey_i in range(self.n_preys):
            self.__update_prey_view(prey_i)


    def __init_full_obs(self):
        if self.scenario == 0:
            self._full_obs = self.__create_grid()
            self.__init_rnd_starting()
        elif self.scenario == 1:
            _grid = self.__create_medium_grid()
            self._full_obs = _grid
            self._base_grid = _grid
            self.__init_rnd_starting()
        elif self.scenario == 2:
            _grid = self.__create_medium_grid()
            self._full_obs = _grid
            self._base_grid = _grid
            self.__init_fixed_medium()

        self.__draw_base_img()

    def get_agent_obs(self):
        # print("Entering get_agent_obs")
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)] # coordinates

            # check if prey is in the view area
            _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            # print("prey_pos", _prey_pos)
            row_offset = (int) (self._agent_view_mask[0] // 2)
            col_offset = (int) (self._agent_view_mask[1] // 2)
            for row in range(max(0, pos[0] - row_offset), min(pos[0] + row_offset + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - col_offset), min(pos[1] + col_offset + 1, self._grid_shape[1])):
                    if PRE_IDS['wall'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - row_offset), col - (pos[1] - col_offset)] = 1  # get relative position for the prey loc.
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - row_offset), col - (pos[1] - col_offset)] = 2  # get relative position for the prey loc.
                    if PRE_IDS['crewmate'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - row_offset), col - (pos[1] - col_offset)] = 3  # get relative position for the crewmate loc.
                    if PRE_IDS['imposter'] in self._full_obs[row][col]:
                        if agent_i < self.n_imposter:
                            _prey_pos[row - (pos[0] - row_offset), col - (pos[1] - col_offset)] = 4  # get relative position for the imposter loc.
                        else:
                            _prey_pos[row - (pos[0] - row_offset), col - (pos[1] - col_offset)] = 3  # get relative position for the imposter loc, masked as crewmate



            # print("prey_pos after", _prey_pos)
            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time

            # add cooldown observations if enabled
            if self.enable_kill_cooldown and agent_i < self.n_imposter:
                _agent_i_obs.append(self._imp_cooldowns[agent_i])
            
            
            _obs.append(_agent_i_obs)
            # print("agent_i_obs", _agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        # print("obs", _obs)
        # print("Exiting get_agent_obs")
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]
        self._agent_alive = [True for _ in range(self.n_agents)]

        self._num_crewmate_dead = 0
        self._num_tasks_finished = 0
        if self.enable_kill_cooldown:
            self._imp_cooldowns = np.array([self.kill_cooldown] * self.n_imposter)

        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row][col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        # move = np.random.choice(len(move), p=move)
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        elif move == 5:  # kill
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __remove_prey_pos(self, prey_i):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __remove_crewmate_pos(self, agent_i):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
    
    # def __update_agent_view(self, agent_i):
    #     self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_agent_view(self, agent_i):
        if agent_i < self.n_imposter:
            self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['imposter'] + str(agent_i + 1)
        else:
            self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['crewmate'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['crewmate'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['crewmate'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['crewmate'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['crewmate'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['crewmate'])[1]) - 1)
        return _count, agent_id

    def _neighbour_crewmate(self, pos):
        # check if crewmate is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['crewmate'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['crewmate'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['crewmate'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['crewmate'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['crewmate'])[1]) - 1)
        return _count, agent_id

    def _neighbour_imposter(self, pos):
        # check if imposter is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['imposter'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['imposter'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['imposter'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['imposter'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['imposter'])[1]) - 1)
        return _count, agent_id

    def get_moves(self, agents_action):
        moves = []
        for agent_i, action_probs in enumerate(agents_action):
            move = np.random.choice(len(action_probs), p=action_probs)
            moves.append(move)
        return moves

    def process_imposter_kills(self, agents_action, rewards):
        for agent_i, action in enumerate(agents_action):
            if agent_i < self.n_imposter and action == 5: # imposter kill action
                if not self._agent_dones[agent_i] and self._agent_alive[agent_i]:
                    crewmate_neighbour_count, n_i = self._neighbour_crewmate(self.agent_pos[agent_i])
                    if crewmate_neighbour_count >= 1:
                        crewmate_to_kill = n_i[0] # picking first crewmate in neighbors to kill
                        if self._agent_alive[crewmate_to_kill]: # Not sure if this check is necessary
                            
                            if self.enable_kill_cooldown: # if cooldowns enabled, only process kill if cooldown is done
                                if self._imp_cooldowns[agent_i] > 0:
                                    continue
                                else:
                                    self._imp_cooldowns[agent_i] = self.kill_cooldown
                            
                            self._agent_alive[crewmate_to_kill] = False
                            self._agent_dones[crewmate_to_kill] = True

                            self._num_crewmate_dead += 1
                            # remove killed crewmate from grid
                            self.__remove_crewmate_pos(crewmate_to_kill)

                            if not self.persistent_kill_reward: # kill rewards only added for this kill
                                for imposter_i in range(self.n_imposter):
                                    rewards[imposter_i] += self._crewmate_capture_reward
        
        if self.persistent_kill_reward: # kill rewards added for every step
            if self._num_crewmate_dead > 0:
                for imposter_i in range(self.n_imposter):
                    rewards[imposter_i] += self._crewmate_capture_reward * self._num_crewmate_dead
    
    def process_crewmate_tasks(self, rewards):
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_crewmate(self.prey_pos[prey_i])

                if predator_neighbour_count >= 1:
                    # _reward = self._penalty if predator_neighbour_count == 1 else self._prey_capture_reward
                    # self._prey_alive[prey_i] = (predator_neighbour_count == 1)

                    self._prey_alive[prey_i] = False
                    self._num_tasks_finished += 1

                    if not self.persistent_task_reward: # task rewards only added for this step
                        for agent_i in range(self.n_imposter, self.n_agents):
                            rewards[agent_i] += self._prey_capture_reward

                    # remove killed prey from grid
                    self.__remove_prey_pos(prey_i)
        
        if self.persistent_task_reward: # task rewards added for every step
            if self._num_tasks_finished > 0:
                for agent_i in range(self.n_imposter, self.n_agents):
                    rewards[agent_i] += self._prey_capture_reward * self._num_tasks_finished

    def process_kill_cooldown(self):
        for agent_i in range(self.n_imposter):
            self._imp_cooldowns[agent_i] = max(0, self._imp_cooldowns[agent_i] - 1)

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        moves = self.get_moves(agents_action)
        # What's the confusion?
        # What if agents attack each other at the same time? Should both of them be effected?
        # Ans: I guess, yes
        # What if other agent moves before the attack is performed in the same time-step.
        # Ans: May be, I can process all the attack actions before move directions to ensure attacks have their effect.

        # process imposter kill action:
        self.process_imposter_kills(moves, rewards)

        # process kill cooldowns if enabled
        if self.enable_kill_cooldown:
            self.process_kill_cooldown()

        # process moves
        for agent_i, action in enumerate(moves):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
            # reward staying alive (currently for both imposters and crewmates)
            if self._agent_alive[agent_i]:
                rewards[agent_i] += self._alive_reward

        # process crewmate task completion
        self.process_crewmate_tasks(rewards)

        if (self._step_count >= self._max_steps):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        # All tasks finished
        if (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        # All crewmates died
        if (True not in self._agent_alive[self.n_imposter:]):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive, 'moves': moves, 'moves_obs': self.get_move_obs(moves)}

    def get_move_obs(self, moves):
        total_obs = []
        for agent_i in range(self.n_imposter, self.n_agents): # all crewmates
            obs_i = []
            for agent_j in range(self.n_agents): # everyone else
                if agent_i != agent_j:
                    if self.is_visible(agent_i, agent_j):
                        obs_i.append(moves[agent_j])
                    else:
                        obs_i.append(-1)
            total_obs.append(obs_i)
        return total_obs
    '''    
    num_crewmates * (num_total - 1)
        [
            agent 2: [agent1ac, agent3 ac, agent4ac, agent5ac]
            agent 3: [agent1ac, agent2 ac, agent4ac, agent5ac]
            agent 4: [agent1ac, agent2 ac, agent3ac, agent5ac]
        ]
    '''
    
    def is_visible(self, agent_i, agent_j):
        source_pos = self.agent_pos[agent_i]
        target_pos = self.agent_pos[agent_j]
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)                

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        # for agent_i in range(self.n_agents):
        #     for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
        #         fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
        #     fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            if self._agent_alive[agent_i]:
                if agent_i < self.n_imposter:
                    draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=IMPOSTER_COLOR)
                else:
                    draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)

                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
IMPOSTER_COLOR = ImageColor.getcolor('red', mode='RGB')

AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'green'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
    5: "KILL",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0',
    'imposter': 'I',
    'crewmate': 'C',
}


MEDIUM_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
]


REAL_GRID = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
]
