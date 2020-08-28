import os
import re
import ast
import cv2
import gym
import gym_ple
import torch
import keyboard
import importlib
import collections
import numpy as np
from enum import Enum
from PIL import Image, ImageDraw, ImageOps, ImageFont
from utils.gym_wrappers import wrap_atari, wrap_deepmind, wrap_pytorch

import games.pong

class EnvId(Enum):
	NONE = 0
	PONG_ATARI = 1
	PONG_CUSTOM = 2
	FLAPPY_BIRD = 3

class EnvSetting:
	def __init__(self, player_count=1, crop_rect=None, resize_rect=None, recolor_arr=None, frame_stack_cnt=0):
		self.player_count = player_count
		self.crop_rect = crop_rect
		self.resize_rect = resize_rect
		self.recolor_arr = recolor_arr
		self.frame_stack_cnt = frame_stack_cnt
		self.ext_args = {}

class EnvWrapper:
	# A wrapper class to support multiple environments with different interface or data
	
	env_map = {
		"PongNoFrameskip": EnvId.PONG_ATARI,
		"PongCustom": EnvId.PONG_CUSTOM,
		"FlappyBird": EnvId.FLAPPY_BIRD
	}
	
	env_settings = {
		EnvId.NONE: EnvSetting (
					player_count = 1,
					frame_stack_cnt = 0
				),
		EnvId.PONG_ATARI: EnvSetting (
					player_count = 1,
					crop_rect = (0, 34, 160, 194),	# 160x160
					resize_rect = (80, 80),			# * 0.5
					recolor_arr = [((0, 87), 0), ((88, 255), 1)],
					frame_stack_cnt = 4
				),
		EnvId.PONG_CUSTOM: EnvSetting (
					player_count = 2,
					crop_rect = (5, 0, 205, 200),	# 200x200
					resize_rect = (80, 80),			# * 0.4
					#recolor_arr = None,								# dont modify pixels
					#recolor_arr = [((0, 80), 0), ((81, 255), 1)], 		# remove red (right) side and all other moving elements white
					#recolor_arr = [((148, 150), 0), ((1, 255), 1)],	# remove left (green) side and all other moving elements white
					recolor_arr = [((0, 50), 0), ((51, 255), 1)],		# make all moving elements white
					frame_stack_cnt = 4
					# mode == 0: Play against SimpleAI
					# mode == 1: Play against yourself, both learning, only main player saving
					# mode == 2: Play against another agent, requires setting {agent='name'} and {model='..\models\Game\Agent\MyModel\model_100.mdl'}
					# mode == 3: Play against Human Player
				),
		EnvId.FLAPPY_BIRD: EnvSetting (
					player_count = 1,
					crop_rect = (58, 8, 288, 408),	# 230x400 (Full: 288x512)
					resize_rect = (115, 200),		# * 0.5
					recolor_arr = [((0, 70), 1), ((71, 255), 0)],		# make all moving elements white
					frame_stack_cnt = 4
				)
	}
	
	# ############################################################################################ #
	
	def __init__(self, args):
		self.args = args
		
		# Handle hotkeys
		def toggle_render():
			args.render = not args.render
		
		if os.name == 'nt':
			keyboard.add_hotkey('F12', toggle_render)
		
		# Extract base environment name (without "-vX" at the end)
		dash_idx = args.environment.rfind("-")
		self.env_base = args.environment if dash_idx == -1 else args.environment[0:dash_idx]
		self.env_id = self.env_map[self.env_base] if self.env_base in self.env_map else EnvId.NONE
		self.cfg = self.env_settings[self.env_id]
		self.frame_buffers = []
		self.episodeBase = 1
		self.video = None
		self.font = None
		
		# Extract the episode base from modelname (model_XXXXX.mdl)
		if self.args.file and os.path.isfile(self.args.file):
			self.episodeBase = int(re.search('model_(.+?).mdl', os.path.basename(self.args.file)).group(1))
		
		if args.argsext:
			self.cfg.ext_args = ast.literal_eval(args.argsext)
		
		for i in range(self.cfg.player_count):
			self.frame_buffers.append(collections.deque(maxlen=self.cfg.frame_stack_cnt))	# Save last x-1 frames & 1 current for each player
		
		# Setup environment
		self.env = self.make_env(args)
	
	def make_env(self, args):
		if self.env_id == EnvId.PONG_ATARI:
			env = gym.make(args.environment)
			env = wrap_atari(env)
			env = wrap_deepmind(env, warp_frame=False)
			#env = wrap_pytorch(env)
		elif self.env_id == EnvId.PONG_CUSTOM:
			env = games.pong.Pong(not args.render)
			if "mode" in self.cfg.ext_args and self.cfg.ext_args["mode"] != 0:
				if self.cfg.ext_args["mode"] == 1:
					agent_module = importlib.import_module("agents." + args.agent)
					self.pong_ai = agent_module.Agent(self.get_input_size(), self.get_output_size(), args.train, self.is_conv(), args.file, args.seed)
					self.pong_ai.init_network()
					env.set_names(args.agent, args.agent)
				elif self.cfg.ext_args["mode"] == 2:
					agent_module = importlib.import_module("agents." + self.cfg.ext_args["agent"])
					self.pong_ai = agent_module.Agent(self.get_input_size(), self.get_output_size(), args.train, self.is_conv(), self.cfg.ext_args["model"], args.seed)
					self.pong_ai.init_network()
					env.set_names(self.cfg.ext_args["agent"] + "_mode2", args.agent)
				elif self.cfg.ext_args["mode"] == 3:
					self.pong_ai = games.pong.PongPlayer(env)
					env.set_names(self.pong_ai.get_name(), args.agent)
			else:
				self.pong_ai = games.pong.PongAi(env)
				env.set_names(self.pong_ai.get_name(), args.agent)
		else:
			env = gym.make(args.environment)
		
		env.seed(args.seed)
		
		return env
	
	def close(self):
		self.env.close()
		if self.video:
			self.video.release()
	
	def render(self, episode=None, action=None, reward=None, avg_reward=None):
		if not self.args.record:
			img = self.env.render()
			return
		
		img = self.env.render(mode='rgb_array')
		img = Image.fromarray(img)									# Create PIL image
		if img.width < 400 or img.height < 400:
			img = img.resize((img.width*2, img.height*2))			# Upsampling -> Improve image sharpness
		img = ImageOps.expand(img, (1, 1, 1, 1), fill="#666666")	# Add gray border
		img = ImageOps.expand(img, (20, 100, 20, 20))				# Add top black comment box
		
		# Draw custom data ontop of image
		if avg_reward != None:
			if self.font == None:
				self.font = ImageFont.truetype("../fonts/Roboto-Regular.ttf", size=24, encoding="unic")
			draw = ImageDraw.Draw(img)
			draw.text((20, 10), "Epoche: {}".format(self.episodeBase + episode), font=self.font, fill="#ffffff")
			draw.text((210, 10), "Reward:  {:.2f}".format(reward), font=self.font, fill="#ffffff")
			draw.text((210, 50), "Average: {:.2f}".format(avg_reward), font=self.font, fill="#ffffff")
			
			if type(action) == int:
				draw.text((20, 50), "Input: ", font=self.font, fill="#ffffff")
				for i in range(self.get_output_size()):
					x = 90 + i*30
					y = 55
					col = 'red' if action==i else 'white'
					draw.rectangle((x, y, x + 20, y + 20), fill=col, outline=col)
			
		
		# Convert to needed format (nparray, BGR)
		frame = np.asarray(img)
		frame = frame[:,:,[2, 1, 0]]					# flip colors from (R,G,B) to (B,G,R)
		
		# Create VideoWriter and PIL font if this is the first frame
		if self.video == None:
			fps = 60
			if hasattr(self.env, 'metadata') and hasattr(self.env.metadata, 'video.frames_per_second'):
				fps = 2*self.env.metadata.video.frames_per_second
			self.video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.width, img.height))
		
		self.video.write(frame)
	
	def step(self, action):
		if self.env_id == EnvId.PONG_CUSTOM:
			# Normalize rewards to [-1, 0, +1] then weight them and add 1 to keep the agent alive as second priority
			def calc_reward(reward):
				#return int(np.sign(reward))*1000 + 1
				return reward
			
			# Handle the AI
			ai_action = self.pong_ai.get_action(self.ai_state)
			
			next_states, rewards, done, info = self.env.step((ai_action, action))
			
			# Handle the AI
			next_ai_state = self.preprocess(next_states[0], 1)
			self.ai_state = next_ai_state
			
			# Handle the main player
			next_state = next_states[1]
			reward = calc_reward(rewards[1])
		else:
			next_state, reward, done, info = self.env.step(action)
		
		return (self.preprocess(next_state), reward, done, info)
	
	def reset(self):
		for fb in self.frame_buffers:
			fb.clear()
		
		# Pre-fill frame buffer with empty frames
		if self.env_id != EnvId.NONE:
			for fb in self.frame_buffers:
				for i in range(fb.maxlen):
					fb.append(np.zeros((self.cfg.resize_rect[1], self.cfg.resize_rect[0])))
		
		if self.env_id == EnvId.PONG_CUSTOM:
			ai_state, state = self.env.reset()
			self.ai_state = self.preprocess(ai_state, 1)
		else:
			state = self.env.reset()
		
		return self.preprocess(state)
	
	def get_input_size(self):
		if self.env_id != EnvId.NONE:
			return (self.frame_buffers[0].maxlen, self.cfg.resize_rect[0], self.cfg.resize_rect[1])
		
		return self.env.observation_space.shape[0]
	
	def get_output_size(self):
		if self.env_id == EnvId.PONG_CUSTOM:
			return 3
		
		return self.env.action_space.n
	
	def preprocess(self, state, playerid=0):
		# Environment specific state pre-processing (e.g. cropping image, remove colors from image, resize image, ...)
		
		if self.env_id != EnvId.NONE:
			# Grayscale -> Crop -> Resize -> Recolor:
			
			def recolor_pixels(val):
				for rule in self.cfg.recolor_arr:
					limits = rule[0]
					new_col = rule[1]
					if val >= limits[0] and val <= limits[1]:
						return new_col
				return val
			
			img = Image.fromarray(state).convert("L")						# convert to greyscale
			if self.cfg.crop_rect: img = img.crop(self.cfg.crop_rect)		# cut out useful image data
			if self.cfg.resize_rect: img = img.resize(self.cfg.resize_rect)	# resize image to fit conv layer and save neurons
			if self.cfg.recolor_arr: img = img.point(recolor_pixels, "L")	# recolor all pixels
			
			curr_frame = np.asarray(img)
			self.frame_buffers[playerid].append(curr_frame)
			
			if os.name == 'nt':
				if playerid == 0 and keyboard.is_pressed("F11") or \
				   playerid == 1 and keyboard.is_pressed("F10"):
					Image.fromarray(np.hstack(self.frame_buffers[playerid])).show()
			
			state = np.stack(self.frame_buffers[playerid])	# do frame stacking into the image channels (1 new channel per frame)
		
		return state
	
	def copy_network(self, src_agent):
		if self.env_id == EnvId.PONG_CUSTOM:
			if "mode" in self.cfg.ext_args and self.cfg.ext_args["mode"] == 1 and self.args.train:
				self.pong_ai.copy_from(src_agent)
	
	def is_conv(self):
		return self.env_id != EnvId.NONE
