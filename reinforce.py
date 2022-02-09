import torch
import torch.optim as optim

import os
import random
import tempfile
import numpy as np
from models import Actor, Sensor, Aggregator, LookAhead
from classifier import Classifier


class Reinforce:
    def __init__(self, env, gamma=1.00, max_episodes=10000, max_step=4, seed=0, hyperparameter_lambda=1, lr=0.0005):
        self.env = env
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_step = max_step
        self.seed = seed

        self.episode_timestep = []
        self.episode_seconds = []
        self.evaluation_scores = []

        self.logpas = []
        self.rewards = []

        self.checkpoint_dir = tempfile.mkdtemp()

        self.hyperparameter_lambda = hyperparameter_lambda

        self.actor_optimizer_lr = lr
        self.actor_model = Actor(256, 2)  # this is (256, 35) in the paper but have to check
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.actor_optimizer_lr)

        self.sensor_optimizer_lr = lr
        self.sensor_model = Sensor
        self.sensor_optimizer = optim.Adam(self.sensor_model.parameters(), lr=self.sensor_optimizer_lr)

        self.aggregator_optimizer_lr = lr
        self.aggregator_model = Aggregator
        self.aggregator_optimizer = optim.Adam(self.aggregator_model.parameters(), lr=self.aggregator_optimizer_lr)

        self.lookahead_optimizer_lr = lr
        self.lookahead_model = LookAhead
        self.lookahead_optimizer = optim.Adam(self.lookahead_model.parameters(), lr=self.lookahead_optimizer_lr)

        self.classifier_optimizer_lr = lr
        self.classifier_model = Classifier
        self.classifier_optimizer = optim.Adam(self.classifier_model.parameters(), lr=self.classifier_optimizer_lr)

    def optimize_model(self, look_ahead_loss, classification_loss):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * self.rewards[t:]) for t in range(T)])

        discounts = torch.FloatTensor(discounts).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        self.logpas = torch.cat(self.logpas)

        policy_loss = -(discounts * returns * self.logpas).mean()

        a_loss = policy_loss
        s_loss = policy_loss + classification_loss + self.hyperparameter_lambda * look_ahead_loss
        r_loss = policy_loss + classification_loss + self.hyperparameter_lambda * look_ahead_loss
        l_loss = look_ahead_loss
        c_loss = policy_loss + classification_loss

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        self.sensor_optimizer.zero_grad()
        s_loss.backward()
        self.sensor_optimizer.step()

        self.aggregator_optimizer.zero_grad()
        r_loss.backward()
        self.aggregator_optimizer.step()

        self.lookahead_optimizer.zero_grad()
        l_loss.backward()
        self.lookahead_optimizer.step()

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        self.classifier_optimizer.step()

    def interaction_step(self, a, p, env):
        m, is_exploratory, logpa, _ = self.actor_model.full_pass(a, p)  # m: motion
        new_view = env.step(m)
        self.logpas.append(logpa)

        return m, new_view

    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        for episode in range(1, self.max_episodes + 1):

            p, label = self.env.reset(), False  # p: random camera pose initialization
            a = torch.zeros(256, dtype=torch.float32)  # a: aggregator feature vector initialization
            a_lookahead = torch.zeros(256, dtype=torch.float32)

            self.rewards = []
            look_ahead_loss = 0

            for step in range(self.max_step):
                m, new_view = self.interaction_step(a, p, self.env)
                p += m
                s = self.sensor_model.forward(m, new_view)
                a = self.aggregator_model.forward(a + s)

                if step != 0:
                    look_ahead_loss += 1 - torch.nn.CosineSimilarity(a, a_lookahead)

                a_lookahead = self.lookahead_model(a, m, p)

                # we need to talk about losses
                if step == self.max_step - 1:
                    c = self.classifier_model(a)
                    classification_loss = torch.nn.functional.mse_loss(c, label)
                    if torch.argmax(c) == torch.argmax(label):
                        self.rewards.append(0)
                    else:
                        self.rewards.append(-1)
                else:
                    self.rewards.append(-1)

            self.optimize_model(look_ahead_loss, classification_loss)

            self.save_checkpoint(episode - 1, self.actor_model, self.sensor_model,
                                 self.aggregator_model, self.lookahead_model)

        print('Training complete.')
        self.env.close()
        del self.env

        return

    def save_checkpoint(self, episode_idx, actor_model, sensor_model,
                        aggregator_model, lookahead_model, classifier_model):
        torch.save(actor_model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'actor.{}.tar'.format(episode_idx)))
        torch.save(sensor_model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'sensor.{}.tar'.format(episode_idx)))
        torch.save(aggregator_model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'aggregator.{}.tar'.format(episode_idx)))
        torch.save(lookahead_model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'lookahead.{}.tar'.format(episode_idx)))
        torch.save(classifier_model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'classfier.{}.tar'.format(episode_idx)))
