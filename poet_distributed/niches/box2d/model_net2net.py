# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


import numpy as np
import copy
import random
import json
from .env import make_env
import time
import logging
logger = logging.getLogger(__name__)

final_mode = False
render_mode = False
RENDER_DELAY = False
record_video = False
MEAN_MODE = False


def make_model(game):
    # can be extended in the future.
    model = ModelNet2Net(game)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def passthru(x):
    return x

# useful for discrete actions


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# useful for discrete actions


def sample(p):
    return np.argmax(np.random.multinomial(1, p))


class ModelNet2Net:
    ''' simple feedforward model '''

    def __init__(self, game):
        self.output_noise = game.output_noise
        self.env_name = game.env_name

        self.layers = game.layers

        self.rnn_mode = False  # in the future will be useful
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = game.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        if game.time_factor > 0:
            self.time_factor = float(game.time_factor)
            self.time_input = 1
        self.input_size = game.input_size
        self.output_size = game.output_size
        if len(self.layers) == 0:
            self.shapes = [(self.input_size + self.time_input, self.output_size)]
        else:
            self.shapes = [(self.input_size + self.time_input, self.layers[0])]
            for i in range(1, len(self.layers)):
                self.shapes.append((self.layers[i - 1], self.layers[i]))
            self.shapes.append((self.layers[-1], self.output_size))

        logger.debug(f"Created model with shapes: {self.shapes}")

        self.sample_output = False
        if game.activation == 'relu':
            self.activations = [relu] * len(self.layers) + [passthru]
        else:
            raise NotImplementedError

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
#             self.param_count += (np.product(shape) + shape[1])
            # if self.output_noise[idx]:
            #     self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)
            idx += 1

        # set param count
        self.set_param_count_from_shape()

        self.render_mode = False

    def __repr__(self):
        return "{}".format(self.__dict__)

    def make_env(self, seed, render_mode=False, env_config=None):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, seed=seed,
                            render_mode=render_mode, env_config=env_config)

    def get_action(self, x, t=0, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            if (self.output_noise[i] and (not mean_mode)):
                out_size = self.shapes[i][1]
                out_std = self.bias_std[i]
                output_noise = np.random.randn(out_size) * out_std
                h += output_noise
            h = self.activations[i](h)

        if self.sample_output:
            h = sample(h)

        return h

    def model_params_to_mod_arch_and_shape(self, model_params):
        """
        model_params: [num_layers, unit_1, unit_2, ..., unit_{num_layers - 1}, theta]

        e.g.
        [1, theta]
        [2, 30, theta]
        [3, 30, 40, theta]

        """
        shape_prev = copy.deepcopy(self.shapes)

        num_layers = int(model_params[0])
        if num_layers == 1:
            self.shapes = [(self.input_size + self.time_input, self.output_size)]
        else:
            num_units_array = model_params[1: num_layers].astype(np.int32)
            self.shapes = [(self.input_size + self.time_input, num_units_array[0])]

            if len(num_units_array) > 1:
                for i in range(len(num_units_array) - 1):
                    self.shapes.append((num_units_array[i], num_units_array[i + 1]))
            self.shapes.append((num_units_array[-1], self.output_size))

        # if shape_prev != self.shapes:
        #     logger.info(f"Changing model's shape from {shape_prev} --> {self.shapes}")

        self.set_param_count_from_shape()
        self.output_noise = [False] * len(self.shapes)
        self.activations = [relu] * (len(self.shapes) - 1) + [passthru]

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            # if self.output_noise[idx]:
            #     self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)
            idx += 1

    def set_param_count_from_shape(self, ):
        self.param_count = 0
        for shape in self.shapes:
            self.param_count += (np.product(shape) + shape[1])

        self.param_count += len(self.shapes)

    def set_model_params(self, model_params):
        self.model_params_to_mod_arch_and_shape(model_params)

#         pointer = 0
        pointer = int(model_params[0])
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.bias_log_std[i] = np.array(
                    model_params[pointer:pointer + s])
                self.bias_std[i] = np.exp(
                    self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
                if self.render_mode:
                    print("bias_std, layer", i, self.bias_std[i])
                pointer += s

    def get_model_params(self, ):
        theta_new = self.get_param_arch_from_shape()

        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s_b = b_shape

            w_flattened = self.weight[i].reshape(s_w)
            b_flattened = self.bias[i].reshape(s_b)

            theta_new = np.concatenate((theta_new, w_flattened))
            theta_new = np.concatenate((theta_new, b_flattened))

        return theta_new

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_param_arch_from_shape(self, ):
        return np.array([len(self.shapes)] + [self.shapes[i][1] for i in range(len(self.shapes) - 1)])

    def get_random_model_params(self, stdev=0.1):
        param_arch = self.get_param_arch_from_shape()
        return np.concatenate((param_arch, np.random.randn(self.param_count - len(param_arch)) * stdev))

    def get_zero_model_params(self, ):
        param_arch = self.get_param_arch_from_shape()
        return np.concatenate((param_arch, np.zeros(self.param_count - len(param_arch))))

    def inject_noise(self, param, sigma=0.0005, flag_inject_noise=False):
        if flag_inject_noise:
            return param + sigma * np.random.randn(*param.shape)
        else:
            return param

    def net2widernet(self, widen_specs, flag_inject_noise=True, quiet_mode=True):
        dummy_input = np.random.randn(self.input_size)
        out_before_transformation = self.get_action(dummy_input)

        assert len(widen_specs) > 0 and len(widen_specs) == (len(self.weight) - 1)
        for i in range(len(widen_specs)):
            if widen_specs[i] == 0:
                continue

            num_units_to_add = widen_specs[i]
            num_units_orig = self.shapes[i][1]

            # select units to mimic
            ind_units_selected = np.random.randint(low=0, high=num_units_orig, size=num_units_to_add)
            if not quiet_mode:
                print(f"Widening layer {i} with selected units: {ind_units_selected}")

            # make copies of prev layers
            weights_layer_pre_act = self.weight[i].copy()
            bias_layer_pre_act = self.bias[i].copy()

            weights_layer_post_act = self.weight[i + 1].copy()

            # modifying layers
            weights_layer_pre_act_mod = np.concatenate(
                (weights_layer_pre_act,
                 self.inject_noise(weights_layer_pre_act[:, ind_units_selected], flag_inject_noise=flag_inject_noise)
                ),
                axis=1
            )
            bias_layer_pre_act_mod = np.concatenate(
                (bias_layer_pre_act,
                 self.inject_noise(bias_layer_pre_act[ind_units_selected], flag_inject_noise=flag_inject_noise)
                ),
                axis=0
            )

            weights_layer_post_act_mod = np.concatenate(
                (weights_layer_post_act,
                 self.inject_noise(weights_layer_post_act[ind_units_selected, :], flag_inject_noise=flag_inject_noise)
                ),
                axis=0
            )

            factor_prev_units = np.ones(num_units_orig, dtype=np.int32)
            factor_new_units = np.ones(num_units_to_add, dtype=np.int32)
            for e in (ind_units_selected):
                factor_prev_units[e] += 1
                factor_new_units[(np.array(ind_units_selected) == e).nonzero()[0]] += 1
            factor_all = np.concatenate((factor_prev_units, factor_new_units))

            weights_layer_post_act_mod = weights_layer_post_act_mod / factor_all[:, np.newaxis]

            # assignment of new layers
            self.weight[i] = weights_layer_pre_act_mod
            self.bias[i] = bias_layer_pre_act_mod
            self.weight[i + 1] = weights_layer_post_act_mod

        # mod other affected variables
        self.shapes = [e.shape for e in self.weight]
        self.layers = len(self.weight)

        self.set_param_count_from_shape()

        # check function preservation
        if not flag_inject_noise:
            out_after_transformation = self.get_action(dummy_input)
            flag_equal = np.allclose(out_after_transformation, out_before_transformation)
            if not flag_equal:
                print(f"Outputs after transformation are marked as not equal: "
                      f"{out_before_transformation} != {out_after_transformation}"
                     )

    def net2deepernet(self, deepen_specs, flag_inject_noise=True, quiet_mode=True):
        dummy_input = np.random.randn(self.input_size)
        out_before_transformation = self.get_action(dummy_input)

        assert len(deepen_specs) > 0 and len(deepen_specs) == (len(self.weight) - 1)

        weights_deepened = [self.weight[0]]
        biases_deepened = [self.bias[0]]
        for i in range(len(deepen_specs)):
            assert deepen_specs[i] == 0 or deepen_specs[i] == 1
            if deepen_specs[i] == 0:
                weights_deepened.append(self.weight[i + 1])
                biases_deepened.append(self.bias[i + 1])
                continue

            num_units_orig = self.shapes[i][1]

            # generate layer to inject
            weight_layer_to_inject = self.inject_noise(np.eye(num_units_orig), flag_inject_noise=flag_inject_noise)
            bias_layer_to_inject = self.inject_noise(np.zeros(num_units_orig), flag_inject_noise=flag_inject_noise)

            # inject the layer
            weights_deepened.append(weight_layer_to_inject)
            biases_deepened.append(bias_layer_to_inject)

            # add the next layer
            weights_deepened.append(self.weight[i + 1])
            biases_deepened.append(self.bias[i + 1])

        # replace the new deepened weights and biases
        self.weight = weights_deepened
        self.bias = biases_deepened

        # mod other affected variables
        self.shapes = [e.shape for e in self.weight]
        self.layers = len(self.weight)

        self.set_param_count_from_shape()

        self.output_noise = [False] * len(self.shapes)
        self.activations = [relu] * (len(self.shapes) - 1) + [passthru]
#         self.output_noise = [False] * len(self.weight)
#         self.activations = [relu] * (len(self.weight) - 1) + [passthru]

        # check function preservation
        if not flag_inject_noise:
            out_after_transformation = self.get_action(dummy_input)
            flag_equal = np.allclose(out_after_transformation, out_before_transformation)
            if not flag_equal:
                print(f"Outputs after transformation are marked as not equal: "
                      f"{out_before_transformation} != {out_after_transformation}"
                     )


def simulate(model, seed, train_mode=False, render_mode=False, num_episode=5,
             max_len=-1, env_config_this_sim=None):
    reward_list = []
    t_list = []

    max_episode_length = 2000

    if train_mode and max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        logger.debug('Setting seed to {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    if env_config_this_sim:
        model.env.set_env_config(env_config_this_sim)

    for _ in range(num_episode):

        if model.rnn_mode:
            model.reset()

        obs = model.env.reset()
        if obs is None:
            obs = np.zeros(model.input_size)

        total_reward = 0.0
        for t in range(max_episode_length):

            if render_mode:
                model.env.render("human")
                if RENDER_DELAY:
                    time.sleep(0.01)

            if model.rnn_mode:
                model.update(obs, t)
                action = model.get_action()
            else:
                if MEAN_MODE:
                    action = model.get_action(
                        obs, t=t, mean_mode=(not train_mode))
                else:
                    action = model.get_action(obs, t=t, mean_mode=False)

            obs, reward, done, info = model.env.step(action)
            total_reward += reward

            if done:
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)
        reward_list.append(total_reward)
        t_list.append(t)

    return reward_list, t_list
