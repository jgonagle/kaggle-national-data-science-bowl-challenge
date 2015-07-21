from __future__ import print_function

import time
import collections

import numpy as np
import numpy.random as R

import theano
import theano.tensor as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid as ag

class GenDBMBinaryLayer(object):
    
    def __init__(self, num_units, down_weights, up_weights, biases, numpy_rng=None, theano_rng=None):
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_units = num_units

        self.down_weights = down_weights
        self.up_weights = up_weights
        self.biases = biases
            
    #calculate the layer activation and sample probability given the states of the adjacent layers
    def sigmoid(self, down_states, up_states, inv_temps=None):
        
        layer_activation = self.biases
        
        #add the input from the previous layer if it exists
        if (self.down_weights is not None) and (down_states is not None):
            layer_activation += T.dot(down_states, self.down_weights)
        
        #add the input from the next layer if it exists
        if (self.up_weights is not None) and (up_states is not None):
            layer_activation += T.dot(up_states, T.transpose(self.up_weights))
        
        #multiply by the inverse temperatures if specified (for the Metropolis-Hastings acceptance rule in Wang-Landau adaptive simulated annealing)
        if inv_temps is not None:
            layer_activation *= inv_temps.dimshuffle(0, 'x')
        
        layer_sigmoid = T.nnet.sigmoid(layer_activation)
                
        return layer_sigmoid
    
    #calculate a layer sample given the states of the adjacent layers
    def sample(self, down_states, up_states, inv_temps=None):
        
        layer_sigmoid = self.sigmoid(down_states, up_states, inv_temps)
        layer_sample = self.theano_rng.binomial(n=1, p=layer_sigmoid, size=layer_sigmoid.shape)
                
        return layer_sample

class GenDBMBinary(object):
    
    def __init__(self, layer_sizes, layer_weights_mean_init, layer_weights_std_init, layer_biases_init, gen_vis_input=None, numpy_rng=None, theano_rng=None):
        
        if gen_vis_input is None:
            self.gen_vis_input = T.dmatrix('Generative visible input')
        else:
            self.gen_vis_input = gen_vis_input
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_layers = len(layer_sizes)
        self.num_inputs = layer_sizes[0]
        self.total_num_states = sum(layer_sizes)
        
        #print('\nBeginning initialization of deep boltzmann machine with {} layers.'.format(self.num_layers))
        
        #instantiate all weights        
        self.weights = [None for i in range(self.num_layers - 1)]
        
        for weight_num in range(self.num_layers - 1):
            
            weights_value = self.numpy_rng.normal(loc=layer_weights_mean_init[weight_num], scale=layer_weights_std_init[weight_num], size=(layer_sizes[weight_num], layer_sizes[weight_num + 1]))
            self.weights[weight_num] = theano.shared(value=weights_value, name='Weight between layers {} and {}'.format(weight_num, (weight_num + 1)), borrow=True)
        
        #instantiate all biases
        self.biases = [None for i in range(self.num_layers)]
        
        for bias_num in range(self.num_layers):
            
            layer_biases_value = np.full(fill_value=layer_biases_init[bias_num], shape=(layer_sizes[bias_num],))
            self.biases[bias_num] = theano.shared(value=layer_biases_value, name='Bias for layer {}'.format(bias_num), borrow=True)
        
        #instantiate all layers
        self.layers = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            if self.is_vis_layer(layer_num):
                down_weights, up_weights = None, self.weights[0]
            elif self.is_last_layer(layer_num):
                down_weights, up_weights = self.weights[layer_num - 1], None
            else:
                down_weights, up_weights = self.weights[layer_num - 1], self.weights[layer_num]
           
            self.layers[layer_num] = GenDBMBinaryLayer(num_units=layer_sizes[layer_num], down_weights=down_weights, up_weights=up_weights, biases=self.biases[layer_num], numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
        
        #set parameters and the last change in parameters (for momentum learning)
        self.params = self.weights + self.biases
        self.param_changes_last = [theano.shared(value=np.zeros_like(param.get_value(borrow='False')), name='{} change last'.format(param.name), borrow=True) for param in self.params]
        
        #print('Finished initialization of deep boltzmann machine with {} layers!'.format(self.num_layers))
    
    def is_vis_layer(self, layer_num):
        
        return (layer_num == 0)
        
    def is_last_layer(self, layer_num):
        
        return (layer_num == (self.num_layers - 1))
        
    #calculate the energy function for the current state of the deep boltzmann machine
    def energy(self, dbm_layer_states, inv_temps=None):
        
        energy = 0
        
        #add the bias components of the energy function
        for bias_num in range(self.num_layers):
            
            energy -= T.dot(dbm_layer_states[bias_num], self.biases[bias_num])
        
        #add the weight components of the energy function
        for weight_num in range(self.num_layers - 1):
            
            energy -= T.sum((dbm_layer_states[weight_num] * T.dot(dbm_layer_states[weight_num + 1], T.transpose(self.weights[weight_num]))), axis=1)
        
        #multiply by the inverse temperatures if specified (for the Metropolis-Hastings acceptance rule in Wang-Landau adaptive simulated annealing)
        if inv_temps is not None:
            energy *= inv_temps
        
        return energy
    
    ##calculate one step of the mean field update for all odd hidden layers given the even hidden layers and the visible layer (i.e. layer 0)
    #def mean_field_odds(self, layer_sigmoids, inv_temps=None):
        
        #new_layer_sigmoids = [lsi for lsi in layer_sigmoids]
        
        ##start with the 1st hidden layer (with index 1) and for every other hidden layer (i.e. all odd hidden layers)
        #for layer_num in range(1, self.num_layers, 2):
            
            #if self.is_last_layer(layer_num):
                #down_states, up_states = layer_sigmoids[layer_num - 1], None
            #else:
                #down_states, up_states = layer_sigmoids[layer_num - 1], layer_sigmoids[layer_num + 1]

            #new_layer_sigmoids[layer_num] = self.layers[layer_num].sigmoid(down_states, up_states, inv_temps)
                
        #return new_layer_sigmoids
    
    ##calculate one step of the mean field update for all even hidden layers given the odd hidden layers (does not include the visible layer (i.e. layer 0))
    #def mean_field_evens(self, layer_sigmoids, inv_temps=None):
        
        #new_layer_sigmoids = [lsi for lsi in layer_sigmoids]
        
        ##start with the 2nd hidden layer (with index 2) and for every other hidden layer (i.e. all even hidden layers)
        ##the visible layer (i.e. layer 0) is not updated since the mean field update is data-dependent (i.e. the visible layer is fixed)
        #for layer_num in range(2, self.num_layers, 2):
            
            #if self.is_last_layer(layer_num):
                #down_states, up_states = layer_sigmoids[layer_num - 1], None
            #else:
                #down_states, up_states = layer_sigmoids[layer_num - 1], layer_sigmoids[layer_num + 1]

            #new_layer_sigmoids[layer_num] = self.layers[layer_num].sigmoid(down_states, up_states, inv_temps)
                
        #return new_layer_sigmoids
    
    #calculate one step of the Gibbs sample for all odd hidden layers given the even hidden layers and the visible layer (i.e. layer 0)
    def gibbs_sample_odds(self, layer_samples, inv_temps=None):
        
        new_layer_samples = [lsa for lsa in layer_samples]
        
        #start with the 1st hidden layer (with index 1) and for every other hidden layer (i.e. all odd hidden layers)
        for layer_num in range(1, self.num_layers, 2):
            
            if self.is_last_layer(layer_num):
                down_states, up_states = layer_samples[layer_num - 1], None
            else:
                down_states, up_states = layer_samples[layer_num - 1], layer_samples[layer_num + 1]

            new_layer_samples[layer_num] = self.layers[layer_num].sample(down_states, up_states, inv_temps)
            
        return new_layer_samples
    
    #calculate one step of the Gibbs sample for all even hidden layers and the visible layer (i.e. layer 0) given the odd hidden layers
    def gibbs_sample_evens(self, layer_samples, inv_temps=None):
        
        new_layer_samples = [lsa for lsa in layer_samples]
        
        #start with the visible layer (i.e. layer 0) and for every other hidden layer (i.e. all even hidden layers)
        for layer_num in range(0, self.num_layers, 2):
            
            if self.is_vis_layer(layer_num):
                down_states, up_states = None, layer_samples[layer_num + 1]
            elif self.is_last_layer(layer_num):
                down_states, up_states = layer_samples[layer_num - 1], None
            else:
                down_states, up_states = layer_samples[layer_num - 1], layer_samples[layer_num + 1]

            new_layer_samples[layer_num] = self.layers[layer_num].sample(down_states, up_states, inv_temps)
    
        return new_layer_samples
    
    ##calculate one full step of the mean field update by mean field updating the odd layers and then the even layers
    #def mean_field_once(self, layer_sigmoids, inv_temps=None):
        
        #odd_layer_sigmoids = self.mean_field_odds(layer_sigmoids, inv_temps)
        #even_layer_sigmoids = self.mean_field_evens(odd_layer_sigmoids, inv_temps)
        
        #return even_layer_sigmoids
    
    def mean_field_once(self, layer_sigmoids, inv_temps=None):
        
        new_layer_sigmoids = [None for i in range(self.num_layers)]
        
        new_layer_sigmoids[0] = layer_sigmoids[0]
        
        for layer_num in range(1, self.num_layers):
            
            if self.is_last_layer(layer_num):
                down_states, up_states = new_layer_sigmoids[layer_num - 1], None
            else:
                down_states, up_states = new_layer_sigmoids[layer_num - 1], layer_sigmoids[layer_num + 1]

            new_layer_sigmoids[layer_num] = self.layers[layer_num].sigmoid(down_states, up_states, inv_temps)
                
        return new_layer_sigmoids
       
    #calculate the result of multiple steps of the mean field update
    def mean_field_many(self, num_mean_field_steps, layer_sigmoids, inv_temps=None):
        
        step_layer_sigmoids = [None for i in range(num_mean_field_steps + 1)]
        
        step_layer_sigmoids[0] = layer_sigmoids
        
        for i in range(1, (num_mean_field_steps + 1)):
            
            step_layer_sigmoids[i] = self.mean_field_once(step_layer_sigmoids[i - 1], inv_temps)
        
        final_layer_sigmoids = step_layer_sigmoids[-1]
        
        return final_layer_sigmoids
    
    #def mean_field_many(self, num_mean_field_steps, layer_sigmoids, inv_temps=None):
        
        #def bare_mean_field_once(*previous_layer_sigmoids):
            
            #next_layer_sigmoids = self.mean_field_once(previous_layer_sigmoids, inv_temps)
            
            #return next_layer_sigmoids
        
        #step_layer_sigmoids, updates = theano.scan(bare_mean_field_once, outputs_info=layer_sigmoids, n_steps=num_mean_field_steps)
        
        #final_layer_sigmoids = [step_layer_sigmoids[layer_num][-1] for layer_num in range(self.num_layers)]
        
        #return final_layer_sigmoids
    
    #calculate one full step of the Gibbs sample by Gibbs sampling the odd layers and then the even layers
    def gibbs_sample_once(self, layer_samples, inv_temps=None):
        
        odd_layer_samples = self.gibbs_sample_odds(layer_samples, inv_temps)
        even_layer_samples = self.gibbs_sample_evens(odd_layer_samples, inv_temps)
        
        return even_layer_samples
    
    #def gibbs_sample_once(self, layer_samples, inv_temps=None):
        
        #new_layer_samples = [None for i in range(self.num_layers)]
        
        #for layer_num in range(self.num_layers):
            
            #if self.is_vis_layer(layer_num):
                #down_states, up_states = None, layer_samples[layer_num + 1]
            #elif self.is_last_layer(layer_num):
                #down_states, up_states = new_layer_samples[layer_num - 1], None
            #else:
                #down_states, up_states = new_layer_samples[layer_num - 1], layer_samples[layer_num + 1]

            #new_layer_samples[layer_num] = self.layers[layer_num].sample(down_states, up_states, inv_temps)
        
        #return new_layer_samples
    
    #calculate the result of multiple steps of the Gibbs sample
    def gibbs_sample_many(self, num_gibbs_steps, layer_samples, inv_temps=None):
        
        step_layer_samples = [None for i in range(num_gibbs_steps + 1)]
        
        step_layer_samples[0] = layer_samples
        
        for i in range(1, (num_gibbs_steps + 1)):
            
            step_layer_samples[i] = self.gibbs_sample_once(step_layer_samples[i - 1], inv_temps)
        
        final_layer_samples = step_layer_samples[-1]
        
        return final_layer_samples
    
    #can't use scan since has trouble with step functions involving random variables, particularly binomials (used in DBMLayerBinary.sample)
    #def gibbs_sample_many(self, num_gibbs_steps, layer_samples, inv_temps=None):
        
        #def bare_gibbs_sample_once(*previous_layer_samples):
            
            #next_layer_samples = self.gibbs_sample_once(previous_layer_samples, inv_temps)
            
            #return next_layer_samples
        
        #step_layer_samples, updates = theano.scan(bare_gibbs_sample_once, outputs_info=layer_samples, n_steps=num_gibbs_steps)
        
        #final_layer_samples = [step_layer_samples[layer_num][-1] for layer_num in range(self.num_layers)]
        
        #return final_layer_samples
    
    #sample the next inverse temperature states given the current inverse temperature states
    def sample_inv_temp_states(self, inv_temp_states, num_inv_temps):
        
        #"boolean" (Theano doesn't have boolean types, but the following is functionally identical) masks for low, high, outer (i.e. low or high), and inner (i.e. neither high nor low) states 
        inv_temp_states_low = T.eq(inv_temp_states, 1)
        inv_temp_states_high = T.eq(inv_temp_states, num_inv_temps)
        inv_temp_states_out = T.ge((inv_temp_states_low + inv_temp_states_high), 1)
        inv_temp_states_in = 1 - inv_temp_states_out
        
        #sample moving up or down a state with equal probability (used for inner states)
        inv_temp_states_dif = 2 * self.theano_rng.binomial(n=1, p=.5, size=inv_temp_states.shape) - 1
        
        #set all low states to 2, all high states to the second highest state (i.e. num_inv_temps - 1), and all inner states to their sampled states
        sample_inv_temp_states = inv_temp_states_low * 2 + inv_temp_states_high * (num_inv_temps - 1) + inv_temp_states_in * (inv_temp_states + inv_temp_states_dif)
        
        #same "boolean" masks as above, though for the sampled inverese temperature states
        sample_inv_temp_states_low = T.eq(sample_inv_temp_states, 1)
        sample_inv_temp_states_high = T.eq(sample_inv_temp_states, num_inv_temps)
        sample_inv_temp_states_out = T.ge((sample_inv_temp_states_low + sample_inv_temp_states_high), 1)
        sample_inv_temp_states_in = 1 - sample_inv_temp_states_out
        
        #calculate the reverse (p(k_(n) | k_(n+1))) and forward (p(k_(n+1) | k_(n))) probabilities of the current and sampled states
        sample_reverse_probs = sample_inv_temp_states_out + (sample_inv_temp_states_in / 2.0)
        sample_forward_probs = inv_temp_states_out + (inv_temp_states_in / 2.0)
        
        return [sample_inv_temp_states, sample_reverse_probs, sample_forward_probs]
    
    #calculate one step of the Gibbs sample using Wang-Landau adaptive simulated annealing and current inverse temperature states
    #must be at least three different inverse temperature states (i.e. all_inv_temps.shape[0] >= 3)
    def wl_adaptive_sa(self, num_gibbs_steps, layer_samples, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor):
        
        num_inv_temps = all_inv_temps.shape[0]
        
        inv_temps = all_inv_temps[inv_temp_states - 1]
        
        new_layer_samples = self.gibbs_sample_many(num_gibbs_steps, layer_samples, inv_temps)

        sample_inv_temp_states, sample_reverse_probs, sample_forward_probs = self.sample_inv_temp_states(inv_temp_states, num_inv_temps)
        sample_inv_temps = all_inv_temps[sample_inv_temp_states - 1]
        
        current_energy = self.energy(new_layer_samples, inv_temps)
        sample_energy = self.energy(new_layer_samples, sample_inv_temps)
        
        current_adaptive_weight = column_select(adaptive_weights, (inv_temp_states - 1))
        sample_adaptive_weight = column_select(adaptive_weights, (sample_inv_temp_states - 1))
                
        #calculate Metropolis-Hastings rule probabilities and accept sampled inverse temperature according to probabilities
        mh_accept_prob = T.minimum(1, ((T.exp(-sample_energy) * sample_reverse_probs * current_adaptive_weight) / (T.exp(-current_energy) * sample_forward_probs * sample_adaptive_weight)))
        mh_accept = self.theano_rng.binomial(n=1, p=mh_accept_prob, size=mh_accept_prob.shape)
        
        new_inv_temp_states = T.switch(mh_accept, sample_inv_temp_states, inv_temp_states)
        
        #update adaptive weights according to Wang-Landau algorithm
        inc_adaptive_weights = column_select(adaptive_weights, (new_inv_temp_states - 1))
        new_adaptive_weights = T.set_subtensor(inc_adaptive_weights, (inc_adaptive_weights * (1 + weight_adapting_factor)))
        #renormalize sum of adaptive weights per particle (since only the ratios between adaptive weights are used)
        new_adaptive_weights /= T.sum(new_adaptive_weights, axis=1).dimshuffle(0, 'x')
        
        return [new_layer_samples, new_inv_temp_states, new_adaptive_weights]

    #vis_input must be passed (instead of calling self.gen_vis_input) since theano.clone called by SemiSupDBMBinary's initializer can't handle shape information passed to RandomStreams methods
    def data_energy_layer_state(self, num_mean_field_steps, vis_input):
        
        num_vis_inputs = vis_input.shape[0]
        
        mean_field_layer_sigmoids = [None for i in range(self.num_layers)]
        
        mean_field_layer_sigmoids[0] = vis_input
        
        #set the initial mean field sigmoids to .5 for all hidden layers
        for layer_num in range(1, self.num_layers):
            
            mean_field_layer_sigmoids[layer_num] = T.cast(T.alloc(.5, num_vis_inputs, self.layers[layer_num].num_units), dtype='float64')
            #mean_field_layer_sigmoids[layer_num] = self.theano_rng.uniform(size=(num_vis_inputs, self.layers[layer_num].num_units))
                
        #calculate the mean field approximation given the visible input
        new_mean_field_layer_sigmoids = self.mean_field_many(num_mean_field_steps, mean_field_layer_sigmoids)
        
        return new_mean_field_layer_sigmoids        
    
    def model_energy_layer_state(self, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor):
        
        #calculate the slow chain gibbs sample and the fast chain gibbs sample (using the Wang-Landau adaptive simulated annealing algorithm)
        new_slow_gibbs_layer_samples = self.gibbs_sample_many(num_gibbs_steps, slow_gibbs_layer_samples)
        new_fast_gibbs_layer_samples, new_inv_temp_states, new_adaptive_weights = self.wl_adaptive_sa(num_gibbs_steps, fast_gibbs_layer_samples, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor)
        
        new_inv_temp_state_low = T.eq(new_inv_temp_states, 1)
        current_fast_push_changed = T.ge((fast_push_changed + new_inv_temp_state_low), 1)
        push = push_fast_slow * current_fast_push_changed
        
        new_low_fast_gibbs_layer_samples = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            new_low_fast_gibbs_layer_samples[layer_num] = T.switch(new_inv_temp_state_low.dimshuffle(0, 'x'), new_fast_gibbs_layer_samples[layer_num], low_fast_gibbs_layer_samples[layer_num])
            
            new_slow_gibbs_layer_samples[layer_num] = T.switch(push.dimshuffle(0, 'x'), new_low_fast_gibbs_layer_samples[layer_num], new_slow_gibbs_layer_samples[layer_num])
        
        new_fast_push_changed = T.switch(push_fast_slow, T.zeros_like(fast_push_changed), current_fast_push_changed)
        
        return [new_slow_gibbs_layer_samples, new_fast_gibbs_layer_samples, new_low_fast_gibbs_layer_samples, new_fast_push_changed, new_inv_temp_states, new_adaptive_weights]
    
    def data_loglike_grad(self, data_layer_states, model_layer_states):
               
        #calculate the average energies of the data-dependent dbm states and the gibbs sample particles dbm states
        data_energy = T.mean(self.energy(data_layer_states))
        model_energy = T.mean(self.energy(model_layer_states))
        
        data_loglike_cost = data_energy - model_energy
        
        data_loglike_cost_layer_states = data_layer_states + model_layer_states
        
        data_loglike_grad_params = T.grad(data_loglike_cost, self.params, consider_constant=data_loglike_cost_layer_states)
        
        return data_loglike_grad_params
    
    def sparsity_grad(self, data_layer_states, layer_sparsity, sparsity_target, sparsity_decay):
        
        new_layer_sparsity = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            current_layer_sparsity = T.mean(data_layer_states[layer_num], axis=0)
            new_layer_sparsity[layer_num] = sparsity_decay * layer_sparsity[layer_num] + (1 - sparsity_decay) * current_layer_sparsity
        
        sparsity_cost = 0
        
        for layer_num in range(1, self.num_layers):
           
            sparsity_cross_entropy = T.nnet.binary_crossentropy(new_layer_sparsity[layer_num], sparsity_target)
            sparsity_cost += T.sum(sparsity_cross_entropy)
                    
        sparsity_grad_params = T.grad(sparsity_cost, self.params, disconnected_inputs='warn')
        
        return [new_layer_sparsity, sparsity_grad_params]
    
    def gen_cost_updates(self, num_mean_field_steps, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor, layer_sparsity, sparsity_target, sparsity_decay, sparsity_cost_weight):
        
        #calculate the learning statistics for the data (i.e. visible input), slow chain gibbs sample particles, and fast chain gibbs sample particles
        new_mean_field_layer_sigmoids = self.data_energy_layer_state(num_mean_field_steps, self.gen_vis_input)
        new_slow_gibbs_layer_samples, new_fast_gibbs_layer_samples, new_low_fast_gibbs_layer_samples, new_fast_push_changed, new_inv_temp_states, new_adaptive_weights = self.model_energy_layer_state(num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor)
        
        data_layer_states = new_mean_field_layer_sigmoids
        model_layer_states = new_slow_gibbs_layer_samples
        
        #calculate the gradient wrt to the model parameters
        data_loglike_grad_params = self.data_loglike_grad(data_layer_states, model_layer_states)
        new_layer_sparsity, sparsity_grad_params = self.sparsity_grad(data_layer_states, layer_sparsity, sparsity_target, sparsity_decay)
        
        param_changes = collections.OrderedDict()
        non_param_updates = collections.OrderedDict()
        
        #update the model parameters in direction of the negative generative cost gradient
        for param, data_loglike_grad_param, sparsity_grad_param in zip(self.params, data_loglike_grad_params, sparsity_grad_params):
            
            param_changes[param] = -(data_loglike_grad_param + (sparsity_cost_weight * sparsity_grad_param))
        
        #update the slow chain, fast chain, and low fast chain gibbs sample particles
        for layer_num in range(self.num_layers):
            
            non_param_updates[slow_gibbs_layer_samples[layer_num]] = new_slow_gibbs_layer_samples[layer_num]
            non_param_updates[fast_gibbs_layer_samples[layer_num]] = new_fast_gibbs_layer_samples[layer_num]
            non_param_updates[low_fast_gibbs_layer_samples[layer_num]] = new_low_fast_gibbs_layer_samples[layer_num]
        
        non_param_updates[fast_push_changed] = new_fast_push_changed
        non_param_updates[inv_temp_states] = new_inv_temp_states
        non_param_updates[adaptive_weights] = new_adaptive_weights
        
        for layer_num in range(self.num_layers):
            
            non_param_updates[layer_sparsity[layer_num]] = new_layer_sparsity[layer_num]
        
        return [param_changes, non_param_updates]
    
    def burn_in_updates(self, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor):
        
        new_slow_gibbs_layer_samples, new_fast_gibbs_layer_samples, new_low_fast_gibbs_layer_samples, new_fast_push_changed, new_inv_temp_states, new_adaptive_weights = self.model_energy_layer_state(num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor)
                        
        updates = collections.OrderedDict()
        
        #update the slow chain, fast chain, and low fast chain gibbs sample particles
        for layer_num in range(self.num_layers):
            
            updates[slow_gibbs_layer_samples[layer_num]] = new_slow_gibbs_layer_samples[layer_num]
            updates[fast_gibbs_layer_samples[layer_num]] = new_fast_gibbs_layer_samples[layer_num]
            updates[low_fast_gibbs_layer_samples[layer_num]] = new_low_fast_gibbs_layer_samples[layer_num]
        
        updates[fast_push_changed] = new_fast_push_changed
        updates[inv_temp_states] = new_inv_temp_states
        updates[adaptive_weights] = new_adaptive_weights
        
        return updates
    
    def inv_temp_states_window_updates(self, inv_temp_states, inv_temp_states_window):
        
        #shift windowed inverse temperature states one to the left
        shifted_inv_temp_states_window = T.set_subtensor(inv_temp_states_window[:, :-1], inv_temp_states_window[:, 1:])
        #replace the most recent inverse temperature state by the current inverse temperature state
        new_inv_temp_states_window = T.set_subtensor(shifted_inv_temp_states_window[:, -1], inv_temp_states)
        
        updates = collections.OrderedDict()
        
        updates[inv_temp_states_window] = new_inv_temp_states_window
        
        return updates
    
    def sparsity_stats(self, layer_sparsity):
        
        mean_hidden_sparsity = T.mean(T.concatenate(layer_sparsity[1:]))
        std_hidden_sparsity = T.std(T.concatenate(layer_sparsity[1:]))
        
        return [mean_hidden_sparsity, std_hidden_sparsity]
    
    def ais_log_partition(self, num_ais_runs, num_gibbs_steps, num_ais_inv_temps):
        
        all_ais_inv_temp_values = np.linspace(start=0, stop=1, num=num_ais_inv_temps)
        all_ais_inv_temps = theano.shared(value=all_ais_inv_temp_values, name='All AIS inverse temperatures', borrow=True)
        
        ais_layer_samples = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            ais_layer_samples_value = R.binomial(n=1, p=.5, size=(num_ais_runs, self.layers[layer_num].num_units))
            ais_layer_samples[layer_num] = theano.shared(value=ais_layer_samples_value, name='AIS samples for layer {}'.format(layer_num), borrow=True)
        
        log_importance_weights_value = np.zeros(shape=(num_ais_runs,))
        log_importance_weights = theano.shared(value=log_importance_weights_value, name='AIS importance weights', borrow=True)
        
        ais_inv_temp_state = T.lscalar('Current AIS inverse temperature state')
        
        previous_ais_inv_temps = T.alloc(all_ais_inv_temps[ais_inv_temp_state - 1], num_ais_runs)
        current_ais_inv_temps = T.alloc(all_ais_inv_temps[ais_inv_temp_state], num_ais_runs)
        
        log_prob_ratio = -self.energy(ais_layer_samples, current_ais_inv_temps) + self.energy(ais_layer_samples, previous_ais_inv_temps)
        
        new_log_importance_weights = log_importance_weights + log_prob_ratio
        new_ais_layer_samples = self.gibbs_sample_many(num_gibbs_steps, ais_layer_samples, current_ais_inv_temps)
        
        max_log_importance_weight = T.max(log_importance_weights)
        scaled_log_importance_weights = log_importance_weights - max_log_importance_weight
        log_partition_estimate = (self.total_num_states * T.log(2)) + max_log_importance_weight + T.mean(T.exp(scaled_log_importance_weights))
        
        updates = collections.OrderedDict()
        
        updates[log_importance_weights] = new_log_importance_weights
        
        for layer_num in range(self.num_layers):
            
            updates[ais_layer_samples[layer_num]] = new_ais_layer_samples[layer_num]
        
        next_log_importance_weights = theano.function([ais_inv_temp_state], new_log_importance_weights, updates=updates)
        stable_log_partition = theano.function([], log_partition_estimate)
                
        for ais_it in range(1, num_ais_inv_temps):
            
            next_log_importance_weights(ais_it)
            
        log_partition = stable_log_partition()
        
        return log_partition
    
    def mean_data_log_likelihood(self, log_partition_estimate, num_mean_field_steps):
        
        new_mean_field_layer_sigmoids = self.data_energy_layer_state(num_mean_field_steps, self.gen_vis_input)
        
        var_energy = self.energy(new_mean_field_layer_sigmoids)
        
        var_entropy = 0
        
        for layer_num in range(1, self.num_layers):
            
            #ones_entropy = -new_mean_field_layer_sigmoids[layer_num] * T.log(new_mean_field_layer_sigmoids[layer_num])
            #zeros_entropy = -(1 - new_mean_field_layer_sigmoids[layer_num]) * T.log(1 - new_mean_field_layer_sigmoids[layer_num])
            
            #var_entropy += T.sum((ones_entropy + zeros_entropy), axis=1)
            
            entropy = T.nnet.binary_crossentropy(new_mean_field_layer_sigmoids[layer_num], new_mean_field_layer_sigmoids[layer_num])
            var_entropy += T.sum(entropy, axis=1)       
        
        var_lower_bound = -var_energy + var_entropy - log_partition_estimate
        
        mean_var_lower_bound = T.mean(var_lower_bound)
        
        return mean_var_lower_bound
    
    def create_theano_variables(self, num_particles, num_inv_temps, low_inv_temp, display_num_past_inv_temp_states):
        
        #all fast chain sample particles start out in state 1
        inv_temp_states_value = np.full(fill_value=1, shape=num_particles, dtype='int64')
        inv_temp_states = theano.shared(value=inv_temp_states_value, name='Fast chain inverse temperature states', borrow=True)
        
        inv_temp_states_window_value = np.full(fill_value=1, shape=(num_particles, display_num_past_inv_temp_states), dtype='int64')
        inv_temp_states_window = theano.shared(value=inv_temp_states_window_value, name='Particle 1 past inverse temperature states', borrow=True)
        
        all_inv_temps_value = np.linspace(start=1, stop=low_inv_temp, num=num_inv_temps)
        all_inv_temps = theano.shared(value=all_inv_temps_value, name='All inverse temperatures', borrow=True)
        
        #assume all partitions (i.e. inverse temperature states) are equally likely at the beginning
        adaptive_weights_value = np.full(fill_value=(1.0 / num_inv_temps), shape=(num_particles, num_inv_temps))
        adaptive_weights = theano.shared(value=adaptive_weights_value, name='Fast chain adaptive weights', borrow=True)
        
        #"boolean" of whether to push the last fast chain gibbs sample particles to the slow chain gibbs sample particles
        push_fast_slow = T.lscalar('Push fast to slow')
        log_partition = T.dscalar('AIS log partition estimate')
        
        layer_sparsity = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            layer_sparsity_values = np.full(fill_value=.5, shape=(self.layers[layer_num].num_units,))
            layer_sparsity[layer_num] = theano.shared(value=layer_sparsity_values, name='Average unit activation probabilities', borrow=True)
        
        #create theano shared variables for layer samples of the slow chain gibbs sample particles, fast chain gibbs sample particles, and low fast chain gibbs sample particles
        slow_gibbs_layer_samples = [None for i in range(self.num_layers)]
        fast_gibbs_layer_samples = [None for i in range(self.num_layers)]
        low_fast_gibbs_layer_samples = [None for i in range(self.num_layers)]
        
        for layer_num in range(self.num_layers):
            
            #starting gibbs sample particle values are uniform binary variables
            slow_gibbs_layer_samples_value = R.binomial(n=1, p=.5, size=(num_particles, self.layers[layer_num].num_units))
            fast_gibbs_layer_samples_value = R.binomial(n=1, p=.5, size=(num_particles, self.layers[layer_num].num_units))
            #initial last fast chain layer states are the same as the fast chain layer states since the particles start out in inverse temperature state 1
            low_fast_gibbs_layer_samples_value = np.copy(fast_gibbs_layer_samples_value)
            
            slow_gibbs_layer_samples[layer_num] = theano.shared(value=slow_gibbs_layer_samples_value, name='Slow chain samples for layer {}'.format(layer_num), borrow=True)
            fast_gibbs_layer_samples[layer_num] = theano.shared(value=fast_gibbs_layer_samples_value, name='Fast chain samples for layer {}'.format(layer_num), borrow=True)
            low_fast_gibbs_layer_samples[layer_num] = theano.shared(value=low_fast_gibbs_layer_samples_value, name='Last low fast chain samples for layer {}'.format(layer_num), borrow=True)
        
        fast_push_changed_value = np.zeros(shape=(num_particles,))
        fast_push_changed = theano.shared(value=fast_push_changed_value, name='Fast chain samples changed since last push', borrow=True)
        
        return [inv_temp_states, inv_temp_states_window, all_inv_temps, adaptive_weights, push_fast_slow, log_partition, layer_sparsity, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed]
    
    def train(self, gen_training_input, gen_validation_input, num_epochs, batch_size, num_particles, num_burn_ins, num_mean_field_steps, num_gibbs_steps, num_inv_temps, low_inv_temp, weight_adapting_factor, learning_rate_func, sparsity_target, sparsity_decay, sparsity_cost_weight, momentum, fast_slow_lag, num_ais_runs, num_ais_inv_temps, log_likelihood_period, display_on, display_period, display_num_past_inv_temp_states, display_slow_samples_grid_dims, display_gen_vis_input_dims):
                
        gen_training_input_size = gen_training_input.get_value(borrow=True).shape[0]
        num_batches = gen_training_input_size // batch_size
        
        print('\nCreating theano variables.')
        
        inv_temp_states, inv_temp_states_window, all_inv_temps, adaptive_weights, push_fast_slow, log_partition, layer_sparsity, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed = self.create_theano_variables(num_particles, num_inv_temps, low_inv_temp, display_num_past_inv_temp_states)
        
        learning_rate = T.dscalar('Learning rate')
        #shuffles the training input
        shuffle_indices = T.lvector('Shuffle indices')
        #index of the batch of training data
        batch_index = T.lscalar('Batch index')
        
        print('Theano variable creation completed!')
        
        print('\nBuilding symbolic theano graphs.')
        
        #symbolic computations for shared variable updates and generative cost
        burn_in_updates = self.burn_in_updates(num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor)
        
        param_changes_current, non_param_updates = self.gen_cost_updates(num_mean_field_steps, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor, layer_sparsity, sparsity_target, sparsity_decay, sparsity_cost_weight)
        
        mean_hidden_sparsity, std_hidden_sparsity = self.sparsity_stats(layer_sparsity)
        
        mean_data_log_likelihood = self.mean_data_log_likelihood(log_partition, num_mean_field_steps)
        
        if display_on:
            inv_temp_states_window_updates = self.inv_temp_states_window_updates(inv_temp_states, inv_temp_states_window)
        
        #add momentum to the parameter updates
        param_updates = collections.OrderedDict()
        
        for param, param_change_last in zip(self.params, self.param_changes_last):
            
            param_change = (momentum * param_change_last) + (learning_rate * param_changes_current[param])
            
            param_updates[param] = param + param_change
            param_updates[param_change_last] = param_change
        
        gen_train_updates = collections.OrderedDict()
        
        for old_value, new_value in (param_updates.items() + non_param_updates.items()):
            
            gen_train_updates[old_value] = new_value
        
        particle_burn_in = theano.function([push_fast_slow], None, updates=burn_in_updates)
        
        gen_train = theano.function([learning_rate, push_fast_slow, shuffle_indices, batch_index], None, updates=gen_train_updates, givens={self.gen_vis_input: gen_training_input[shuffle_indices[(batch_index * batch_size):((batch_index + 1) * batch_size)]]})
        
        sparsity_observe = theano.function([], [mean_hidden_sparsity, std_hidden_sparsity])
        
        var_lower_bound = theano.function([log_partition], mean_data_log_likelihood, givens={self.gen_vis_input: gen_validation_input})
        
        if display_on:            
            window_shift = theano.function([], None, updates=inv_temp_states_window_updates)
        
        print('Symbolic theano graphs building completed!')
        
        print('\nStarting sample particle burn in.')
        
        particle_update_count = 0
                
        #burn in the slow chain gibbs sample particles and fast chain gibbs sample particles
        for burn_in_count in range(num_burn_ins):
            
            current_push_fast_slow = 1 if (particle_update_count % fast_slow_lag == 0) else 0
            
            particle_burn_in(current_push_fast_slow)
            
            if display_on:
                window_shift()
            
            particle_update_count += 1
            
            #print('\tBurn in completion percentage is {0:06.2f}%.'.format(100 * ((burn_in_count + 1) / float(num_burn_ins))), end='\r')
            
        print('Sample particle burn in completed!')
                
        print('\nStarting training.')
        
        training_start_time = time.time()
        
        param_update_count = 0
        
        if display_on:
            particle_display = plt.figure(1)
            window_shift_display = plt.figure(2)
                        
            plt.show(block=False)
            
            #display slow gibbs particle samples before training
            self.display_sample_info(inv_temp_states_window, slow_gibbs_layer_samples, particle_display, display_slow_samples_grid_dims, display_gen_vis_input_dims)
            
            last_display_time = time.time()
                
        for epoch_count in range(num_epochs):
                       
            #print('\n\tStarting training of epoch {}.'.format(epoch_count + 1))
            
            if (epoch_count % log_likelihood_period) == 0:
                
                log_partition_estimate = self.ais_log_partition(num_ais_runs, num_gibbs_steps, num_ais_inv_temps)
                valid_mean_log_likelihood = var_lower_bound(log_partition_estimate)
                                
                print('\n\tGenerative validation input mean log likelihood > {0} nats.'.format(valid_mean_log_likelihood))
                
            current_shuffle_indices = R.permutation(gen_training_input_size)
            current_learning_rate = learning_rate_func(epoch_count)
            
            epoch_start_time = time.time()
            
            for current_batch_index in range(num_batches):
                
                current_push_fast_slow = 1 if ((particle_update_count % fast_slow_lag) == 0) else 0
                                                
                gen_train(current_learning_rate, current_push_fast_slow, current_shuffle_indices, current_batch_index)
                
                if display_on:
                    window_shift()
                    
                    #if it's been more than display_period seconds since the last slow gibbs samples were displayed
                    if (time.time() - last_display_time) > display_period:
                        #display the slow gibbs samples
                        self.display_sample_info(inv_temp_states_window, slow_gibbs_layer_samples, particle_display, display_slow_samples_grid_dims, display_gen_vis_input_dims)
                        
                        last_display_time = time.time()
                
                particle_update_count += 1
                param_update_count += 1
                                
                #print('\t\tEpoch completion percentage is {0:06.2f}%.'.format(100 * ((current_batch_index + 1) / float(num_batches))), end='\r')
                        
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            sparsity_mean, sparsity_std = sparsity_observe()
            
            print('\n\tTraining epoch {0} completed in {1:.2f} seconds!\n\t\tTotal number of parameter updates is {2}.\n\t\tSparsity mean is {3}.\n\t\tSparsity standard deviation is {4}.'.format((epoch_count + 1), epoch_duration, param_update_count, sparsity_mean, sparsity_std))
   
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        print('Training on {} epochs completed in {} seconds!'.format(num_epochs, training_duration))
    
    def display_sample_info(self, inv_temp_states_window, layer_samples, particle_display, display_slow_samples_grid_dims, display_gen_vis_input_dims):
        
        gen_vis_input_sigmoids_value = ((self.layers[0]).sigmoid(None, layer_samples[1])).eval()
        
        num_gen_samples = gen_vis_input_sigmoids_value.shape[0]
        num_grid_spaces = display_slow_samples_grid_dims[0] * display_slow_samples_grid_dims[1]
        
        num_display_samples = min(num_gen_samples, num_grid_spaces)
        gen_vis_input_height, gen_vis_input_width = display_gen_vis_input_dims[0], display_gen_vis_input_dims[1]
        
        plt.figure(1)
        plt.clf()
                
        display_grid = ag.ImageGrid(particle_display, 111, nrows_ncols=display_slow_samples_grid_dims, axes_pad=.1, label_mode='1')
        
        for i in range(num_display_samples):
            
            display_grid[i].imshow(gen_vis_input_sigmoids_value[i].reshape(gen_vis_input_height, gen_vis_input_width), interpolation='none', cmap=mpl.cm.gray)
        
        plt.draw()
        
        plt.figure(2)
        plt.clf()
        
        plt.plot(inv_temp_states_window.get_value(borrow=True)[0])
        
        plt.draw()

class HiddenLayer(object):
    
    def __init__(self, num_inputs, num_outputs, weights_mean_init, weights_std_init, biases_init, sup_vis_input=None, numpy_rng=None, theano_rng=None):
        
        if sup_vis_input is None:
            self.sup_vis_input = T.dmatrix('Supervised visible input')
        else:
            self.sup_vis_input = sup_vis_input
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        weights_value = self.numpy_rng.normal(loc=weights_mean_init, scale=weights_std_init, size=(self.num_inputs, self.num_outputs))
        self.weights = theano.shared(value=weights_value, name='Hidden layer weights', borrow=True)
        
        biases_value = np.full(fill_value=biases_init, shape=(self.num_outputs,))
        self.biases = theano.shared(value=biases_value, name='Hidden layer biases', borrow=True)
        
        self.params = [self.weights, self.biases]
        self.param_changes_last = [theano.shared(value=np.zeros_like(param.get_value(borrow=True)), name='{} change last'.format(param.name), borrow=True) for param in self.params]
    
    def output(self):
        
        hidden_activations = T.dot(self.sup_vis_input, self.weights) + self.biases
        hidden_sigmoids = T.nnet.sigmoid(hidden_activations)
        
        return hidden_sigmoids

class LogRegLayer(object):
    
    def __init__(self, num_inputs, num_outputs, weights_mean_init, weights_std_init, biases_init, sup_vis_input=None, sup_label_input=None, numpy_rng=None, theano_rng=None):
        
        if sup_vis_input is None:
            self.sup_vis_input = T.dmatrix('Supervised visible input')
        else:
            self.sup_vis_input = sup_vis_input
        
        if sup_label_input is None:
            self.sup_label_input = T.lvector('Supervised label input')
        else:
            self.sup_label_input = sup_label_input
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        weights_value = self.numpy_rng.normal(loc=weights_mean_init, scale=weights_std_init, size=(self.num_inputs, self.num_outputs))
        self.weights = theano.shared(value=weights_value, name='Logistic regression layer weights', borrow=True)
        
        biases_value = np.full(fill_value=biases_init, shape=(self.num_outputs,))
        self.biases = theano.shared(value=biases_value, name='Logistic regression layer biases', borrow=True)
        
        self.params = [self.weights, self.biases]
        self.param_changes_last = [theano.shared(value=np.zeros_like(param.get_value(borrow=True)), name='{} change last'.format(param.name), borrow=True) for param in self.params]
    
    def output(self):
        
        log_reg_activations = T.dot(self.sup_vis_input, self.weights) + self.biases
        log_reg_softmaxs = T.nnet.softmax(log_reg_activations)
        
        return log_reg_softmaxs
    
    def label_predictions(self):
    
        output = self.output()
        label_predictions = T.argmax(output, axis=1)
        
        return label_predictions
    
    def mean_neg_label_log_likelihood(self):
        
        output = self.output()
        label_probs = column_select(output, self.sup_label_input)
        mean_neg_label_log_likelihood = T.mean(-T.log(label_probs))
        
        return mean_neg_label_log_likelihood
    
    def label_prediction_accuracy(self):
        
        label_predictions = self.label_predictions()
        label_prediction_accuracy = T.mean(T.eq(label_predictions, self.sup_label_input))
        
        return label_prediction_accuracy

class MLPLayer(object):
    
    def __init__(self, num_inputs, num_hidden_units, num_outputs, hidden_weights_mean_init, hidden_weights_std_init, hidden_biases_init, log_reg_weights_mean_init, log_reg_weights_std_init, log_reg_biases_init, sup_vis_input=None, sup_label_input=None, numpy_rng=None, theano_rng=None):
        
        if sup_vis_input is None:
            self.sup_vis_input = T.dmatrix('Supervised visible input')
        else:
            self.sup_vis_input = sup_vis_input
        
        if sup_label_input is None:
            self.sup_label_input = T.lvector('Supervised label input')
        else:
            self.sup_label_input = sup_label_input
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.hidden_layer = HiddenLayer(num_inputs=self.num_inputs, num_outputs=num_hidden_units, weights_mean_init=hidden_weights_mean_init, weights_std_init=hidden_weights_std_init, biases_init=hidden_biases_init, sup_vis_input=self.sup_vis_input, numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
        
        log_reg_layer_input = self.hidden_layer.output()
        
        self.log_reg_layer = LogRegLayer(num_inputs=num_hidden_units, num_outputs=self.num_outputs, weights_mean_init=log_reg_weights_mean_init, weights_std_init=log_reg_weights_std_init, biases_init=log_reg_biases_init, sup_vis_input=log_reg_layer_input, sup_label_input=self.sup_label_input, numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
                
        self.params = self.hidden_layer.params + self.log_reg_layer.params
        self.param_changes_last = self.hidden_layer.param_changes_last + self.log_reg_layer.param_changes_last
    
    def output(self):
        
        return self.log_reg_layer.output()
    
    def label_predictions(self):
    
        return self.log_reg_layer.label_predictions()
    
    def mean_neg_label_log_likelihood(self):
        
        return self.log_reg_layer.mean_neg_label_log_likelihood()
    
    def label_prediction_accuracy(self):
        
        return self.log_reg_layer.label_prediction_accuracy()

class SemiSupDBMBinary(object):
    
    def __init__(self, dbm_layer_sizes, hidden_layer_size, log_reg_layer_size, num_mean_field_steps, dbm_layer_weights_mean_init, dbm_layer_weights_std_init, dbm_layer_biases_init, hidden_weights_mean_init, hidden_weights_std_init, hidden_biases_init, log_reg_weights_mean_init, log_reg_weights_std_init, log_reg_biases_init, sup_vis_input=None, gen_vis_input=None, sup_label_input=None, numpy_rng=None, theano_rng=None):
        
        if sup_vis_input is None:
            self.sup_vis_input = T.dmatrix('Supervised visible input')
        else:
            self.sup_vis_input = sup_vis_input
        
        if gen_vis_input is None:
            self.gen_vis_input = T.dmatrix('Generative visible input')
        else:
            self.gen_vis_input = gen_vis_input
        
        if sup_label_input is None:
            self.sup_label_input = T.lvector('Supervised label input')
        else:
            self.sup_label_input = sup_label_input
        
        if numpy_rng is None:
            self.numpy_rng = R.RandomState(3142)
        else:
            self.numpy_rng = numpy_rng
            
        if theano_rng is None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(self.numpy_rng.randint(2**31))
        else:
            self.theano_rng = theano_rng
        
        self.num_inputs = dbm_layer_sizes[0]
        self.num_outputs = log_reg_layer_size
        self.num_mean_field_steps = num_mean_field_steps
        
        self.gen_dbm_binary = GenDBMBinary(layer_sizes=dbm_layer_sizes, layer_weights_mean_init=dbm_layer_weights_mean_init, layer_weights_std_init=dbm_layer_weights_std_init, layer_biases_init=dbm_layer_biases_init, gen_vis_input=self.gen_vis_input, numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
        
        sup_mean_field_layer_sigmoids = self.gen_dbm_binary.data_energy_layer_state(self.num_mean_field_steps, self.sup_vis_input)
        
        mlp_layer_input = sup_mean_field_layer_sigmoids[-1]
        
        self.mlp_layer = MLPLayer(num_inputs=dbm_layer_sizes[-1], num_hidden_units=hidden_layer_size, num_outputs=self.num_outputs, hidden_weights_mean_init=hidden_weights_mean_init, hidden_weights_std_init=hidden_weights_std_init, hidden_biases_init=hidden_biases_init, log_reg_weights_mean_init=log_reg_weights_mean_init, log_reg_weights_std_init=log_reg_weights_std_init, log_reg_biases_init=log_reg_biases_init, sup_vis_input=mlp_layer_input, sup_label_input=self.sup_label_input, numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
        
        self.params = self.gen_dbm_binary.params + self.mlp_layer.params
        self.param_changes_last = self.gen_dbm_binary.param_changes_last + self.mlp_layer.param_changes_last
    
    def output(self):
        
        return self.mlp_layer.output()
    
    def label_predictions(self):
    
        return self.mlp_layer.label_predictions()
    
    def mean_neg_label_log_likelihood(self):
        
        return self.mlp_layer.mean_neg_label_log_likelihood()
    
    def label_prediction_accuracy(self):
        
        return self.mlp_layer.label_prediction_accuracy()
    
    def sup_grad(self):
        
        sup_cost = self.mean_neg_label_log_likelihood()
        
        sup_grad_params = T.grad(sup_cost, self.params, disconnected_inputs='warn')
        
        return sup_grad_params
    
    def sup_cost_updates(self):
        
        #calculate the gradient wrt to the model parameters
        sup_grad_params = self.sup_grad()
        
        sup_param_changes = collections.OrderedDict()
        
        #update the model parameters in direction of the negative supervised cost gradient
        for param, sup_grad_param in zip(self.params, sup_grad_params):
            
            sup_param_changes[param] = -sup_grad_param
        
        return sup_param_changes
    
    def semi_sup_cost_update(self, sup_cost_weight, gen_cost_weight, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor, layer_sparsity, sparsity_target, sparsity_decay, sparsity_cost_weight):
        
        sup_param_changes = self.sup_cost_updates()
        gen_param_changes, gen_non_param_updates = self.gen_dbm_binary.gen_cost_updates(self.num_mean_field_steps, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor, layer_sparsity, sparsity_target, sparsity_decay, sparsity_cost_weight)
        
        param_changes = collections.OrderedDict()
        
        for param in self.params:
            
            if sup_param_changes.has_key(param):
                sup_param_change = sup_param_changes[param]
            else:
                sup_param_change = 0
            
            if gen_param_changes.has_key(param):
                gen_param_change = gen_param_changes[param]
            else:
                gen_param_change = 0
            
            param_changes[param] = (sup_cost_weight * sup_param_change) + (gen_cost_weight * gen_param_change)
        
        return [param_changes, gen_non_param_updates]
    
    def train(self, sup_training_input, gen_training_input, sup_validation_input, gen_validation_input, sup_training_label_input, sup_validation_label_input, num_epochs, batch_size, num_particles, num_burn_ins, num_gibbs_steps, num_inv_temps, low_inv_temp, weight_adapting_factor, learning_rate_func, sup_gen_cost_weights_func, sparsity_target, sparsity_decay, sparsity_cost_weight, momentum, fast_slow_lag, num_ais_runs, num_ais_inv_temps, log_likelihood_period, display_on, display_period, display_num_past_inv_temp_states, display_slow_samples_grid_dims, display_gen_vis_input_dims):
        
        sup_training_input_size = sup_training_input.get_value(borrow=True).shape[0]
        gen_training_input_size = gen_training_input.get_value(borrow=True).shape[0]
        
        num_batches = sup_training_input_size // batch_size
        
        sup_batch_size = batch_size
        gen_batch_size = gen_training_input_size // num_batches
        
        print('\nCreating theano variables.')
        
        inv_temp_states, inv_temp_states_window, all_inv_temps, adaptive_weights, push_fast_slow, log_partition, layer_sparsity, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed = self.gen_dbm_binary.create_theano_variables(num_particles, num_inv_temps, low_inv_temp, display_num_past_inv_temp_states)
        
        learning_rate = T.dscalar('Learning rate')
        #shuffles the supervised training input
        sup_cost_weight = T.dscalar('Supervised cost weight')
        gen_cost_weight = T.dscalar('Generative cost weight')
        sup_shuffle_indices = T.lvector('Supervised shuffle indices')
        #shuffles the generative training input
        gen_shuffle_indices = T.lvector('Generative shuffle indices')
        #index of the batch of training data
        batch_index = T.lscalar('Batch index')
        
        print('Theano variable creation completed!')
        
        print('\nBuilding symbolic theano graphs.')
        
        gen_burn_in_updates = self.gen_dbm_binary.burn_in_updates(num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor)
        
        param_changes_current, non_param_updates = self.semi_sup_cost_update(sup_cost_weight, gen_cost_weight, num_gibbs_steps, slow_gibbs_layer_samples, fast_gibbs_layer_samples, low_fast_gibbs_layer_samples, fast_push_changed, push_fast_slow, inv_temp_states, all_inv_temps, adaptive_weights, weight_adapting_factor, layer_sparsity, sparsity_target, sparsity_decay, sparsity_cost_weight)
        
        label_prediction_accuracy = self.label_prediction_accuracy()
        
        mean_hidden_gen_sparsity, std_hidden_gen_sparsity = self.gen_dbm_binary.sparsity_stats(layer_sparsity)
        
        gen_mean_data_log_likelihood = self.gen_dbm_binary.mean_data_log_likelihood(log_partition, self.num_mean_field_steps)
        
        if display_on:
            inv_temp_states_window_updates = self.gen_dbm_binary.inv_temp_states_window_updates(inv_temp_states, inv_temp_states_window)
        
        #add momentum to the parameter updates
        param_updates = collections.OrderedDict()
        
        for param, param_change_last in zip(self.params, self.param_changes_last):
            
            param_change = (momentum * param_change_last) + (learning_rate * param_changes_current[param])
            
            param_updates[param] = param + param_change
            param_updates[param_change_last] = param_change
        
        semi_sup_train_updates = collections.OrderedDict()
        
        for old_value, new_value in (param_updates.items() + non_param_updates.items()):
            
            semi_sup_train_updates[old_value] = new_value
        
        particle_burn_in = theano.function([push_fast_slow], None, updates=gen_burn_in_updates)
        
        semi_sup_train = theano.function([learning_rate, sup_cost_weight, gen_cost_weight, push_fast_slow, sup_shuffle_indices, gen_shuffle_indices, batch_index], None, updates=semi_sup_train_updates, givens={self.sup_vis_input: sup_training_input[sup_shuffle_indices[(batch_index * sup_batch_size):((batch_index + 1) * sup_batch_size)]], self.gen_vis_input: gen_training_input[gen_shuffle_indices[(batch_index * gen_batch_size):((batch_index + 1) * gen_batch_size)]], self.sup_label_input: sup_training_label_input[sup_shuffle_indices[(batch_index * sup_batch_size):((batch_index + 1) * sup_batch_size)]]})
        
        label_accuracy_observe = theano.function([], label_prediction_accuracy, givens={self.sup_vis_input: sup_validation_input, self.sup_label_input: sup_validation_label_input})
        
        sparsity_observe = theano.function([], [mean_hidden_gen_sparsity, std_hidden_gen_sparsity])
        
        var_lower_bound = theano.function([log_partition], gen_mean_data_log_likelihood, givens={self.gen_vis_input: gen_validation_input})
        
        if display_on:            
            window_shift = theano.function([], None, updates=inv_temp_states_window_updates)
        
        print('Symbolic theano graphs building completed!')
        
        print('\nStarting sample particle burn in.')
        
        particle_update_count = 0
                
        #burn in the slow chain gibbs sample particles and fast chain gibbs sample particles
        for burn_in_count in range(num_burn_ins):
            
            current_push_fast_slow = 1 if (particle_update_count % fast_slow_lag == 0) else 0
            
            particle_burn_in(current_push_fast_slow)
            
            if display_on:
                window_shift()
            
            particle_update_count += 1
            
            #print('\tBurn in completion percentage is {0:06.2f}%.'.format(100 * ((burn_in_count + 1) / float(num_burn_ins))), end='\r')
            
        print('Sample particle burn in completed!')
                
        print('\nStarting training.')
        
        training_start_time = time.time()
        
        param_update_count = 0
        
        if display_on:
            particle_display = plt.figure(1)
            window_shift_display = plt.figure(2)
                        
            plt.show(block=False)
            
            #display slow gibbs particle samples before training
            self.gen_dbm_binary.display_sample_info(inv_temp_states_window, slow_gibbs_layer_samples, particle_display, display_slow_samples_grid_dims, display_gen_vis_input_dims)
            
            last_display_time = time.time()
                
        for epoch_count in range(num_epochs):
                       
            #print('\n\tStarting training of epoch {}.'.format(epoch_count + 1))
            
            if (epoch_count % log_likelihood_period) == 0:
                
                log_partition_estimate = self.gen_dbm_binary.ais_log_partition(num_ais_runs, num_gibbs_steps, num_ais_inv_temps)
                valid_mean_log_likelihood = var_lower_bound(log_partition_estimate)
                                
                print('\n\tGenerative validation input mean log likelihood > {0} nats.'.format(valid_mean_log_likelihood))
                
            current_sup_shuffle_indices = R.permutation(sup_training_input_size)
            current_gen_shuffle_indices = R.permutation(gen_training_input_size)
            
            current_learning_rate = learning_rate_func(epoch_count)
            current_sup_cost_weight, current_gen_cost_weight = sup_gen_cost_weights_func(epoch_count)
            
            epoch_start_time = time.time()
            
            for current_batch_index in range(num_batches):
                
                current_push_fast_slow = 1 if ((particle_update_count % fast_slow_lag) == 0) else 0
                                                
                semi_sup_train(current_learning_rate, current_sup_cost_weight, current_gen_cost_weight, current_push_fast_slow, current_sup_shuffle_indices, current_gen_shuffle_indices, current_batch_index)
                
                if display_on:
                    window_shift()
                    
                    #if it's been more than display_period seconds since the last slow gibbs samples were displayed
                    if (time.time() - last_display_time) > display_period:
                        #display the slow gibbs samples
                        self.gen_dbm_binary.display_sample_info(inv_temp_states_window, slow_gibbs_layer_samples, particle_display, display_slow_samples_grid_dims, display_gen_vis_input_dims)
                        
                        last_display_time = time.time()
                
                particle_update_count += 1
                param_update_count += 1
                                
                #print('\t\tEpoch completion percentage is {0:06.2f}%.'.format(100 * ((current_batch_index + 1) / float(num_batches))), end='\r')
                        
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            label_accuracy = label_accuracy_observe()
            sparsity_mean, sparsity_std = sparsity_observe()
            
            print('\n\tTraining epoch {0} completed in {1:.2f} seconds!\n\t\tTotal number of parameter updates is {2}.\n\t\tPredicted label accuracy is {3}.\n\t\tSparsity mean is {4}.\n\t\tSparsity standard deviation is {5}.'.format((epoch_count + 1), epoch_duration, param_update_count, label_accuracy, sparsity_mean, sparsity_std))
   
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        print('Training on {} epochs completed in {} seconds!'.format(num_epochs, training_duration))
    
def column_select(matrix, column_indices):
    
    return matrix[T.arange(column_indices.shape[0]), column_indices]

def sigmoid(matrix):
    
    return (1 / (1 + np.exp(-matrix)))