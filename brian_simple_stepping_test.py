'''
This script shows that I can make one population of neurons step up at different times in the trial
by varying the strength of the noisy input into the network. This works for even small groups of
neurons of 10
'''


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from brian2 import *
from brian2tools import *
prefs.codegen.target = 'numpy'  # this switches off fancy code generation for now



#start_scope() # This tells Brian to only look at the model defined under this line (in case you have more models)

# Cm = 200*pF
# E_L = -70*mV
# I= 1.5*nA
#
# eqs = '''
# dVm/dt = (I +g_L*(E_L - Vm))/Cm : volt
# g_L : siemens
# '''
#
# group = NeuronGroup(20,eqs)
# group.Vm = E_L
# group.g_L = '10*nS + rand()*5*nS'

num_neuron_E = 20 # number of excitatory neurons
tau = 10*ms # membrane voltage time constant

# input stimulus is nonhomogenous Poisson process switching between two frequencies of firing
#stimulus_tile = 35.0+np.array([0., 1., 0., 2., 0., 3., 0., 4., 0., 5., 0., 6., 0., 7., 0., 8., 0.]) # input noise frequency blocks
stimulus_tile = np.array([35.0])
block_size = 1000.
print('stim tile: ',stimulus_tile)
stimulus = TimedArray(stimulus_tile*Hz, dt=block_size*ms)
total_sim_time = len(stimulus_tile)*block_size # total length of time to run simulation
#P = PoissonGroup(1, rates='stimulus(t)')


# population of input noise
population_input = NeuronGroup(num_neuron_E, 'rates = stimulus(t) : Hz', threshold='rand()<rates*dt')
# population of excitatory neurons
population_E =  NeuronGroup(num_neuron_E, 'dVm/dt = (-70*mV-Vm) / tau : volt',threshold='Vm > -50*mV',reset='Vm = -70*mV',
                            method='euler')
population_E.Vm = -70*mV


# Define subgroups in excitatory population
half_pop_excit = int(num_neuron_E/2)
excite_pop_connected = population_E[:half_pop_excit]
excite_pop_unconnected = population_E[half_pop_excit:]


# connect input and excitatory with synapses
syn_input = Synapses(population_input,population_E,'w : volt',on_pre='Vm_post += w') # spikes to put in, which group of neurons, synapse model (direct change in postsynapstic voltage potential), define when to do something when presynaptic synapse spikes
syn_input.connect() # how to connect synapses (lots of ways to do this)
syn_input.w = '1*mV + rand()*1*mV'
syn_input.delay = '1*ms + rand()*5*ms'


# connect one subgroup of neurons to each other
syn_excit_conn = Synapses(excite_pop_connected,excite_pop_connected,'w : volt',on_pre='Vm_post += w') # spikes to put in, which group of neurons, synapse model (direct change in postsynapstic voltage potential), define when to do something when presynaptic synapse spikes
syn_excit_conn.connect() # how to connect synapses (lots of ways to do this)
syn_excit_conn.w = '5*mV + rand()*3*mV'
syn_excit_conn.delay = '1*ms + rand()*2*ms'



# monitor spikes and membrane potentials
spikes_input = SpikeMonitor(population_input)
spikes_E = SpikeMonitor(population_E)
#monitor_input = StateMonitor(population_input, 'Vm', record=True)
#monitor_pop_E = StateMonitor(population_E, 'Vm', record=True)
run(total_sim_time*ms, report='text')




# brian_plot(spikes_input)
# for split in range(int(total_sim_time/block_size)):
#     plt.axvline((block_size+1)*split,color='r',linewidth=0.5)
# plt.show()
# plt.close()

brian_plot(spikes_E)
for split in range(int(total_sim_time/block_size)):
    plt.axvline((block_size+1)*split,color='r',linewidth=0.5)
plt.show()
plt.close()
