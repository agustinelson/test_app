import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scienceplots
plt.style.use(['science', 'no-latex'])

WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3

def variational_circuit(params,hamiltonian):
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires = [0,1]))
def optimize_circuit(hamiltonian):
    hamiltonian = np.array(hamiltonian, requires_grad = False)
    hamiltonian = np.array(hamiltonian,float).reshape((2 ** WIRES), (2 ** WIRES))

    ### WRITE YOUR CODE BELOW THIS LINE
    ### Solution Template
    dev = qml.device("default.qubit", wires = WIRES)# Initialize the device.

    w = qml.numpy.random.rand(NUM_PARAMETERS)
#    qnod = qml.QNode(variational_circuit(w,hamiltonian),dev)
    @qml.qnode(dev)
    def cost(wei):
        return variational_circuit(wei,hamiltonian)

# Write your code to minimize the circuit
    opt=qml.GradientDescentOptimizer(stepsize=0.1)
    #opt=qml.AdamOptimizer(stepsize=0.2, beta1=0.9, beta2=0.99, eps=1e-08)
    #opt=qml.MomentumOptimizer(stepsize=0.1, momentum=0.9)
    #opt=qml.QNGOptimizer(stepsize=0.01, approx='block-diag', lam=1)

# set the number of steps
    steps = 100

# set the initial parameter values
    params = w

    for i in tqdm(range(steps)):
        # update the circuit parameters
        params = opt.step(cost, params)

        if (cost(params) - 0.61745341) < 1e-9:
          print('Step_size {} : Cost{:.8f}'.format(i, cost(params)))
          break


in1=np.array([0.863327072347624,0.0167108057202516,0.07991447085492759,0.0854049026262154,
              0.0167108057202516,0.8237963773906136,-0.07695947154193797,0.03131548733285282,
              0.07991447085492759,-0.07695947154193795,0.8355417021014687,-0.11345916130631205,
              0.08540490262621539,0.03131548733285283,-0.11345916130631205,0.758156886827099])
#Expected output: 0.61745341
in2=np.array([0.32158897156285354,-0.20689268438270836,0.12366748295758379,-0.11737425017261123,
              -0.20689268438270836,0.7747346055276305,-0.05159966365446514,0.08215539696259792,
              0.12366748295758379,-0.05159966365446514,0.5769050487087416,0.3853362904758938,
              -0.11737425017261123,0.08215539696259792,0.3853362904758938,0.3986256655167206])
#Expected output: 0.00246488


#Run test cases
optimize_circuit(in1)

#optimize_circuit(in2)

ChatGPT
Sure, here's the modified code using Streamlit to display the graphs:

python
Copy code
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import streamlit as st

WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3

def variational_circuit(params, hamiltonian):
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires=[0, 1]))

def optimize_circuit(hamiltonian):
    hamiltonian = np.array(hamiltonian, requires_grad=False)
    hamiltonian = np.array(hamiltonian, float).reshape((2 ** WIRES, 2 ** WIRES))

    dev = qml.device("default.qubit", wires=WIRES)  # Initialize the device.

    w = qml.numpy.random.rand(NUM_PARAMETERS)

    @qml.qnode(dev)
    def cost(wei):
        return variational_circuit(wei, hamiltonian)

    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    steps = 3000
    params = w
    costs = []

    for i in tqdm(range(steps)):
        params = opt.step(cost, params)
        costs.append(cost(params))

        if (cost(params) - 0.61745341) < 1e-9:
            print('Step_size {} : Cost{:.8f}'.format(i, cost(params)))
            break

    return costs

in1 = np.array([0.863327072347624, 0.0167108057202516, 0.07991447085492759, 0.0854049026262154,
                0.0167108057202516, 0.8237963773906136, -0.07695947154193797, 0.03131548733285282,
                0.07991447085492759, -0.07695947154193795, 0.8355417021014687, -0.11345916130631205,
                0.08540490262621539, 0.03131548733285283, -0.11345916130631205, 0.758156886827099])

in2 = np.array([0.32158897156285354, -0.20689268438270836, 0.12366748295758379, -0.11737425017261123,
                -0.20689268438270836, 0.7747346055276305, -0.05159966365446514, 0.08215539696259792,
                0.12366748295758379, -0.05159966365446514, 0.5769050487087416, 0.3853362904758938,
                -0.11737425017261123, 0.08215539696259792, 0.3853362904758938, 0.3986256655167206])

# Run test cases
costs1 = optimize_circuit(in1)
costs2 = optimize_circuit(in2)

# Streamlit app
st.title('Variational Quantum Circuit Optimization')

st.header('Test Case 1')
st.line_chart(costs1)

st.header('Test Case 2')
st.line_chart(costs2)
