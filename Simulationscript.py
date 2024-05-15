#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:20:37 2023

@author: janice
"""
import numpy as np
import time
import netsquid as ns
import netsquid.components.instructions as instr
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.component import Message
from netsquid.components import QuantumChannel, ClassicalChannel
from netsquid.nodes.connections import Connection
from netsquid.components.qprogram import QuantumProgram
from netsquid.protocols import NodeProtocol
from netsquid.nodes.network import Network
from netsquid.components.models import FibreDelayModel, FibreLossModel
import random
from ion_trap_local import IonTrap
import yaml
from yaml.loader import SafeLoader
from instructions_local import XX
from argparse import ArgumentParser
import ast

steady_param_yaml = "/home/timalbers/CODE/Measurement-Only-BQC/steady_params.yaml" # Path to yaml file containing the paramters that are not varied over

with open(steady_param_yaml) as f: #find parameters as stored in the yaml file
    steady_params = yaml.load(f, Loader=SafeLoader)

def create_ClientProcessor(I):
    darkcount_noise_model = DepolarNoiseModel(depolar_rate=steady_params["darkcount_prob"], time_independent=True)
    phys_instrs = [PhysicalInstruction(instr.INSTR_UNITARY, duration=2),
                   PhysicalInstruction(instr.IMeasureFaulty(name='faulty client measurement',
                                                            p0=steady_params['PBS_crosstalk'],
                                                            p1=steady_params['PBS_crosstalk']),
                                        duration=1,
                                       quantum_noise_model=darkcount_noise_model),
                    PhysicalInstruction(instr.INSTR_DISCARD, duration=2)]
    qproc_c = QuantumProcessor("clientqproc", num_positions=I, phys_instructions=phys_instrs, replace=False)
    return qproc_c

def waveplate(retardance, retardation_deviation, angle_fast_axis, fast_axis_tilt):
    """ Jones matrix of a waveplate with specific retardance (pi/2 for QWP, pi for HWP) with errors

    parameters
    ----------
    retardance: float
        relative phase retardance by waveplate (pi/2 for QWP, pi for HWP)
    retardation_deviation: float
        error in retardance
    angle_fast_axis: float
        angle of the fast axis of the waveplate wrt the x axis
    fast_axis_tilt: float
        error in fast axis angle

    Returns
    --------
    :class:'~netsquid.qubits.operators.Operator'
        An operator acting as a wave plate with errors
    """
    deltap = retardance + retardation_deviation
    xp = angle_fast_axis + fast_axis_tilt
    return ns.qubits.operators.Operator("WP", [[(np.exp(deltap*1j/2)*np.cos(xp)**2 + np.exp(-deltap*1j/2)*np.sin(xp)**2), (np.exp(deltap*1j/2)-np.exp(-deltap*1j/2))*np.cos(xp)*np.sin(xp)], 
                                                [(np.exp(deltap*1j/2) - np.exp(-deltap*1j/2))*np.cos(xp)*np.sin(xp), np.exp(deltap*1j/2)*np.sin(xp)**2 + np.exp(-deltap*1j/2)*np.cos(xp)**2]])


def meas_angle_delta(mbqc_bases_i, i, G):
    """ Function to calculate the angle to be send from the Client to the Server in which it will measure the qubits.
    """
    # Generate random bit and keep track of it
    r_i = random.randrange(0, 2)
    r.append(r_i)
    # Get measurement basis and measurement result for rsp of qubit i
    m_i = m[i]
    theta_i = theta[i]
    thetap_i = theta_i + m_i*np.pi
    # Get measurement basis for MBQC protocol and angle adapatations based on propagation of pauli corrections
    phi_i = mbqc_bases_i
    if i>0:
        sx_i = s[i-1]
    else:
        sx_i = 0
    # Get a list of indices of the neighbors of previous qubit = Z dependency
    def find_neighbors(node_to_check, graph):
        neighbors = []
        for edge in graph:
            if node_to_check in edge:
                neighbor_node = edge[1] if edge[0] == node_to_check else edge[0]
                if neighbor_node < i - 1:
                    neighbors.append(neighbor_node)
        return neighbors
    SZ_indices = find_neighbors(i-1, G)
    sz_i = np.sum([s[i] for i in SZ_indices])
    phip_i = (-1)**sx_i*phi_i + sz_i*np.pi
    return (phip_i + thetap_i + r_i*np.pi)

class PrepareGraph(QuantumProgram):
    """ Prepares graph state from initialized qubits in the server memory by applying CZ to the edges defined
    by G. """
    def program(self, G, I):
        for g in G:
            #Prepare graph state to qubits in the memory: apply CZ gate with native gates
            q1, q2 = g
            self.apply(instr.INSTR_ROT_Y, q1, angle=np.pi/2)
            self.apply(instr.INSTR_ROT_Y, q2, angle=np.pi/2)
            self.apply(instr.INSTR_ROT_X, q2, angle=np.pi)
            self.apply(XX(num_positions=2), [q1, q2])
            self.apply(instr.INSTR_ROT_Y, q2, angle=-np.pi/2)
            self.apply(instr.INSTR_ROT_X, q1, angle=-np.pi/2)
            self.apply(instr.INSTR_ROT_Y, q1, angle=-np.pi/2)
            self.apply(instr.INSTR_ROT_Z, q2, angle=np.pi/2)
            yield self.run()

class IonTrapEmitProg(QuantumProgram):
    def program(self, memory_position, emission_position):
        self.apply(instr.INSTR_INIT, memory_position)
        self.apply(instr.INSTR_EMIT, [memory_position, emission_position], header="test")
        yield self.run()


class Measure_Client3wp(QuantumProgram):
    """ Program for client to measure incoming photonic qubit in basis
    along the equator of the Bloch sphere using rotations
    by half and quarter wave plates. Saves measurement outcomes in the list m.

    parameter
    ---------
    i: int
    index of the qubit to be measured

    """
    def program(self, i, retardation_deviation, fast_axis_tilt):
        # Remotely prepare the qubits in |+_theta>, by measuring along a +/-_theta basis
        # Choose a randomly and independenly chosen measurement angle theta
        k = random.randrange(0, 8)
        theta_i = k*np.pi/4

        # Set fast axes of waveplates corresponding to a measurement in the theta basis
        a = -np.pi/2
        b = theta_i
        c = 0
        phi1 = np.pi/4 + a/2
        phi2 = np.pi/4 + (a+b)/2
        phi3 = -np.pi/4 + (a+b-c)/4

        # Apply the 'waveplate' operators (one HWP followed by to QWPs)
        self.apply(instr.INSTR_UNITARY, i, operator=waveplate(np.pi, retardation_deviation, phi3, fast_axis_tilt))
        self.apply(instr.INSTR_UNITARY, i, operator=waveplate(np.pi/2, retardation_deviation, phi2, fast_axis_tilt))
        self.apply(instr.INSTR_UNITARY, i, operator=waveplate(np.pi/2, retardation_deviation, phi1, fast_axis_tilt))

        theta.append(theta_i) # Keep track of the angles used (the client needs this information to calculate the angles to send to the server)
        self.apply(instr.IMeasureFaulty(name='faulty client measurement',
                                                            p0=steady_params['PBS_crosstalk'],
                                                            p1=steady_params['PBS_crosstalk']), i, output_key=f"m{i}")
        self.apply(instr.INSTR_DISCARD, i)
        yield self.run()
        m.append(self.output[f"m{i}"][0]) # Keep track of the measurement outcomes (the client needs this information to calculate the angles to send to the server)

class Measure_Server(QuantumProgram):
    """ Program for the server to perform the measurements in the +/-_delta basis (as instructed by the client).
    It is possible to adjust this such that the last qubit is measured in the standard basis (or any other basis)
    rather than in a basis along the equator of the Bloch sphere, but this is currently set to False.

    Parameters
    ----------
    i : (int)
        Qubit to be measured
    delta : (float)
        Measurement angle (angle wrt the positive x-axis on the equator of the Bloch sphere) provided by the client

    Returns
    -------
    b : (int)
        Measurement outcome

    """
    def program(self, i, delta, I):
        lastqubit = False #i==I-1
        if lastqubit:
            # Measure last qubit in standard basis
            self.apply(instr.INSTR_ROT_Z, i, angle=delta)
            faulty_measurement_instruction = instr.IMeasureFaulty("faulty_z_measurement_ion_trap",
                                                            p0=steady_params["prob_error_0"],
                                                            p1=steady_params["prob_error_1"])
            self.apply(faulty_measurement_instruction, i, output_key=f"b{i}")
            yield self.run()
            return self.output[f"b{i}"]
        else:
            # If not last qubit, measure in +/-_delta
            self.apply(instr.INSTR_ROT_Z, i, angle=delta)
            self.apply(instr.INSTR_ROT_Y, i, angle=np.pi/2)
            self.apply(instr.INSTR_ROT_X, i, angle=np.pi)
            faulty_measurement_instruction = instr.IMeasureFaulty("faulty_z_measurement_ion_trap",
                                                            p0=steady_params["prob_error_0"],
                                                            p1=steady_params["prob_error_1"])
            self.apply(faulty_measurement_instruction, i, output_key=f"b{i}")
            yield self.run()
            return self.output[f"b{i}"]

class ServerProtocol(NodeProtocol):
    """ Protocol for server to perform a round of rVBQC with the client.
    """
    def __init__(self, node):
        self._deltas = []
        self._qubit_arrived_confirmation = False
        self.opt_params = None
        self.I = None
        self._server_rsp_counter = 0
        self._attempt_counter = 0
        super().__init__(node=node, name="Server_Protocol")

    def start(self, opt_params):
        self.opt_params=opt_params
        super().start()

    def _handle_angle_income(self, msg):
        # When the client sends a measurement angle, we add it to the list of delta's, such that we can measure in this basis later
        delta = msg.items[0]
        self._deltas.append(delta)

    def run(self):
        #Receive graph state description
        description_arrived = yield self.await_port_input(self.node.ports["cin_s"])
        if description_arrived:
            mes = self.node.ports["cin_s"].rx_input()
            self.I = mes.items[0]
            G = mes.items[1]
        self.node.ports["cin_s"].bind_input_handler(self._handle_angle_income)
        graph_prog = PrepareGraph()
        ent_prog = IonTrapEmitProg()
        arrived_at_attempts = [] #list to keep track of at which attempt the qubits are initialized in the memory, to check cutoff
        while self._server_rsp_counter < self.I:
            unused_mem_positions = self.node.subcomponents["ion_trap_quantum_communication_device"].unused_positions

            self.node.subcomponents["ion_trap_quantum_communication_device"].execute_program(ent_prog, memory_position=unused_mem_positions[0], emission_position=self.I) #prepare EPR pair
            yield self.await_program(self.node.subcomponents["ion_trap_quantum_communication_device"]) # Wait for the program to finish running
            yield self.await_timer(1)
            self.node.ports["IDout_s"].tx_output([str(self._attempt_counter), str(unused_mem_positions[0])])
            self._attempt_counter += 1

            # Wait to see if the client received the qubit
            expr = yield self.await_port_input(self.node.ports["IDin_s"])
            if expr.value:
                ID = self.node.ports["IDin_s"].rx_input() 
                attempt_time = 500000 # Time it takes to complete one attempt of sending an ion-entangled photon to the client
                attempts_before_cutoff = int((self.opt_params['coherence_time']/2)/attempt_time) # How many attempts we allow before cutoff (roughly half a coherence time)
                if attempts_before_cutoff == 0:
                    bool_cutoff = False
                else:
                    bool_cutoff = any(self._attempt_counter - np.array(arrived_at_attempts) > attempts_before_cutoff) # If any qubit has been in the memory for too many attempts, bool_cutoff=True
                bool_received = "received "+str(self._attempt_counter-1) in ID.items[0] # Bool to reflect if a qubit was received (client will communicate this clasically)

                if bool_cutoff and bool_received :
                    # If the qubit was received, but there is also another qubit in the memory for too long, we add the new qubit, but throw away the too-old qubit
                    k = int(np.where((self._attempt_counter - np.array(arrived_at_attempts)) > attempts_before_cutoff)[0][0]) # Which qubit needs to be cutoff
                    self.node.subcomponents["ion_trap_quantum_communication_device"].discard(k) # Discard old qubit
                    arrived_at_attempts.pop(k) # Remove old, discarded qubit from the list that keeps track of qubit ages
                    arrived_at_attempts.append(self._attempt_counter-1) # Add new, received qubit to the list that keeps track of qubit ages
                    self.node.ports["IDout_s"].tx_output(f"DISCARD {k}") # Communicate to client which qubit was discarded

                if bool_cutoff and not bool_received:
                    # Qubit was not received, but another qubit surpassed the cutoff time and needs to be discarded 
                    k = int(np.where((self._attempt_counter - np.array(arrived_at_attempts)) > attempts_before_cutoff)[0][0])
                    self.node.subcomponents["ion_trap_quantum_communication_device"].discard(k)
                    arrived_at_attempts.pop(k)
                    self.node.ports["IDout_s"].tx_output(f"DISCARD {k}")
                    self.node.subcomponents["ion_trap_quantum_communication_device"].discard(self._server_rsp_counter) # Discard ion of which photon got lost 
                    self._server_rsp_counter -= 1 # Number of completed qubits goes down by one

                if not bool_cutoff and not bool_received:
                    # Qubit was not received, none of the qubits in the memory have gone over cutoff
                    self.node.subcomponents["ion_trap_quantum_communication_device"].discard(self._server_rsp_counter)

                if not bool_cutoff and bool_received:
                    # Qubit was received, no qubits have gone over cutoff
                    arrived_at_attempts.append(self._attempt_counter-1)
                    self._server_rsp_counter += 1 # Number of completed qubits goes up by one

        # All qubits have arrived at the client within cutoff, so start graph forming
        self.node.ports["IDout_s"].tx_output("DONE") # Communicate to the client
        attempts.append(self._attempt_counter)
        self.node.subcomponents["ion_trap_quantum_communication_device"].execute_program(graph_prog, G=G, I=self.I) # Make remaining qubits in memory into a graph state
        for j in range(self.I):
            angle_arrived = yield self.await_port_input(self.node.ports["cin_s"]) # Receive measurement angle delta from client
            if angle_arrived:
                if self.node.subcomponents["ion_trap_quantum_communication_device"].busy:
                    yield self.await_program(self.node.subcomponents["ion_trap_quantum_communication_device"]) # Wait for a bit if the server is still busy
                delta_j = self._deltas[j]
                meas_prog = Measure_Server()
                self.node.subcomponents["ion_trap_quantum_communication_device"].execute_program(meas_prog, i=j, delta=delta_j, I=self.I) # Measure the qubit
                yield self.await_program(self.node.subcomponents["ion_trap_quantum_communication_device"])
                b_j = meas_prog.output[f"b{j}"]
                self.node.ports["cout_s"].tx_output(Message(items=b_j, header=f"b{j}")) # Send measurement outcome back to the client

class ClientProtocol(NodeProtocol):
    """ Protocol for client to perform a round of rVBQC on the server.
    """
    def __init__(self, node):
        self.I = None
        self.G = None
        self._mbqc_bases = None
        self._rsp_complete = False
        self._client_rsp_counter = 0
        super().__init__(node=node, name=f"ClientProtocol running on node {node}")

    def start(self, I, G, mbqc_bases):
        self.I = I
        self.G = G
        self._mbqc_bases = mbqc_bases
        super().start()

    def _handle_meas_income(self, msg):
        #decode and safe the measurment outcome that the Server sends back
        b = msg.items[0]
        r_i = r[len(s)]
        s_i = (b + r_i) % 2
        s.append(s_i)

    def _handle_ID_income(self, msg):
        meas_prog = Measure_Client3wp()
        ID = msg.items
        if "DISCARD" in ID[0]: # Server tells client that a qubit has reached the cutoff, remove measurement
            to_be_removed = int(ID[0].replace("DISCARD ", ""))
            self._client_rsp_counter -= 1
            m.pop(to_be_removed)
            theta.pop(to_be_removed)

        elif "DONE" in ID: # Server tells client that the RSP phase of the protocol is complete
            self._rsp_complete = True

        else: # If not done or discard, a qbitID came in, check if a qubit came into the memory
            if self.node.subcomponents["clientqproc"].used_positions: # There is a qubit: start measurement program and declare success
                self._client_rsp_counter += 1
                retardation_deviation = steady_params['retardation_deviation']
                fast_axis_tilt=steady_params['fast_axis_tilt']
                self.node.subcomponents["clientqproc"].execute_program(meas_prog, i=0, retardation_deviation=retardation_deviation, fast_axis_tilt=fast_axis_tilt)
                self.node.ports["IDout_c"].tx_output("received "+str(ID[0]))
            else: # Photon got lost, communicate this to server
                self.node.ports["IDout_c"].tx_output("did not receive "+str(ID[0]))

    def run(self):
        # Send graph state description to Server (number of qubits and list of edges)
        message = [self.I, self.G]
        self.node.ports["cout_c"].tx_output(message)
        self.node.ports["cin_c"].bind_input_handler(self._handle_meas_income)
        self.node.ports["IDin_c"].bind_input_handler(self._handle_ID_income)
        self.node.ports["qin_c"].forward_input(self.node.subcomponents["clientqproc"].ports["qin0"])
        while not self._rsp_complete: # Things are handeled by input handlers untill RSP is complete
            assert not self._rsp_complete
            yield self.await_port_input(self.node.ports["IDin_c"])
        for j in range(self.I): # For all qubits:
            # Calculate (updated) measurement angle
            delta_j = meas_angle_delta(self._mbqc_bases[j], j, self.G)
            # Send updated measurement angle to Server
            mes = Message(items=delta_j, header=f"delta{j}")
            self.node.ports["cout_c"].tx_output(mes)
            #wait till measruement outcome have been send back (income will be handled by _handle_meas_income)
            yield self.await_port_input(self.node.ports["cin_c"])


def networksetupproc(I, opt_params):
    """
    Creates setup with a trapped-ion quantum server and a photonic client, with one quantum channel and two classical channels
    (one classical channel is just used for qubit IDs, this was just easier in coding).
    Parameters are taken from yaml files (baseline.yaml and steady_params.yaml)
    """
    network = Network("BQC network")

    #Create nodes and add to network
    server = ns.nodes.node.Node("server", port_names=["qout_s", "cin_s", "cout_s", "IDout_s", "IDin_s"])
    client = ns.nodes.node.Node("client", port_names=["qin_c", "cout_c", "cin_c", "IDout_c", "IDin_c"])
    network.add_nodes([server, client])

    #Add quantum processor to nodes
    #print("test ", params["coherence_time"])
    qproc_s = IonTrap(num_positions=I, coherence_time=opt_params["coherence_time"],
                      prob_error_0=steady_params["prob_error_0"], prob_error_1=steady_params["prob_error_1"],
                      init_depolar_prob=steady_params["init_depolar_prob"], rot_x_depolar_prob=opt_params["single_qubit_depolar_prob"],
                      rot_y_depolar_prob=opt_params["single_qubit_depolar_prob"], rot_z_depolar_prob=opt_params["single_qubit_depolar_prob"],
                      ms_depolar_prob=opt_params["ms_depolar_prob"], emission_fidelity=opt_params["emission_fidelity"],
                      collection_efficiency=steady_params["collection_efficiency"], emission_duration=steady_params["emission_duration"],
                      measurement_duration=steady_params["measurement_duration"], initialization_duration=steady_params["initialization_duration"],
                      single_qubit_rotation_duration=steady_params["single_qubit_rotation_duration"], ms_pi_over_2_duration=steady_params["ms_pi_over_2_duration"],
                      ms_optimization_angle=steady_params["ms_optimization_angle"])
    server.add_subcomponent(qproc_s)
    qproc_c = create_ClientProcessor(I)
    client.add_subcomponent(qproc_c)

    if args.channel_length:
        fibre_length = args.channel_length
    else:
        fibre_length = steady_params['channel_length']
    loss_length = steady_params['p_loss_length']
    #Add quantum connection with quantum channel & connect the ports
    qcon = Connection("Qconnection")
    network.add_connection(server, client, connection=qcon, label="Qconnection", port_name_node1="qout_s", 
                           port_name_node2="qin_c")
    loss_model = FibreLossModel(p_loss_init=opt_params['p_loss_init'], p_loss_length=loss_length)
    qchan = QuantumChannel("qchan", length=fibre_length, models={"delay_model": FibreDelayModel(), 'quantum_loss_model': loss_model})
    server.subcomponents['ion_trap_quantum_communication_device'].ports["qout"].forward_output(server.ports["qout_s"])
    qcon.add_subcomponent(qchan, forward_input=[("A", "send")], forward_output=[("B", "recv")])

    ccon1 = Connection("Cconnection1")
    network.add_connection(client, server, connection=ccon1, label="Connection1", port_name_node1="cout_c", 
                           port_name_node2="cin_s")
    cchan1 = ClassicalChannel("cchan1", length=fibre_length, models={"delay_model": FibreDelayModel()})
    ccon1.add_subcomponent(cchan1, forward_input=[("A", "send")], forward_output=[("B", "recv")])

    ccon2 = Connection("Cconnection2")
    network.add_connection(server, client, connection=ccon2, label="Connection2", port_name_node1="cout_s", 
                           port_name_node2="cin_c")
    cchan2 = ClassicalChannel("cchan2", length=fibre_length, models={"delay_model": FibreDelayModel()})
    ccon2.add_subcomponent(cchan2, forward_input=[("A", "send")], forward_output=[("B", "recv")])

    IDconsc = Connection("IDConnectionsc")
    network.add_connection(server, client, connection=IDconsc, label="IDConnectionsc", port_name_node1="IDout_s", 
                           port_name_node2="IDin_c")
    IDchansc = ClassicalChannel("IDchansc", length=fibre_length, models={"delay_model": FibreDelayModel()})
    IDconsc.add_subcomponent(IDchansc, forward_input=[("A", "send")], forward_output=[("B", "recv")])

    IDconcs = Connection("IDConnectionsc")
    network.add_connection(client, server, connection=IDconcs, label="IDConnectioncs", port_name_node1="IDout_c", 
                           port_name_node2="IDin_s")
    IDchancs = ClassicalChannel("IDchancs", length=fibre_length, models={"delay_model": FibreDelayModel()})
    IDconcs.add_subcomponent(IDchancs, forward_input=[("A", "send")], forward_output=[("B", "recv")])

    return network

#keep track of...
m = [] #measurement results of RSP measurements by Client
theta = [] #measurement angles used by the client for RSP
r = [] #random bit used to one-time-pad the angles that the Client sends to the Server for computation
s = [] #decoded measurment MBQC outcomes (= b(j)+r(j) with b(j) meas results as send by Server)
attempts = []
def run_experiment(I, G, mbqc_bases, opt_params, run_amount):
    resses= []
    for i in range(run_amount):
        # Clear everything 
        ns.sim_reset()
        m.clear()
        theta.clear()
        r.clear()
        s.clear()
        attempts.clear()
        # Set up the network and processors
        network = networksetupproc(I=I, opt_params=opt_params)
        server = network.get_node("server")
        client = network.get_node("client")
        server.subcomponents["ion_trap_quantum_communication_device"].reset()
        client.subcomponents["clientqproc"].reset()
        # Start protocols
        ClientProtocol(client).start(I, G, mbqc_bases)
        ServerProtocol(server).start(opt_params=opt_params)
        ns.sim_run()
        resses.append(s[-1]) # Decoded outcome of the final measurement is the output of the computation
    result = sum(resses)/len(resses) # Average of per-iteration outcomes
    print(f"--------------------\nResults for {fibre_length} km:\n{result}\n--------------------")
    return result

def main():
    t1 = time.time()
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument('--opt_params', type=str, help="Input dictionary as a string", required=False)
    parser.add_argument('--channel_length', type=float, help="Length of the quantum channel in km", required=False)
    parser.add_argument('--run_amount', type=int, help="Number of iterations the simulation is repeater for (=number of test rounds)", required=True)
    parser.add_argument('--mbqc_bases', nargs='+', type=float)
    args = parser.parse_args()
    if args.opt_params:
        try:
            # Safely evaluate the string representation of the dictionary
            args.opt_params = ast.literal_eval(args.opt_params)
        except ValueError:
            print("Error: Unable to parse the input dictionary.")
            exit(1)
    else:
        print('No input dict was provided, so take baseline as default')
        baseline_yaml = "/home/timalbers/CODE/Measurement-Only-BQC/baseline.yaml"
        with open(baseline_yaml) as f:
            default_dict = yaml.load(f, Loader=SafeLoader)
        args.opt_params = default_dict
    I = 2
    G = [[0,1]]
    run_experiment(I=I, G=G, mbqc_bases=args.mbqc_bases, opt_params=args.opt_params, run_amount=args.run_amount)
    t2 = time.time()
    print("Runs: ", args.run_amount)
    print(convert_seconds(t2-t1))

def convert_seconds(total_seconds):
    days = total_seconds // (24 * 3600)  # Calculate the number of days
    total_seconds %= (24 * 3600)         # Update total_seconds to the remainder
    hours = total_seconds // 3600        # Calculate the number of hours
    total_seconds %= 3600                # Update total_seconds to the remainder
    minutes = total_seconds // 60        # Calculate the number of minutes
    seconds = total_seconds % 60         # Calculate the number of remaining seconds
    return f"Runtime: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {round(seconds, 2)} seconds"

if __name__ == "__main__":
    main()

