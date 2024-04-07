import random
from abc import abstractmethod
import numpy as np
from scipy.linalg import expm
import netsquid.qubits.qubitapi as qapi
from netsquid.components.instructions import Instruction
from netsquid.qubits import operators as ops
from netsquid.qubits.ketstates import BellIndex

"""Locally held copy of NetSquid TrappedIons snippet, to include explicit two-qubit XX (bichromatic) gate"""


class IonTrapMultiQubitGate(Instruction):  # TODO: make this an ABC?
    """Base class for multi-qubit gates in ion trap.

    Allows for construction of S-matrix.
    Child classes only need to provide a construct_operator method.

    Parameters
    ----------
    name : str
        Name of instruction for identification purposes.
    num_positions : int, optional
        Number of positions_of_connections the gate acts on. Only used if ``operator`` is None.

    """

    def __init__(self, num_positions):
        self._num_positions = num_positions
        self._name = None
        self._theta = None
        self._phi = None

    @property
    def theta(self):
        """float: optimization angle"""
        return self._theta

    @property
    def phi(self):
        """float: rotation axis angle"""
        return self._phi

    @property
    def name(self):
        """str: instruction name."""
        return self._name

    @property
    def num_positions(self):
        """int: number of targeted memory positions_of_connections. If -1, number is unrestricted."""
        return self._num_positions

    @abstractmethod
    def construct_operator(self, phi=np.pi / 4, theta=np.pi / 2):
        """Construct operator which is applied by the gate.
        Used by execute method.

        """

        self._operator = None

    def construct_s(self, phi=np.pi / 4):
        """Constructs S-matrix that is needed for the multi-qubit gates encountered in ion traps.
            Used by construct_operator method.

        Parameters
        ----------
        phi: float, optional
        Angle that characterizes the S-matrix. Can be chosen per application.

        """

        def create_op_i(op, i, N):
            return np.kron(np.kron(np.eye(2 ** (i - 1)), op), np.eye(2 ** (N - i)) if N > i else 1)

        N = self._num_positions
        X, Y = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
        smatrix_parts = [np.cos(phi) * create_op_i(X, i, N) + np.sin(phi) * create_op_i(Y, i, N) for i in
                         range(1, N + 1)]
        self._smatrix = np.kron(X, X)#np.sum(smatrix_parts, axis=0)
        self._phi = phi

    def execute(self, quantum_memory, positions, phi=np.pi / 4, theta=np.pi / 2, **kwargs):
        """Execute instruction on a quantum memory.

        Parameters
        ----------
        quantum_memory : :obj:`~netsquid.components.qmemory.QuantumMemory`
            Quantum memory to execute instruction on.
        positions : list of int
            Memory positions_of_connections to do instruction on. Can be empty.
        phi : float, optional
            Angle that characterizes the S-matrix of the operation.
            Rotation is performed around axis cos(phi) X + sin(phi) Y.
        theta: float, optional
            Rotation angle.

        """

        if self._theta != theta or self._phi != phi:
            self.construct_operator(phi, theta)
        return quantum_memory.operate(self._operator, positions)

class IonTrapMultiQubitRotation(IonTrapMultiQubitGate):
    """Gate that performs a single-qubit rotation on all qubits within an ion trap, around and axis in the XY plane.

    Parameters
    ----------
    name : str
        Name of instruction for identification purposes.
    num_positions : int, optional
        Number of positions_of_connections the gate acts on. Only used if ``operator`` is None.

    """

    def __init__(self, num_positions):
        super().__init__(num_positions)
        self._name = 'multi_qubit_XY_rotation'

    def construct_operator(self, phi=np.pi / 4, theta=np.pi / 2):
        """Construct operator which is applied by the gate.
            Used by execute method.

        Parameters
        ----------
        phi : float, optional
            Angle that characterizes the S-matrix of the operation.
            Rotation is performed around axis cos(phi) X + sin(phi) Y.
        theta: float, optional
            Rotation angle.

        """
        self._theta = theta
        if phi != self._phi:
            self.construct_s(phi)
        self._operator = ops.Operator('many_qubit_rot', expm(-1j * theta / 4 * self._smatrix))

class XX(IonTrapMultiQubitGate):
    def __init__(self, num_positions, phi=np.pi / 4):
        super().__init__(num_positions)
        self._name = 'XX(phi)'
        self._phi = phi
    
    def construct_operator(self, phi=np.pi/4, theta=np.pi/2):
        matrix = np.array([[np.cos(self._phi), 0, 0, -1.j*np.sin(self._phi)], 
                                                   [0, np.cos(self._phi), -1.j*np.sin(self._phi), 0],
                                                   [0, -1.j*np.sin(self._phi), np.cos(self._phi), 0],
                                                   [-1.j*np.sin(self._phi), 0, 0, np.cos(self._phi)]])
        self._operator = ops.Operator('XX', matrix)
        
    def execute(self, quantum_memory, positions, phi=np.pi / 4, **kwargs):
        super().execute(quantum_memory=quantum_memory, positions=positions, phi=phi, **kwargs)
        
class IonTrapMStest(IonTrapMultiQubitGate):
    def __init__(self, num_positions, theta=np.pi / 2):
        super().__init__(num_positions)
        self._name = 'ms_gate_theta=' + str(theta / np.pi) + '_pi'
        self._theta = theta
    
    def construct_operator(self, phi=np.pi / 4, theta=np.pi / 2):
        self.construct_s(phi)
        print(expm(-1.j * theta / 2 * self._smatrix))
        self._operator = ops.Operator('Rxx', expm(-1.j * theta / 2 * self._smatrix))
        print(expm(-1.j * theta / 2 * self._smatrix))
    
    def execute(self, quantum_memory, positions, phi=np.pi / 4, **kwargs):
        print(expm(-1.j * self._theta / 2 * self._smatrix))
        print("THETA ", self._theta)
        super().execute(quantum_memory=quantum_memory, positions=positions, phi=phi, theta=self._theta, **kwargs)
        print(expm(-1.j * self._theta / 2 * self._smatrix))
        
class IonTrapMSGate(IonTrapMultiQubitGate):
    """Mølmer–Sørensen gate for ion traps at a specific optimization angle.

    Parameters
    ----------
    name : str
        Name of instruction for identification purposes.
    num_positions : int, optional
        Number of positions_of_connections the gate acts on. Only used if ``operator`` is None.
    theta : float, optional
        Angle for which the ion trap has been optimized.
    """

    def __init__(self, num_positions, theta=np.pi / 2):
        super().__init__(num_positions)
        self._name = 'ms_gate_theta=' + str(theta / np.pi) + '_pi'
        self._theta = theta

    def construct_operator(self, phi=np.pi / 4, theta=np.pi / 2):
        """Construct operator which is applied by the gate. Used by execute method.

        Parameters
        ----------
        phi : float, optional
            Angle that characterizes the S-matrix of the operation.

        """

        self.construct_s(phi)
        self._operator = ops.Operator('MS_gate',
                                      expm(-1j * theta / 2 * np.linalg.matrix_power(self._smatrix, 2)))

    def execute(self, quantum_memory, positions, phi=np.pi / 4, **kwargs):
        """Execute instruction on a quantum memory.

        Parameters
        ----------
        quantum_memory : :obj:`~netsquid.components.qmemory.QuantumMemory`
            Quantum memory to execute instruction on.
        positions : list of int
            Memory positions_of_connections to do instruction on. Can be empty.
        phi : float, optional
            Angle that characterizes the S-matrix of the operation.
            Rotation is performed around axis cos(phi) X + sin(phi) Y.

        """
        super().execute(quantum_memory=quantum_memory, positions=positions, phi=phi, theta=self._theta, **kwargs)


class IInitRandom(Instruction):
    """Instruction that initializes a qubit in a random state.

    """
    _standard_rotation_ops = [ops.Rx90, ops.Rx180, ops.Rx270, ops.Ry90, ops.Ry270, ops.I]

    @property
    def name(self):
        """str: instruction name."""
        return "init_random_qubit"

    @property
    def num_positions(self):
        """int: number of targeted memory positions_of_connections. If -1, number is unrestricted."""
        return -1

    def execute(self, quantum_memory, positions, standard_states=True):
        """Create random qubits.

        Parameters
        ----------
        quantum_memory : :obj:`~netsquid.components.qmemory.QuantumMemory`
            Quantum memory to execute instruction on.
        positions : list of int
            Memory positions_of_connections where random qubits are created.
        standard_states : bool, optional
            True for standard states (less computationally heavy),
            False for cominstr.INSTR_MEASUREplete randomness (non-uniform)

        """

        qubit = qapi.create_qubits(len(positions))
        for i in range(len(qubit)):
            if standard_states:
                operator = random.choice(self._standard_rotation_ops)
            else:
                theta = random.random() * np.pi * 2
                n1 = random.random()
                n2 = random.random()
                n3 = random.random()
                operator = ops.create_rotation_op(theta, (n1, n2, n3))
            qapi.operate(qubit[i], operator)
        quantum_memory.put(qubit, positions=positions, replace=True,
                           check_positions=True)


class IInitBell(Instruction):
    """Instruction that initializes a qubit.

    """

    @property
    def name(self):
        """str: instruction name."""
        return "init_bell_op"

    @property
    def num_positions(self):
        """int: number of targeted memory positions_of_connections. If -1, number is unrestricted."""
        return 2

    def execute(self, quantum_memory, positions, bell_index, **kwargs):
        """....

        Expects two positions_of_connections...

        Parameters
        ----------
        quantum_memory : :obj:`~netsquid.components.qmemory.QuantumMemory`
            Quantum memory to execute instruction on.
        positions : list of int
            Memory positions_of_connections to do instruction on.
        bell_index: :class:`netsquid.qubits.ketstates.BellIndex`
            Bell index of Bells tate to initialize.

        """
        if len(positions) != 2:
            raise ValueError("Bell state must be initialized over two qubits")

        bell_index = BellIndex(bell_index)  # raises ValueError if invalid Bell index

        # Initialize phi+
        q1, q2 = qapi.create_qubits(2)
        qapi.operate([q1], ops.H)
        qapi.operate([q1, q2], ops.CNOT)

        # Apply X and/or Z to turn it into desired Bell state
        if bell_index in [BellIndex.B10, BellIndex.B11]:
            qapi.operate([q1], ops.Z)
        if bell_index in [BellIndex.B01, BellIndex.B11]:
            qapi.operate([q2], ops.X)

        quantum_memory.put([q1, q2], positions=positions, replace=True,
                           check_positions=True)


# Defining usable (physical) instructions
INSTR_INIT_BELL = IInitBell()
INSTR_INIT_RANDOM = IInitRandom()
