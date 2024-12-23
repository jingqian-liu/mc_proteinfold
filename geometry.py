import numpy as np

def readstdgeometry(inputgeo):
    """
    Reads the standard geometry file and processes it into a structured NumPy array.

    Args:
        inputgeo (str): Path to the input geometry file.

    Returns:
        np.ndarray: Processed geometry data as a NumPy array.
    """
    # Define a mapping dictionary for atom types
    atom_mapping = {
        'CA': 0, 'C': 1, 'M': 2, 'F': 3, 'I': 4, 'L': 5, 'V': 6, 'W': 7, 'Y': 8,
        'A': 9, 'G': 10, 'T': 11, 'S': 12, 'Q': 13, 'N': 14, 'E': 15, 'D': 16,
        'H': 17, 'R': 18, 'K': 19, 'P': 20
    }

    # Read the geometry file
    with open(inputgeo, 'r') as f:
        lines = f.readlines()

    # Process each line
    stdgeo = []
    for line in lines:
        tokens = line.split()[:-1]  # Exclude the last empty token
        atom_type = tokens[0]
        if atom_type in atom_mapping:
            tokens.append(atom_mapping[atom_type])
            stdgeo.append(tokens)

    # Sort and clean up
    stdgeo.sort(key=lambda x: x[-1])  # Sort by the appended mapping value
    stdgeo = [[float(val) for val in row[1:4]] for row in stdgeo]  # Convert to float and exclude non-numerics

    return np.array(stdgeo)



import math

def nextCA(lastCA, llastCA, lastSC, phi):
    """
    Calculates the position of the next CA atom.

    Args:
        lastCA (np.ndarray): Coordinates of the last CA atom.
        llastCA (np.ndarray): Coordinates of the second-to-last CA atom.
        lastSC (np.ndarray): Coordinates of the last side-chain atom.
        phi (float): Phi torsion angle in radians.

    Returns:
        np.ndarray: Coordinates of the next CA atom.
    """
    # Calculate orthonormal vectors
    vector1 = normalize(np.cross(lastCA - llastCA, lastSC - lastCA))
    vector2 = normalize(np.cross(vector1, lastCA - llastCA))

    # Calculate next CA position
    tmp = math.cos(phi) * 3.8 * vector2 + math.sin(phi) * 3.8 * vector1

    # Handle special case where phi == 0
    if phi == 0:
        phi = 2 * (np.random.random_sample() - 0.5) * math.pi
        vector1 = llastCA - lastCA
        vector2 = normalize(np.array([1, 1, -(vector1[0] + vector1[1]) / vector1[2]]))
        vector3 = normalize(np.cross(vector2, vector1))
        tmp = math.cos(math.pi / 3) * 3.8 * vector2 + math.sin(math.pi / 3) * 3.8 * vector3

    return lastCA + tmp

def nextSC(CA, lastCA, lastSC, gamma, bsc, tsc):
    """
    Calculates the position of the next side-chain atom.

    Args:
        CA (np.ndarray): Coordinates of the current CA atom.
        lastCA (np.ndarray): Coordinates of the last CA atom.
        lastSC (np.ndarray): Coordinates of the last side-chain atom.
        gamma (float): Gamma angle in radians.
        bsc (float): Bond length for side-chain.
        tsc (float): Torsion angle for side-chain.

    Returns:
        np.ndarray: Coordinates of the next side-chain atom.
    """
    # Calculate orthonormal vectors
    vector1 = lastCA - CA
    vector2 = normalize(np.cross(vector1, lastSC - lastCA))
    vector3 = normalize(np.cross(vector1, vector2))

    # Calculate next side-chain position
    vector4 = bsc * math.cos(tsc) * (vector1 / 3.8)
    return CA + vector4 + bsc * math.sin(tsc) * (math.cos(gamma) * vector3 + math.sin(gamma) * vector2)

def transtocoord(gammalist, geometrylist, bsclist, tsclist, philist, recenter=False, C1A=np.array([0.0, 0.0, 0.0])):
    """
    Transforms torsion angles and bond parameters into 3D coordinates.

    Args:
        gammalist (list): Gamma angles in radians.
        geometrylist (list): Geometry type identifiers.
        bsclist (list): Bond lengths for side-chains.
        tsclist (list): Torsion angles for side-chains.
        philist (list): Phi torsion angles in radians.
        recenter (bool): Whether to recenter the structure around the origin.
        C1A (np.ndarray): Initial coordinate for the first atom.

    Returns:
        np.ndarray: Array of 3D coordinates for all atoms.
    """
    coord = np.zeros((len(geometrylist), 3))

    # Initialize the first few atoms
    if geometrylist[1] == 10:
        coord = init_glycine(coord, bsclist, tsclist, philist, geometrylist, C1A)
        start = 7
        gammaid = 0
    else:
        coord = init_other(coord, bsclist, tsclist, philist, geometrylist, C1A)
        start = 6
        gammaid = 0

    # Generate coordinates for remaining atoms
    for i in range(start, len(geometrylist) - 1, 3):
        coord[i] = nextSC(coord[i - 1], coord[i - 4], coord[i - 3],
                          gammalist[gammaid], bsclist[geometrylist[i]], tsclist[geometrylist[i]])
        coord[i + 2] = nextCA(coord[i - 1], coord[i - 4], coord[i], philist[geometrylist[i]])
        coord[i + 1] = 0.5 * (coord[i - 1] + coord[i + 2])
        gammaid += 1

    # Handle terminal atom
    if geometrylist[-1] == 10:
        coord[-1] = coord[-2]

    # Recenter if needed
    if recenter:
        coord -= np.mean(coord, axis=0)

    return coord

def normalize(vector):
    """Normalizes a vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def init_glycine(coord, bsclist, tsclist, philist, geometrylist, C1A):
    """Initializes coordinates for glycine."""
    coord[0] = coord[1] = C1A
    coord[3] = np.array([3.8, 0.0, 0.0])
    coord[2] = 0.5 * (coord[3] + coord[1])
    coord[4] = coord[3] + np.array([-bsclist[geometrylist[4]] * math.cos(tsclist[geometrylist[4]]),
                                    bsclist[geometrylist[4]] * math.sin(tsclist[geometrylist[4]]), 0])
    coord[6] = nextCA(coord[3], coord[0], coord[4], philist[geometrylist[4]])
    coord[5] = 0.5 * (coord[3] + coord[6])
    return coord

def init_other(coord, bsclist, tsclist, philist, geometrylist, C1A):
    """Initializes coordinates for non-glycine residues."""
    coord[0] = C1A
    coord[2] = np.array([3.8, 0.0, 0.0])
    coord[1] = 0.5 * (coord[2] + coord[0])
    coord[3] = coord[2] + np.array([-bsclist[geometrylist[3]] * math.cos(tsclist[geometrylist[3]]),
                                    bsclist[geometrylist[3]] * math.sin(tsclist[geometrylist[3]]), 0])
    coord[5] = nextCA(coord[2], coord[0], coord[3], philist[geometrylist[3]])
    coord[4] = 0.5 * (coord[2] + coord[5])
    return coord


def initialize(inputfasta, inputgeo):
    """
    Initializes the sequence, geometry, interactions, and coordinates from input FASTA and geometry files.

    Args:
        inputfasta (str): Path to the input FASTA file containing the sequence.
        inputgeo (str): Path to the input geometry file.

    Returns:
        tuple: Processed sequence list, geometry list, interaction list, gamma list, coordinates,
               bond side chain lengths (bsclist), torsion angles (tsclist), and phi angles (philist).
    """
    # Residue mapping dictionary
    residue_mapping = {
        'C': 1, 'M': 2, 'F': 3, 'I': 4, 'L': 5, 'V': 6, 'W': 7, 'Y': 8,
        'A': 9, 'G': 10, 'T': 11, 'S': 12, 'Q': 13, 'N': 14, 'E': 15, 'D': 16,
        'H': 17, 'R': 18, 'K': 19, 'P': 20
    }

    # Read and process sequence from the FASTA file
    with open(inputfasta, 'r') as f:
        sequence = f.readline().strip()

    seq_list = []
    for residue in sequence:
        if residue in residue_mapping:
            seq_list.append(residue_mapping[residue])
        elif residue == '\n':
            print("Sequence successfully read")
        else:
            print(f"Residue '{residue}' cannot be recognized. Please check.")
            exit()

    # Initialize geometry and interaction lists
    geometry_list, interaction_list = [], []

    # Handle the first residue
    if seq_list[0] == 10:  # Glycine
        geometry_list.extend([0, 10, 21])
        interaction_list.extend([0, 10, 21])
    else:  # Add dummy residue if the first residue is not Glycine
        geometry_list.extend([0, 21, 0, seq_list[0], 21])
        interaction_list.extend([0, 0, 0, seq_list[0], 21])

    # Process middle residues
    for i in range(1, len(seq_list) - 1):
        geometry_list.extend([0, seq_list[i], 21])
        interaction_list.extend([0, seq_list[i], 21])

    # Handle the last residue
    if seq_list[-1] == 10:  # Glycine
        geometry_list.extend([0, 10])
        interaction_list.extend([0, 10])
    else:  # Add dummy residue if the last residue is not Glycine
        geometry_list.extend([0, seq_list[-1], 21, 0])
        interaction_list.extend([0, seq_list[-1], 0, 0])

    # Convert lists to NumPy arrays
    seq_list = np.array(seq_list)
    geometry_list = np.array(geometry_list)
    interaction_list = np.array(interaction_list)

    # Initialize torsion angles
    gamma_list = np.zeros(np.count_nonzero(geometry_list == 0) - 3)
    gamma_list[::2] = math.pi

    # Read and process geometry data
    standardgeo = readstdgeometry(inputgeo)
    bsclist = standardgeo[:, 0]  # Bond side-chain lengths
    tsclist = standardgeo[:, 1] * (math.pi / 180.0)  # Torsion angles in radians
    philist = standardgeo[:, 2] * (math.pi / 180.0)  # Phi angles in radians

    # Transform torsion angles and bond parameters into 3D coordinates
    coord_list = transtocoord(gamma_list, geometry_list, bsclist, tsclist, philist)

    return seq_list, geometry_list, interaction_list, gamma_list, coord_list, bsclist, tsclist, philist

