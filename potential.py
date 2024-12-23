import numpy as np
print("Numpy imported in potential:", "np" in globals())

def get_SCSC_parameter(MJparameterfile, r0parameterfile, interactionlist, egg=0.025): 
    # Calculate side-chain contact parameters based on the given interaction list and parameter files.
    # egg: epsilon for Glycine-Glycine (GLY, GLY) interaction.

    # Import MJ contact energies and symmetrize the matrix
    MJlist = np.loadtxt(MJparameterfile)  # Load MJ contact energy matrix
    for i in range(1, MJlist.shape[0]):
        for j in range(0, i):
            MJlist[i, j] = MJlist[j, i]  # Ensure the matrix is symmetric

    # Import van der Waals radii
    r0list = np.loadtxt(r0parameterfile)

    # Extract side-chain indices and calculate r0ij (average van der Waals radii for pairs)
    SClist = interactionlist[(interactionlist < 21) & (interactionlist > 0)]  # Side-chain list
    r0ij = np.zeros((len(SClist), len(SClist)))
    for i in range(len(SClist)):
        for j in range(len(SClist)):
            r0ij[i, j] = r0list[SClist[i] - 1] + r0list[SClist[j] - 1]  # Sum radii for pairs
    r0ij = r0ij / 2.0  # Compute the average radii

    # Calculate eij (interaction energies) for the side-chain pairs
    eMJij = np.zeros((len(SClist), len(SClist)))
    fij = np.ones((len(SClist), len(SClist)))  # Scaling factors for interaction energies

    for i in range(len(SClist)):
        for j in range(len(SClist)):
            eMJij[i, j] = MJlist[SClist[i] - 1, SClist[j] - 1]  # MJ interaction energy
            # Adjust fij based on specific conditions involving Proline (P)
            if SClist[i] == 20 and SClist[j] == 20: 
                fij[i, j] = 0.41
            elif SClist[i] == 20 or SClist[j] == 20: 
                fij[i, j] = 0.79

    # Calculate adjusted interaction energies (eij)
    #print(MJlist[9, 9])  # Debugging output for Glycine-Glycine interaction energy
    eij = 0.6 * (eMJij - fij * MJlist[9, 9])  # Adjusted by scaling and Glycine interaction energy
    for i in range(len(SClist)):
        for j in range(len(SClist)):
            if SClist[i] == 10 and SClist[j] == 10:  # Special case for Glycine-Glycine interaction
                eij[i, j] = egg 

    # Set self-interactions to zero
    eij[np.diag_indices(len(SClist))] = 0.0

    # Separate hydrophobic (eij_hypho) and hydrophilic (eij_hyphi) interactions
    eij_hypho = np.copy(eij)
    eij_hypho[eij_hypho < 0] = 0.0  # Keep only positive interactions
    eij_hyphi = np.copy(eij)
    eij_hyphi[eij_hyphi > 0] = 0.0  # Keep only negative interactions

    return MJlist, r0ij, eij_hypho, eij_hyphi


def potential_SCSC(MJlist, r0ij, interactionlist, coord, eij_hypho, eij_hyphi, rc):
    """
    Calculate the side-chain contact potential energy for a given configuration.

    Args:
        MJlist, r0ij (np.ndarray): Matrices of MJ interaction energies and radii.
        interactionlist (np.ndarray): Interaction list containing atom types.
        coord (np.ndarray): Atomic coordinates.
        eij_hypho, eij_hyphi (np.ndarray): Hydrophobic and hydrophilic interaction energies.
        rc (float): Cutoff radius for interactions.

    Returns:
        float: Total side-chain contact potential energy.
    """
    # Extract coordinates for side chains
    SCcoord = coord[(interactionlist < 21) & (interactionlist > 0)]

    # Calculate pairwise displacements and distances
    SC_displacements = SCcoord[:, np.newaxis, :] - SCcoord[np.newaxis, :, :]
    rSC = np.linalg.norm(SC_displacements, axis=-1)
    rSC[np.diag_indices(len(rSC))] = np.inf  # Avoid self-interaction

    # Compute hydrophobic interaction potentials
    tmp = np.power(r0ij / rSC, 6)
    v_hypho = eij_hypho * tmp * (tmp - 2.0)  # Lennard-Jones-like term

    # Compute hydrophilic interaction potentials
    v_hyphi = -eij_hyphi * tmp

    # Apply cutoff corrections for interactions
    tmp2 = np.power(r0ij / rc, 6)
    vc_hypho = eij_hypho * tmp2 * (tmp2 - 2.0)  # Cutoff for hydrophobic interactions
    vc_hyphi = -eij_hyphi * tmp2  # Cutoff for hydrophilic interactions

    v_hypho[rSC < rc] -= vc_hypho[rSC < rc]  # Shift hydrophobic interactions
    v_hypho[rSC >= rc] = 0  # Cut off hydrophobic interactions beyond rc
    v_hyphi[rSC < rc] -= vc_hyphi[rSC < rc]  # Shift hydrophilic interactions
    v_hyphi[rSC >= rc] = 0  # Cut off hydrophilic interactions beyond rc

    # Return total potential energy
    return 0.5 * np.sum(v_hypho + v_hyphi)




def get_torsion_parameter(torsionparameter, interactionlist, geometry_list):
    """
    Extracts torsion parameters (alist and blist) for given side-chain interactions.

    Args:
        torsionparameter (str): Path to the torsion parameter file.
        interactionlist (np.ndarray): Interaction list for the system.

    Returns:
        tuple: alist_for_link and blist_for_link matrices containing torsion parameters.
    """
    # Load torsion parameter file
    parameterlist = np.loadtxt(torsionparameter)  # Load torsion parameters from file
    alist = parameterlist[:, 0::2]  # Extract A coefficients
    blist = parameterlist[:, 1::2]  # Extract B coefficients

    # Identify side-chain indices excluding terminal ones
    SCindex = np.array(np.where(geometry_list == 0))[0][1:-1] + 1  # Middle SCs only
    SClist = interactionlist[SCindex.tolist()]  # Extract interaction list for the side chains

    # Initialize torsion parameter matrices
    alist_for_link = np.zeros((len(SClist) - 1, 6))  # Matrix for A coefficients
    blist_for_link = np.zeros((len(SClist) - 1, 6))  # Matrix for B coefficients

    # Populate torsion parameter matrices
    for i in range(len(SClist) - 1):
        tmpbefore = SClist[i]  # Residue before the link
        tmpnext = SClist[i + 1]  # Residue after the link

        # Assign torsion parameters based on residue type pairs
        if tmpbefore == 10:  # Glycine
            if tmpnext == 10:  # Glycine-Glycine
                alist_for_link[i] = alist[0]
                blist_for_link[i] = blist[0]
            elif tmpnext == 20:  # Glycine-Proline
                alist_for_link[i] = alist[6]
                blist_for_link[i] = blist[6]
            else:  # Glycine-other
                alist_for_link[i] = alist[3]
                blist_for_link[i] = blist[3]
        elif tmpbefore == 20:  # Proline
            if tmpnext == 10:  # Proline-Glycine
                alist_for_link[i] = alist[2]
                blist_for_link[i] = blist[2]
            elif tmpnext == 20:  # Proline-Proline
                alist_for_link[i] = alist[8]
                blist_for_link[i] = blist[8]
            else:  # Proline-other
                alist_for_link[i] = alist[5]
                blist_for_link[i] = blist[5]
        else:  # Other residues
            if tmpnext == 10:  # Other-Glycine
                alist_for_link[i] = alist[1]
                blist_for_link[i] = blist[1]
            elif tmpnext == 20:  # Other-Proline
                alist_for_link[i] = alist[7]
                blist_for_link[i] = blist[7]
            else:  # Other-other
                alist_for_link[i] = alist[4]
                blist_for_link[i] = blist[4]

    return alist_for_link, blist_for_link  # Return matrices of torsion parameters


def potential_torsion(gammalist, alist_for_link, blist_for_link):
    """
    Calculates the torsion potential energy for the system.

    Args:
        gammalist (np.ndarray): List of gamma torsion angles in radians.
        alist_for_link (np.ndarray): A coefficients for torsion potential.
        blist_for_link (np.ndarray): B coefficients for torsion potential.

    Returns:
        float: Total torsion potential energy.
    """
    # Expand gamma angles for harmonic terms
    gammamap = np.array(np.tile(gammalist, (6, 1))).transpose()  # Repeat gamma angles for six harmonics

    # Multiply gamma by harmonic factors (1, 2, ..., 6)
    tmp = gammamap * np.tile(np.array(range(1, 7)), (len(gammalist), 1))

    # Compute torsion potential energy
    p = np.sum(alist_for_link * np.cos(tmp) + blist_for_link * np.sin(tmp))  # Sum energy contributions

    return p  # Return total torsion potential energy



def get_SCP_parameter(interactionlist):
    """
    Calculates the SCP (side-chain to peptide-bond center) mask based on interaction distances.

    Args:
        interactionlist (np.ndarray): Interaction list containing atom types.

    Returns:
        np.ndarray: SCP mask with constraints applied for specific distances.
    """
    # Identify indices for side chains (SC) and peptide-bond centers (P)
    SCindex = np.where((interactionlist > 0) & (interactionlist < 21))[0]  # SC indices
    Pindex = np.where(interactionlist == 21)[0]  # P indices

    # Calculate absolute distances between SC and P indices
    SCPdistance = SCindex[:, np.newaxis] - Pindex[np.newaxis, :]  # Pairwise differences
    SCPdistance = np.abs(SCPdistance)  # Take absolute values for distances

    # Create an SCP mask initialized with ones
    SCPmask = np.ones(SCPdistance.shape)

    # Set mask to 0 for interactions within a distance of 1 or 2
    SCPmask[(SCPdistance == 1) | (SCPdistance == 2)] = 0.0

    # Debugging: Print the shape of the SCP mask
    print(SCPmask.shape)

    return SCPmask  # Return the SCP mask


def potential_SCP(interactionlist, coord, SCPmask, rscp=4.0, escp=0.3):
    """
    Calculates the SCP potential energy based on interaction distances and a mask.

    Args:
        interactionlist (np.ndarray): Interaction list containing atom types.
        coord (np.ndarray): Atomic coordinates.
        SCPmask (np.ndarray): SCP mask for excluding specific interactions.
        rscp (float): Reference SCP interaction distance.
        escp (float): SCP interaction strength.

    Returns:
        float: Total SCP potential energy.
    """
    # Extract coordinates for SC and P atoms
    SCcoord = coord[(interactionlist > 0) & (interactionlist < 21)]  # Coordinates of side chains
    Pcoord = coord[(interactionlist == 21)]  # Coordinates of peptide-bond centers

    # Calculate pairwise displacements between SC and P atoms
    SCP_displacements = SCcoord[:, np.newaxis, :] - Pcoord[np.newaxis, :, :]

    # Compute pairwise distances
    rSCP = np.linalg.norm(SCP_displacements, axis=-1)

    # Compute potential energy contribution using rSCP and mask
    v = np.power(rSCP, -6) * SCPmask  # Weighted by SCP mask to exclude invalid interactions

    # Scale and sum the potential energy contributions
    return escp * np.power(rscp, 6) * np.sum(v)



def get_PP_parameter(interactionlist, PPparameterfile):
    """
    Computes pairwise parameters (Aij, Bij, rpij, epij) for peptide-bond interactions.

    Args:
        interactionlist (np.ndarray): Interaction list containing atom types.
        PPparameterfile (str): Path to the peptide-bond parameter file.

    Returns:
        tuple: Matrices Aij, Bij, rpij, epij for pairwise interactions, and lists/masks for interactions.
    """
    # Load peptide-bond parameters from file
    pp_parameter = np.loadtxt(PPparameterfile)
    ep = pp_parameter[:, 0]  # Interaction strengths
    rp = pp_parameter[:, 1]  # Reference distances
    A = pp_parameter[:, 2]  # Coefficients for first term
    B = pp_parameter[:, 3]  # Coefficients for second term

    # Identify valid side chains (excluding terminal proline residues)
    valid_SClist = []
    for i in range(0, len(interactionlist)):
        if interactionlist[i] == 21:
            valid_SClist.append(interactionlist[i - 1])  # Add the residue before proline
        else:
            continue
    valid_SClist = np.array(valid_SClist)  # Convert to a NumPy array

    # Initialize parameter matrices
    n = len(valid_SClist)
    Aij = A[0] * np.ones((n, n))
    Bij = B[0] * np.ones((n, n))
    rpij = rp[0] * np.ones((n, n))
    epij = ep[0] * np.ones((n, n))

    # Find indices of proline residues
    op_index = np.asarray(np.where(valid_SClist == 20)[0])

    if len(op_index) != 0:
        # Adjust parameters for interactions involving proline
        for i in op_index:
            Aij[i, :] = A[1] * np.ones(n)
            Aij[:, i] = A[1] * np.ones(n)
            Bij[i, :] = B[1] * np.ones(n)
            Bij[:, i] = B[1] * np.ones(n)
            rpij[i, :] = rp[1] * np.ones(n)
            rpij[:, i] = rp[1] * np.ones(n)
            epij[i, :] = ep[1] * np.ones(n)
            epij[:, i] = ep[1] * np.ones(n)

        # Adjust parameters for proline-proline interactions
        if len(op_index) >= 2:
            for j in op_index:
                for k in op_index:
                    Aij[j, k] = A[2]; Aij[k, j] = A[2]
                    Bij[j, k] = B[2]; Bij[k, j] = B[2]
                    rpij[j, k] = rp[2]; rpij[k, j] = rp[2]
                    epij[j, k] = ep[2]; epij[k, j] = ep[2]

    # Set diagonal entries to zero to avoid self-interaction
    Aij[np.diag_indices(n)] = 0.0
    Bij[np.diag_indices(n)] = 0.0
    rpij[np.diag_indices(n)] = 0.0
    epij[np.diag_indices(n)] = 0.0

    # Identify indices for peptide-bond-related atoms
    tmp = np.asarray(np.where(interactionlist == 21)[0])
    Cabeforelist = [i - 2 for i in tmp]  # Indices of atoms before peptide bonds
    Caafterlist = [i + 1 for i in tmp]  # Indices of atoms after peptide bonds
    Plist = [i for i in tmp]  # Indices of peptide bonds themselves

    # Compute pairwise distances for peptide bonds
    Pindex = np.array(range(len(Plist)))
    PPdistance = Pindex[:, np.newaxis] - Pindex[np.newaxis, :]
    PPdistance = np.abs(PPdistance)

    # Create a mask for valid interactions (exclude self and adjacent interactions)
    PPmask = np.ones(PPdistance.shape)
    PPmask[(PPdistance == 1) | (PPdistance == 0)] = 0.0

    return Aij, Bij, rpij, epij, Cabeforelist, Caafterlist, Plist, PPmask


def potential_PP(Aij, Bij, rpij, epij, coord, Cabeforelist, Caafterlist, Plist, PPmask):
    """
    Computes the potential energy for peptide-bond interactions.

    Args:
        Aij, Bij, rpij, epij (np.ndarray): Matrices of pairwise parameters.
        coord (np.ndarray): Atomic coordinates.
        Cabeforelist (list): Indices of atoms before peptide bonds.
        Caafterlist (list): Indices of atoms after peptide bonds.
        Plist (list): Indices of peptide bonds.
        PPmask (np.ndarray): Mask for valid peptide-bond interactions.

    Returns:
        float: Total potential energy of the peptide-bond interactions.
    """
    # Extract coordinates of peptide-bond-related atoms
    Cabefore_coord = coord[Cabeforelist]
    Caafter_coord = coord[Caafterlist]
    P_coord = coord[Plist]

    # Compute vectors and pairwise distances
    v = Caafter_coord - Cabefore_coord
    aij = np.sum(v[:, np.newaxis, :] * v[np.newaxis, :, :], axis=2)
    rij_disp = P_coord[:, np.newaxis, :] - P_coord[np.newaxis, :, :]
    bij = np.sum(v[:, np.newaxis, :] * rij_disp, axis=2)  # Projection of displacement vectors
    gij = -bij.transpose()  # Transpose of bij
    rij = np.linalg.norm(rij_disp, axis=-1)  # Pairwise distances

    # Avoid self-interaction
    rij[np.diag_indices(rij.shape[0])] = np.inf

    # Compute terms for potential energy
    tmp1 = np.power(rij, -6)
    tmp2 = np.power(3.8, -2) * (aij - 3 * bij * gij / (rij**2))
    tmp3 = np.power(rpij / rij, 6)

    # Compute the potential energy
    p = (Aij * np.power(rij, -3) * tmp2
         - Bij * tmp1 * (4 + tmp2**2 - 3 * (bij**2 + gij**2) / (rij**2 * 3.8**2))
         + epij * tmp3 * (tmp3 - 2.0))

    # Apply mask to exclude invalid interactions
    p = p * PPmask

    # Return total potential energy
    return np.sum(0.5 * p)

