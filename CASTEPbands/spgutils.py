"""
The module handles space group and crystallographic symmetry functionality.

This is mainly needed for the determination of high-symmetry point labels within
the Brillouin zone based on the symmetry of the Bravais lattice.
"""
# Created by: V Ravindran, 28/08/2024
import ase
import ase.lattice as latt
import ase.spacegroup as spg
import numpy as np


def _get_bravais_lattice_spg(cell: ase.Atoms):
    """Determine the high-symmetry points uisng the spacegroup.

    ASE will use just use the crystal system based on solely the lattice parameters
    whereas CASTEP will utilise the space group, which will be particularly important
    in e.g. magnetic materials where the computational cell may have a different crystal system
    to the (conventional) crystallographic one.

    For low-symmetry Bravais lattices where the special/high-symmetry points are
    lattice parameter dependent, we will use the lattice parameters of the computational cell.

    Author : V Ravindran 08/05/2024

    Updated: 28/08/2024
    * Moved from Spectral to here
    * Now returns the Bravais lattice itself rather than the high symmetry points
    * Wraps to _get_bravais_lattice_usr to actually get the Bravais lattice
      saving on some duplication of code.

    """
    # Get the space group information for this cell
    spg_cell = spg.get_spacegroup(cell)
    spg_no = spg_cell.no

    # Get the first letter of the spacegroup in international notation.
    # This gives the information about the Bravais lattice
    bv_symb = spg_cell.symbol.split()[0].upper()

    # Now use the space group to determine the crystal system.
    # We can determine the actual Bravais lattice using the first
    # letter of the international notation symbol.
    #
    # Like in CASTEP, particularly for low symmetry Bravais lattices
    # where the high symmetry points depend on lattice parameters,
    # we will use the computational cell's lattice parameters.
    # Variations for each bravais lattice should be handled by ASE (in principle...)
    if 1 <= spg_no <= 2:
        # Triclinic lattice
        bv_type = 'TRI'
    elif 3 <= spg_no <= 15:
        # Monoclinic
        if bv_symb == 'P':  # Primitive monoclinic
            bv_type = 'MCL'
        elif bv_symb == 'C':  # Base-centred (C-centred) monoclinic
            bv_type = 'MCLC'
        else:
            raise IndexError(f'Unknown monoclinic lattice with space group: {spg_cell.symbol}')
    elif 16 <= spg_no <= 74:
        # Orthorhombic
        if bv_symb == 'P':  # Primitive Orthorhombic
            bv_type = 'ORC'
        elif bv_symb == 'I':  # Body-Centred Orthorhombic
            bv_type = 'ORCI'
        elif bv_symb == 'F':  # Face-Centred Orthorhombic
            bv_type = 'ORCF'
        elif bv_symb == 'A' or bv_symb == 'C':  # A/C-centred Orthorhombic
            bv_type = 'ORCC'
        else:
            raise IndexError(f'Unknown orthorhombic lattice with space group: {spg_cell.symbol}')
    elif 75 <= spg_no <= 142:
        # Tetragonal
        if bv_symb == 'P':  # Primitive Tetragonal
            bv_type = 'TET'
        elif bv_symb == 'I':  # Body-Centred Tetragonal
            bv_type = 'BCT'
        else:
            raise IndexError(f'Unknown tetragonal lattice with space group: {spg_cell.symbol}')
    elif 143 <= spg_no <= 167:
        # Trigonal
        if bv_symb == 'R':  # R-trigonal/Rhombohedral
            bv_type = 'RHL'
        elif bv_symb == 'P':  # Hexagonal
            bv_type = 'HEX'
        else:
            raise IndexError(f'Unknown trigonal lattice with space group: {spg_cell.symbol}')
    elif 168 <= spg_no <= 194:
        # Hexagonal
        bv_type = 'HEX'
    elif 195 <= spg_no <= 230:
        # Cubic
        if bv_symb == 'P':  # Primitive/Simple Cubic
            bv_type = 'CUB'
        elif bv_symb == 'I':  # Body-Centred Cubic
            bv_type = 'BCC'
        elif bv_symb == 'F':  # Face-Centred Cubic
            bv_type = 'FCC'
        else:
            raise IndexError(f'Unknown cubic lattice with space group: {spg_cell.symbol}')
    else:
        raise IndexError(f'Unknown Spacegroup {spg_no}: {spg_cell.symbol}')

    # Now get the Bravais lattice
    bv = _get_bravais_lattice_usr(cell, bv_type)
    return bv


def _get_bravais_lattice_usr(cell: ase.Atoms, bv_type: str):
    """Get a Bravais lattice matching the lattice parameters of the user's unit cell.

    This effectively is a wrapper for ASE. Since the ASE routine takes only the arguments
    required to specify the Bravais lattice (the others implied by symmetry), we need to
    account for each Bravais lattice type ourselves.
    """
    # Get all the Bravais lattices
    bv_dict = latt.bravais_lattices

    # Get the cell's lattice parameters
    a, b, c, alpha, beta, gamma = cell.cell.cellpar()

    # Here begins the boring bit - there should be 14 Bravais lattices here!
    # Since the ASE interface is not overloaded, we will have to
    # set the arguments ourselves here.
    # Lattice Parameters not specified have values implied by type of Bravais lattice.
    if bv_type == 'TRI':  # Triclinic lattice
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha, beta=beta, gamma=gamma)
    elif bv_type in ('MCL', 'MCLC'):  # Primitive/Base-centred (C-centred) Monoclinic
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha)
    elif bv_type in ('ORC', 'ORCI', 'ORCF', 'ORCC'):  # Primitive or Body-/Face-Centred or A/C-Centred Orthorhombic
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type in ('TET', 'BCT'):  # Primitive/Body-Centred Tetragonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type == 'RHL':  # R-trigonal/Rhombohedral
        bv = bv_dict[bv_type](a=a, alpha=alpha)
    elif bv_type == 'HEX':  # Hexagonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type in ('CUB', 'BCC', 'FCC'):  # Primitive/Simple or Body-Centred or Face-Centred Cubic
        bv = bv_dict[bv_type](a=a)
    else:
        # Unless someone's reinvented crystallography and how 3D space works...
        raise IndexError(f'Unknown Bravais lattice: {bv_type}')

    return bv


def _check_sym_pt(vec):
    """Checks if current point is a high-symmetry point.

    Written by Zachary Hawkhead
    Moved here as part of by Visagan Ravindran 06/11/24 SYM_LINES_REFACTOR
    """
    # TODO V Ravindran: I am not sure this covers all bases, need to check some more....
    fracs = np.array([0.5, 0.0, 0.25, 0.75, 0.33333333, 0.66666667])
    frac = []
    for i in vec:
        # frac.append(i.as_integer_ratio()[0])
        # frac.append(i.as_integer_ratio()[1])
        buff = []
        for j in fracs:
            buff.append(np.isclose(i, j))
        frac.append(any(buff))

    if all(frac):
        # print(vec)
        return True
    else:
        return False


def _get_high_sym_lines(kpt_array: np.ndarray, cell: ase.Atoms,
                        high_sym_spacegroup: bool = True,
                        override_bv: str = None,
                        tol: float = 1e-5,
                        ret_special_pts: bool = False
                        ):
    """Determine location of high-symmetry points from a set of kpoints for a unit cell.

    IMPORTANT: This routine relies on the kpoints having been sorted prior to call!

    The determination of the Bravais lattice for the unit cell is done on the following priority:
    1. override_bv is specified: use the specified Bravais lattice into the function

    2. high_sym_spacegroup(default) : Use the space group to determine the Bravais lattice
       This matches what is done in CASTEP and is more robust than what follows.

    3. Use Bravais lattice based on lattice parameters:
       This may not always match CASTEP convention - particularly when a cell is magnetic and one has a
       e.g. a rhombohedral cell for the magnetic primitive cell that is crystallographically still FCC.

    Parameters
    ----------
    kpt_array : np.ndarray
        list of kpoints shape=(nkpts,3)
    cell : ase.Atoms
        the unit cell for the calculation
    high_sym_spacegroup : bool
        use the spacegroup to determine the Bravais lattice.
    override_bv : str
        str
    tol : float
        tolerance in determining if current point is a high-symmetry point
    ret_special_pts : bool
        if true, return the special points for the Bravais lattice

    Returns
    -------
    high_sym: ndarray
        kpoint index of high-symmetry points
    tick_labels : list
        labels for all ticks (blank string if not high-symmetry point, label otherwise)

    Raises
    ------
    IndexError
        kpt_array does not have the correct shape

    """

    # Let's first check if you have an array of kpoints with the correct ordering
    if kpt_array.ndim != 2:
        raise IndexError('Expected shape of kpt_array is (nkpts, 3) with 2 dimensions')
    if kpt_array.shape[1] != 3:
        raise IndexError('Expected shape of kpt_array is (nkpts, 3) but final index has size ' +
                         str(kpt_array.shape[1]))

    # Determine the ticks for high-symmetry points
    k_ticks = []
    for ik, vec in enumerate(kpt_array):
        if _check_sym_pt(vec):
            k_ticks.append(ik)

    # Determine the kpoint gradient
    # kpt_grad = []
    # for i in range(1, len(kpt_array)):
    #     diff = kpt_array[i] - kpt_array[i-1]
    #     kpt_grad.append(diff)
    # V Ravindran 06/11/2024 - we can use vectorised numpy functions instead
    kpt_grad = np.diff(kpt_array, axis=0)

    # Get the second derivative
    high_sym = [0]  # HACK: Assume we are starting at high-symmetry point
    # for i in range(1, len(kpt_grad)):
    #     kpt_2grad = kpt_grad[i] - kpt_grad[i-1]
    #     if any(np.abs(kpt_2grad) > tol):
    #         high_sym.append(i)
    # V Ravindran 06/11/2024 - we can use vectorised numpy functions instead
    kpt_2grad = np.diff(kpt_grad, axis=0)
    for i, diff in enumerate(kpt_2grad):
        if any(np.abs(diff) > tol):
            high_sym.append(i+1)  # +1 since we want index of kpt_grad which is 1 behind due to difference

    # HACK : Assume we finish on a high-symmetry point
    high_sym.append(len(kpt_array) - 1)

    # Turn into a Numpy array - the commented out +1 is historic when we insisted on using
    # Fortran style indexing starting from 1 rather than 0
    high_sym = np.array(high_sym, dtype=int)  # + 1 V Ravindran 06/11/2024 SYM_LINES_REFACTOR

    # Now set up the special point labels - this requires getting our Bravais lattice
    if override_bv is not None:
        # Allow user to manually specify the Bravais lattice.   V Ravindran OVERRIDE_BV 28/08/2024
        # Useful for defect calculations if one wants to        V Ravindran OVERRIDE_BV 28/08/2024
        # e.g. use high-symmetry labels of the main crystal.    V Ravindran OVERRIDE_BV 28/08/2024
        bv_latt = _get_bravais_lattice_usr(cell, override_bv)
    elif high_sym_spacegroup is True:
        # Default for special points is now to get them from the space group. V Ravindran 08/05/2024
        bv_latt = _get_bravais_lattice_spg(cell)
    else:
        # Get Bravais lattice from the crystal system of the computational cell.
        bv_latt = cell.cell.get_bravais_lattice()

    special_points = bv_latt.get_special_points()

    # Now initialise the tick labels
    tick_labels = [""] * len(high_sym)
    found = False
    for k_count, k in enumerate(kpt_array[high_sym]):
        found = False

        for i in special_points:
            if abs(special_points[i][0] - k[0]) < tol and abs(special_points[i][1] - k[1]) < tol and abs(special_points[i][2] - k[2]) < tol:
                if i == "G":
                    tick_labels[k_count] = r"$\Gamma$"
                else:
                    tick_labels[k_count] = i
                found = True

    if ret_special_pts:
        return high_sym, tick_labels, special_points
    else:
        return high_sym, tick_labels
