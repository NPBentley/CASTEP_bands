"""
The module handles space group and crystallographic symmetry functionality.

This is mainly needed for the determination of high-symmetry point labels within
the Brillouin zone based on the symmetry of the Bravais lattice.
"""
# Created by: V Ravindran, 28/08/2024
import ase.lattice as latt
import ase.spacegroup as spg


def _get_bravais_lattice_spg(cell):
    """Determine the high-symmetry points uisng the spacegroup.

    ASE will use just use the crystal system based on solely the lattice parameters
    whereas CASTEP will utilise the space group, which will be particularly important
    in e.g. magnetic materials where the computational cell may have a different crystal system
    to the (conventional) crystallographic one.

    For low-symmetry Bravais lattices where the special/high-symmetry points are
    lattice parameter dependent, we will use the lattice parameters of the computational cell.

    Author : V Ravindran 08/05/2024
    Moved from Spectral to here and updated to return the Bravais lattice itself instead 28/08/2024

    """
    # Get all the Bravais lattices
    bv_dict = latt.bravais_lattices

    # Get the cell's lattice parameters
    a, b, c, alpha, beta, gamma = cell.cell.cellpar()

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
        bv = bv_dict['TRI'](a=a, b=b, c=c,
                            alpha=alpha, beta=beta, gamma=gamma)
    elif 3 <= spg_no <= 15:
        # Monoclinic
        if bv_symb == 'P':  # Primitive monoclinic
            bv = bv_dict['MCL'](a=a, b=b, c=c,
                                alpha=alpha)
        elif bv_symb == 'C':  # Base-centred (C-centred) monoclinic
            bv = bv_dict['MCLC'](a=a, b=b, c=c,
                                 alpha=alpha)
        else:
            raise IndexError(f'Unknown monoclinic lattice with space group: {spg_cell.symbol}')
    elif 16 <= spg_no <= 74:
        # Orthorhombic
        if bv_symb == 'P':  # Primitive Orthorhombic
            bv = bv_dict['ORC'](a=a, b=b, c=c)
        elif bv_symb == 'I':  # Body-Centred Orthorhombic
            bv = bv_dict['ORCI'](a=a, b=b, c=c)
        elif bv_symb == 'F':  # Face-Centred Orthorhombic
            bv = bv_dict['ORCF'](a=a, b=b, c=c)
        elif bv_symb == 'A' or bv_symb == 'C':  # A/C-centred
            bv = bv_dict['ORCC'](a=a, b=b, c=c)
        else:
            raise IndexError(f'Unknown orthorhombic lattice with space group: {spg_cell.symbol}')
    elif 75 <= spg_no <= 142:
        # Tetragonal
        if bv_symb == 'P':  # Primitive Tetragonal
            bv = bv_dict['TET'](a=a, c=c)
        elif bv_symb == 'I':  # Body-Centred Tetragonal
            bv = bv_dict['BCT'](a=a, c=c)
        else:
            raise IndexError(f'Unknown tetragonal lattice with space group: {spg_cell.symbol}')
    elif 143 <= spg_no <= 167:
        # Trigonal
        if bv_symb == 'R':  # R-trigonal/Rhombohedral
            bv = bv_dict['RHL'](a=a, alpha=alpha)
        elif bv_symb == 'P':  # Hexagonal
            bv = bv_dict['HEX'](a=a, c=c)
        else:
            raise IndexError(f'Unknown trigonal lattice with space group: {spg_cell.symbol}')
    elif 168 <= spg_no <= 194:
        # Hexagonal
        bv = bv_dict['HEX'](a=a, c=c)
    elif 195 <= spg_no <= 230:
        # Cubic
        if bv_symb == 'P':  # Primitive/Simple Cubic
            bv = bv_dict['CUB'](a=a)
        elif bv_symb == 'I':  # Body-Centred Cubic
            bv = bv_dict['BCC'](a=a)
        elif bv_symb == 'F':  # Face-Centred Cubic
            bv = bv_dict['FCC'](a=a)
        else:
            raise IndexError(f'Unknown cubic lattice with space group: {spg_cell.symbol}')
    else:
        raise IndexError(f'Unknown Spacegroup {spg_no}: {spg_cell.symbol}')

    return bv


def _get_bravais_lattice_usr(cell, bv_type):
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
    elif bv_type == 'MCL':  # Primitive Monoclinic
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha)
    elif bv_type == 'MCLC':  # Base-centred (C-centred) monoclinic
        bv = bv_dict[bv_type](a=a, b=b, c=c,
                              alpha=alpha)
    elif bv_type == 'ORC':  # Primitive Orthorhombic
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type == 'ORCI':  # Body-Centred Orthorhombic
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type == 'ORCF':  # Face-Centred Orthorhombic
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type == 'ORCC':  # A/C-centred Orthorhombic
        bv = bv_dict[bv_type](a=a, b=b, c=c)
    elif bv_type == 'TET':  # Primitive Tetragonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type == 'BCT':  # Body-Centred Tetragonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type == 'RHL':  # R-trigonal/Rhombohedral
        bv = bv_dict[bv_type](a=a, alpha=alpha)
    elif bv_type == 'HEX':  # Hexagonal
        bv = bv_dict[bv_type](a=a, c=c)
    elif bv_type == 'CUB':  # Primitive/Simple Cubic
        bv = bv_dict[bv_type](a=a)
    elif bv_type == 'BCC':  # Body-Centred Cubic
        bv = bv_dict[bv_type](a=a)
    elif bv_type == 'FCC':  # Face-Centred Cubic
        bv = bv_dict[bv_type](a=a)
    else:
        # Raise error unless someone's reinvented crystallography and how 3D space works...
        raise IndexError(f'Unknown Bravais lattice: {bv_type}')

    return bv
