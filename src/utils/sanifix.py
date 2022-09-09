#! /bin/env/python
from rdkit import Chem

##### Sanifix 4 by James Davidson ######
""" sanifix4.py

  Contribution from James Davidson
"""


def _FragIndicesToMol(oMol, indices):
    em = Chem.rdchem.EditableMol(Chem.rdchem.Mol())

    newIndices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx] = i

    for i, idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx < idx:
                continue
            em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.rdmolops.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = newIndices
    return res


def _recursivelyModifyNs(mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    while len(matches) and res is None:
        tIndices = indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.rdchem.Mol(mol)
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(0)
        cp = Chem.rdchem.Mol(nm)
        try:
            Chem.rdmolops.SanitizeMol(cp)
            res, indices = _recursivelyModifyNs(nm, matches, indices=tIndices)
        except ValueError:
            indices = tIndices
            res = cp
    return res, indices


def AdjustAromaticNs(m, nitrogenPattern="[n&D2&H1;r5,r6]"):
    """
    default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
    to fix: O=c1ccncc1
    """
    # Get a symmetrized SSSR for a molecule.
    Chem.rdmolops.GetSymmSSSR(m)
    # Regenerates computed properties like implicit valence and ring
    # information.
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = Chem.rdchem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.rdmolfiles.MolFromSmarts("[r]!@[r]"))
    plsFix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at = nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.rdmolops.GetMolFrags(nm)
    frags = [_FragIndicesToMol(nm, x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    for i, frag in enumerate(frags):
        Chem.rdchem.Mol(frag)
        # need something to only do this if n present  maybe
        matches = [
            x[0] for x in frag.GetSubstructMatches(
                Chem.rdmolfiles.MolFromSmarts(nitrogenPattern))
        ]
        lres, indices = _recursivelyModifyNs(frag, matches)
        if not lres:
            # print 'frag %d failed (%s)'%(i,str(fragLists[i]))
            break
        else:
            revMap = {}
            for k, v in frag._idxMap.items():
                revMap[v] = k
            for idx in indices:
                oatom = m.GetAtomWithIdx(revMap[idx])
                oatom.SetNoImplicit(True)
                oatom.SetNumExplicitHs(0)

    return m
