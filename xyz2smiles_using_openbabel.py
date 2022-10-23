import pybel

def xyz_to_smiles(fname: str) -> str:
    mol = next(pybel.readfile("xyz", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()

smi = xyz_to_smiles("../tmqm_data_github/data/tmQM_X1_first_mol.xyz")
print(smi)
