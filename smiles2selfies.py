import selfies as sf
from rdkit import Chem

# smiles = "CN1C(=O)C2=C(c3cc4c(s3)-c3sc(-c5ncc(C#N)s5)cc3C43OCCO3)N(C)C(=O)" \
        #  "C2=C1c1cc2c(s1)-c1sc(-c3ncc(C#N)s3)cc1C21OCCO1"
smiles_with_dative = "O=C[Fe](<-[PH4])(C=O)(C=O)C=O"
smiles_without = "O=C[Fe]([PH4])(C=O)(C=O)C=O"
smiles = smiles_without
encoded_selfies = sf.encoder(smiles)  # SMILES --> SEFLIES
decoded_smiles = sf.decoder(encoded_selfies)  # SELFIES --> SMILES

print(f"Original SMILES: {smiles}")
print(f"Translated SELFIES: {encoded_selfies}")
print(f"Translated SMILES: {decoded_smiles}")

print(f"== Equals: {smiles == decoded_smiles}")

# Recomended
can_smiles = Chem.CanonSmiles(smiles)
can_decoded_smiles = Chem.CanonSmiles(decoded_smiles)
print(f"RDKit Equals: {can_smiles == can_decoded_smiles}")
print("CURRENT CONCLUSION: Even selfies library cannot recognize -> dative symbol")