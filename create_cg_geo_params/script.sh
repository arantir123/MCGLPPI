#!/bin/csh


#pdbname="1AK4.pdb"    # All-atom PDB file

#python2 py/martini.py -f $pdbname -o cg_M2.top -x cg_M2.pdb -dssp py/dssp -p backbone -ff martini22

#python py/itpconv.py Protein_A.itp Protein_A.itp > cg_A_M2.itp
#python py/itpconv.py Protein_B.itp > cg_B_M2.itp

#martinize2 -f $pdbname -o cg_M3.top -x cg_M3.pdb -dssp /home/guojiabin/software/miniconda3/envs/vermouth0.7.3/bin/mkdssp -p backbone -ff martini3001 -maxwarn 10
#python py/m3_to_m2.py molecule_0.itp Protein_A.itp > molecule_0_M2.itp
#python py/m3_to_m2.py molecule_1.itp Protein_B.itp > molecule_1_M2.itp
#python py/itpconv.py molecule_0_M2.itp > cg_A_M3.itp
#python py/itpconv.py molecule_1_M2.itp > cg_B_M3.itp

pdbname="4P2Q.pdb"
python2 py/martini.py -f $pdbname -o cg.top -x cg.pdb -dssp py/dssp -p backbone -ff martini22
python py/itpconv.py Protein_A.itp Protein_B.itp Protein_C.itp > cg_A.itp
python py/itpconv.py Protein_D.itp Protein_E.itp > cg_B.itp
