package require psfgen
package require solvate
package require autoionize

topology /home/zhenyuwei/nutstore/ZhenyuWei/Notes_Research/MDPy/mdpy/data/charmm/top_all36_prot.rtf
pdbalias residue HIS HSE

# Detach all elements but protein 
# set prot_name 6PO6
# mol load pdb $prot_name.pdb
# set all [atomselect top "all"]
# set com [measure center $all]
# set move_vec [vecmul $com {-1 -1 -1}]
# $all moveby $move_vec

# set prot [atomselect top "all protein"]
# put [measure minmax $prot]
# $prot writepdb $prot_name\_ditached.pdb

# # Patch protein
# mol delete top
# mol load pdb $prot_name\_ditached.pdb

# set sel [atomselect top protein]
# set chains [lsort -unique [$sel get chain]] ;# return A B C D

# foreach chain $chains {
#     puts "Adding protein chain $chain to psfgen"
#     set seg ${chain}
#     set sel [atomselect top "protein and chain $chain"]
#     $sel set segid $seg
#     $sel writepdb $prot_name\_tmp.pdb

#     segment $seg { pdb $prot_name\_tmp.pdb }
#     coordpdb $prot_name\_tmp.pdb
# }
# guesscoord       

# writepdb $prot_name\_patched.pdb
# writepsf $prot_name\_patched.psf

# mol load psf $prot_name\_patched.psf pdb $prot_name\_patched.pdb

# set all [atomselect top "all"]

# set minmax [measure minmax $all]
# set boxsize 15
# solvate $prot_name\_patched.psf $prot_name\_patched.pdb -minmax {{-15 -15 -15} {15 15 15}} -o $prot_name\_solvated

# autoionize -psf $prot_name\_solvated.psf -pdb $prot_name\_solvated.pdb -sc 0.15 -o $prot_name\_ionized


solvate -minmax {{-15 -15 -15} {15 15 15}} -o ion_solution
autoionize -psf ion_solution.psf -pdb ion_solution.pdb -sc 0.15 -o ion_solution
