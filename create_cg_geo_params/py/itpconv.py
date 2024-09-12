import sys

cg_itp_type   = ['[ atoms ]', '[ bonds ]', '; Backbone bonds', '; Sidechain bonds', '; Short elastic bonds for extended regions', '; Long elastic bonds for extended regions', '[ constraints ]', '[ angles ]', '; Backbone angles', '; Backbone-sidechain angles', '; Sidechain angles', '[ dihedrals ]', '; Backbone dihedrals', '; Sidechain improper dihedrals', '#ifdef POSRES', '#endif', '#endif']
ss_type = ['; Sequence:','; Secondary Structure:',' ']

def process_file(file_path):
    rl_o = ""
    bond = ""
    output_lines = []
    with open(file_path, 'r') as file:
        for rl in file:
            if rl_o[:-1] == "[ bonds ]":
                b_short = "no"
                b_long = "no"
                bond = "yes"
                if rl[:-1] == "; Sidechain bonds":
                    output_lines.append("; Backbone bonds\n")
            if rl_o[:-1] == "[ constraints ]":
                bond = "no"
            if rl[:-1] == "; Short elastic bonds for extended regions":
                b_short = "yes"
            if rl[:-1] == "; Long elastic bonds for extended regions":
                b_long = "yes"
            if rl[:-1] == "" and bond == "yes":
                if b_short == "no":
                    output_lines.append("; Short elastic bonds for extended regions\n")
                if b_long == "no":
                    output_lines.append("; Long elastic bonds for extended regions\n")
            elif rl_o[:-1] == "[ bonds ]" and rl[:-1] == "":
                output_lines.append("; Backbone bonds\n")
                output_lines.append("; Sidechain bonds\n")
            elif rl_o[:-1] == "; Backbone bonds" and rl[:-1] == "":
                output_lines.append("; Sidechain bonds\n")
            output_lines.append(rl)
            rl_o = rl
    with open(file_path, 'w') as file:
        file.writelines(output_lines)

def extract_content(filename, itp_type):
    itp_content = []
    for i in range(len(itp_type)-1):
        start_line = itp_type[i]
        end_line   = itp_type[i+1]
        content    = []
        #content.append(";; " + start_line)
        is_extracting = False
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line == start_line:
                   is_extracting = True
                elif line == end_line:
                   is_extracting = False
                elif is_extracting:
                   content.append(line)
        itp_content.append(content)
    return itp_content

file_num = len(sys.argv[1:])
itp_data = []
seq_data = ""
ss_data = ""
#for i in range(1, file_num+1):
#    cg_itp = sys.argv[i + 1]
for cg_itp in sys.argv[1:]:
    if __name__ == "__main__":
        process_file(cg_itp)
    cg_itp_content  = extract_content(cg_itp, cg_itp_type)
    seq_itp_content  = extract_content(cg_itp, ss_type)[0][0].split()[1]
    ss_itp_content = extract_content(cg_itp, ss_type)[1][0].split()[1]
    seq_data = seq_data + seq_itp_content
    ss_data  = ss_data + ss_itp_content
    itp_data.append(cg_itp_content)


print("; Sequence:")
print('''; %s'''%seq_data)
print('''; Secondary Structure:''')
print('''; %s'''%ss_data)
print("")

#print(len(cg_itp_type))
#print(len(itp_data[0]))

atom_start  = [0]
resid_start = [0]
atom_index  = 0
resid_index = 0
for i in range(file_num):
    atom_sln    = itp_data[i][0][-2].split()[:]
    atom_index  += int(atom_sln[0])
    resid_index += int(atom_sln[2])
    atom_start.append(atom_index)
    resid_start.append(resid_index)
#print(atom_start,resid_start)

#atom_line     = "%7s %5s %5s %5s %5s %7s %9s  %2s  %s"           ## atoms
atom_line     = "%7s %5s %5s %5s %5s %7s %9s"                    ## atoms
bond_line     = "%5s %7s   %s "                                  ## bonds, Backbone bonds, Sidechain bonds, constraints
angle_line    = "%5s %5s %5s   %s "                              ## angles, Backbone angles, Backbone-sidechain angles, Sidechain angles
dihedral_line = "%5s %5s %5s %5s   %s  "                          ## dihedrals, Backbone dihedrals, Sidechain improper dihedrals
posre_line    = "%6s     1  POSRES_FC    POSRES_FC    POSRES_FC" ## endif 


def get_new_itp(itp_content, itp_type, atom_start, res_start):
    new_itp_content = []
    new_itp_content.append(itp_type)
    if itp_type == "[ atoms ]":
       for line in itp_content:
           srl = line.split()[:]
           if len(line) != 0:
              if str(line[0]) not in [";", "#"]:
                 a1 = int(srl[0]) + int(atom_start)
                 r1 = int(srl[2]) + int(res_start)
                 #new_itp_content.append(atom_line%(a1, srl[1], r1, srl[3], srl[4], a1, srl[6], srl[7], ' '.join(srl[8:])))
                 new_itp_content.append(atom_line%(a1, srl[1], r1, srl[3], srl[4], a1, srl[6]))
              else:
                 new_itp_content.append(line)

    elif itp_type == "[ bonds ]" or itp_type == "; Backbone bonds" or itp_type == "; Sidechain bonds" or itp_type == "[ constraints ]" or itp_type == "[ exclusions ]" or itp_type == "[ pairs ]" or itp_type == "; Short elastic bonds for extended regions" or itp_type == "; Long elastic bonds for extended regions":
       for line in itp_content:
           srl = line.split()[:]
           if len(line) != 0:
              if str(line[0]) not in [";", "#"]:
                 a1 = int(srl[0]) + int(atom_start)
                 a2 = int(srl[1]) + int(atom_start)
                 new_itp_content.append(bond_line%(a1, a2, ' '.join(srl[2:])))
              else:
                 new_itp_content.append(line)

    elif itp_type == "[ angles ]" or itp_type == "; Backbone angles" or itp_type == "; Backbone-sidechain angles" or itp_type == "; Sidechain angles":
       for line in itp_content:
           srl = line.split()[:]
           if len(line) != 0:
              if str(line[0]) not in [";", "#"]:
                 a1 = int(srl[0]) + int(atom_start)
                 a2 = int(srl[1]) + int(atom_start)
                 a3 = int(srl[2]) + int(atom_start)
                 new_itp_content.append(angle_line%(a1, a2, a3, ' '.join(srl[3:])))
              else:
                 new_itp_content.append(line)

    elif itp_type == "[ dihedrals ]" or itp_type == "; Backbone dihedrals" or itp_type == "; Sidechain improper dihedrals":
       for line in itp_content:
           srl = line.split()[:]
           if len(line) != 0:
              if str(line[0]) not in [";", "#", "["]:
                 a1 = int(srl[0]) + int(atom_start)
                 a2 = int(srl[1]) + int(atom_start)
                 a3 = int(srl[2]) + int(atom_start)
                 a4 = int(srl[3]) + int(atom_start)
                 new_itp_content.append(dihedral_line%(a1, a2, a3, a4, ' '.join(srl[4:])))
              else:
                 new_itp_content.append(line)       
    elif itp_type == "#ifdef POSRES" or itp_type == "#endif":
       for line in itp_content:
           srl = line.split()[:]
           if len(line) != 0:
              if str(line[0]) not in [";", "#", "[", " "]:
                 a1 = int(srl[0]) + int(atom_start)
                 new_itp_content.append(posre_line%(a1))
              else:
                 new_itp_content.append(line)
    return new_itp_content

### writing pacem.itp 
###
print("[ moleculetype ]")
print("; Name         Exclusions")
print("Protein            1\n")

def writeItp(itp_type, itp_type_index_num):
    for i in range(len(itp_data)):
        ln = itp_data[i][itp_type_index_num]
        new_ln = get_new_itp(ln, itp_type, atom_start[i], resid_start[i])
        for line in new_ln:
            if len(line) != 0 and line[0] not in ["[", ";", "#"]:
               print(line)
    return

itp_type_index_num = 0
for itp_type in cg_itp_type[:-1]:
    if itp_type in ['[ atoms ]', '[ bonds ]', '[ constraints ]', '[ angles ]', '[ dihedrals ]', '#ifdef POSRES']:
        print(" ")
    print(itp_type)
    if itp_type in ['#ifdef POSRES']:
        print("#ifndef POSRES_FC")
        print("#define POSRES_FC 1000.00")
    new_itp = writeItp(itp_type, itp_type_index_num)
    itp_type_index_num += 1 
    if itp_type in ['#endif']:
        print("#endif")



