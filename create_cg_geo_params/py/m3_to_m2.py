import sys


posit = "no"
bs = "no"
ex = "yes"
dele_type = ["#ifdef FLEXIBLE","[ exclusions ]","[ virtual_sitesn ]"]

wrl = "yes"

m3_to_m2 = {"#ifndef FLEXIBLE": " ", "#endif": " ", "; Side chain bonds": "; Sidechain bonds", "; BBB angles": "; Backbone angles", "; BBS angles regular martini": "; Backbone-sidechain angles"}

dih_num = 0
cap = "no"
prl_o = ""
itp_data = []
for rl in open(sys.argv[1],'r'):
    if rl[:-1] in dele_type:
        wrl = "no"
    elif rl[:-1] == "":
        wrl = "yes"
    if wrl == "yes":
        prl = rl[:-1]
        
        if prl in ["#ifndef FLEXIBLE", "[ constraints ]"]:
            cap = "yes"
        if prl in ["#endif", "[ angles ]"]:
            cap = "no"
        
        if prl in ["#ifndef FLEXIBLE","#endif","; First SBB regular martini"]:
            prl = " "
        
        if prl == "; Backbone bonds" and cap == "yes":
            prl = " "
        if prl == "; Side chain bonds" and cap == "yes":
            prl = " "
        if prl == "; Side chain bonds" and cap == "no":
            prl = "; Sidechain bonds"
        
        if prl == "; BBB angles":
            prl = "; Backbone angles"
        if prl == "; BBS angles regular martini":
            prl = "; Backbone-sidechain angles"
        if prl == "; Side chain angles":
            prl = "; Sidechain angles"
        
        if prl_o == "[ dihedrals ]":
            if prl.split()[4] == "1" and prl.split()[5] == "-120" and prl.split()[6] == "400":
                itp_data.append("; Backbone dihedrals")
            elif prl.split()[4] == "2":
                itp_data.append("; Sidechain improper dihedrals")

        if prl == "[ dihedrals ]":
            if dih_num != 0:
                prl = " "
            dih_num += 1
        itp_data.append(prl)
        prl_o = rl[:-1]

def fill_missing_lines(input_file, output_file, key_lines):
    itp_data = []
    existing_lines = [line.strip() for line in input_file]
    
    output_lines = []
    missing_lines = []
    existing_lines_iter = iter(existing_lines)  
    next_line = next(existing_lines_iter, None)

    for expected_line in key_lines:
        if expected_line in existing_lines:
            while next_line != expected_line:
                if next_line is not None:
                    output_lines.append(next_line)
                    next_line = next(existing_lines_iter, None)
                else:
                    break
            
            output_lines.extend(missing_lines)
            missing_lines = []  
            output_lines.append(expected_line)  
            next_line = next(existing_lines_iter, None)
        else:
            missing_lines.append(expected_line)

    while next_line is not None:
        output_lines.append(next_line)
        next_line = next(existing_lines_iter, None)

    output_lines.extend(missing_lines)

    for line in output_lines:
        itp_data.append(line)
    itp_data.append(" ")
    return itp_data

def getSeqSS(file):
    rl_o = " "
    for rl in open(file,'r'):
        if rl_o == "; Sequence:":
            print("; Sequence:")
            print(rl[:-1])
        if rl_o == "; Secondary Structure:":
            print("; Secondary Structure:")
            print(rl[:-1])
        rl_o = rl[:-1]

getSeqSS(sys.argv[2])
print()

key_types = ["[ moleculetype ]", "[ atoms ]", "[ position_restraints ]", "[ bonds ]", "; Backbone bonds", "; Long elastic bonds for extended regions", "; Short elastic bonds for extended regions", "; Sidechain bonds", "[ constraints ]", "[ angles ]", "; Backbone angles", "; Backbone-sidechain angles", "; Sidechain angles", "[ dihedrals ]", "; Backbone dihedrals", "; Sidechain improper dihedrals", " "]

m3_to_m2_itp = fill_missing_lines(itp_data, 'output.itp', key_types)


def extract_content(files, itp_type):
    itp_content = []
    for i in range(len(itp_type)-1):
        start_line = itp_type[i]
        end_line   = itp_type[i+1]
        content    = []
        is_extracting = False
        for line in files:
                line = line.strip()
                if line == start_line:
                   is_extracting = True
                elif line == end_line:
                   is_extracting = False
                elif is_extracting:
                   content.append(line)
        itp_content.append(content)
    return itp_content

new_itp = extract_content(m3_to_m2_itp, key_types)
itp_dict = dict(zip(key_types, new_itp))

key_type_re_order = ["[ moleculetype ]", "[ atoms ]", "[ bonds ]", "; Backbone bonds", "; Sidechain bonds", "; Short elastic bonds for extended regions", "; Long elastic bonds for extended regions", "[ constraints ]", "[ angles ]", "; Backbone angles", "; Backbone-sidechain angles", "; Sidechain angles", "[ dihedrals ]", "; Backbone dihedrals", "; Sidechain improper dihedrals", "[ position_restraints ]"]

for itp_type in key_type_re_order:
    lines = itp_dict[itp_type]
    if itp_type == "[ position_restraints ]":
        print('''#ifdef POSRES\n#ifndef POSRES_FC\n#define POSRES_FC 1000.00\n#endif\n[ position_restraints ]''')
    else:
        print(itp_type)
    if len(lines) != 0:
        for line in lines:
            if line != "#ifdef POSRES" and len(line) != 0:
                print(line)
    if itp_type == "[ position_restraints ]":
        print("#endif")


