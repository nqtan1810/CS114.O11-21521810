fo_1 = open("D:\CS114\project\dataset.txt", 'r')
fo_2 = open("D:\CS114\project\handled_dataset.txt", 'w')
string = fo_1.readlines()
for i in string:
    j = i.split()
    if len(j) > 1:
        output_string = "\t".join(j[1:])
        fo_2.write(output_string+'\n')
        #print(output_string)