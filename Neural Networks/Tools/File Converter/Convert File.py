import pandas as panda

Check_Data = True
Modify_Data = True

Zero = 0
One = 0
Two = 0
Three = 0
Four = 0

M_Zero = 0
M_One = 0

Path = "/home/raf_muz/Downloads/Python/Python Project File's 2021/January 2021/File Converter/reprocessed.hungarian.txt"
Data = panda.read_csv (Path, sep=" ", header=None)

NP_Data = Data.to_numpy ()


for x in Data.index:

    if Check_Data is True:

        if NP_Data [x, -1] == 0:

            Zero = Zero + 1


        if NP_Data [x, -1] == 1:

            One = One + 1


        if NP_Data [x, -1] == 2:

            Two = Two + 1


        if NP_Data [x, -1] == 3:

            Three = Three + 1


        if NP_Data [x, -1] == 4:

            Four = Four + 1


    if Modify_Data is True:

        if NP_Data [x, -1] == 0:

            M_Zero = M_Zero + 1


        if NP_Data [x, -1] >= 1:

            NP_Data [x, -1] = 1

            M_One = M_One + 1


print ("The Original Data Has \n\n{0} Zero's \n{1} One's \n{2} Two's \n{3} Three's \nAnd {4} Four's\n".format (Zero, One, Two, Three, Four))

print ("The Modified Data Has \n\n{0} Zero's \nand {1} One's\n".format (M_Zero, M_One))

Modified_Data = panda.DataFrame (NP_Data)


print (NP_Data)
print (Modified_Data)


Data.to_csv (r"/home/raf_muz/Downloads/Python/Python Project File's 2021/January 2021/File Converter/Normal Data.csv", index = False, header = False)

Modified_Data.to_csv (r"/home/raf_muz/Downloads/Python/Python Project File's 2021/January 2021/File Converter/Modified Data.csv", index = False, header = False)
