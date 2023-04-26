import pandas as panda

# Take the Path for the dataset.
Data_Path = "C:/Users/Raf/Downloads/Python/Python Project File's 2021/January 2021/Modify File/R40k_splice_feat.csv"
Phones_Path = "C:/Users/Raf/Downloads/Python/Python Project File's 2021/January 2021/Modify File/phones.txt"

# Read the Data dataset.
Data = panda.read_csv (Data_Path, header = None)
Data_Values = Data.values

# Take the Phones dataset
Phones_Data = panda.read_csv (Phones_Path, sep = " ", header = None)
Phones_Values = Phones_Data.values


# Separating data and targets.
Inputs = Data_Values [1:, 1:]

x = 0

# Change the Expected_Outputs of the Input dataset to numbers instead of Phones.
while x < Inputs.shape [0]:
    y = 0

    while y < Phones_Values.shape [0]:

        # If you find the correct value to replace the phone by then, replace it.
        if Inputs [x, -1] == Phones_Values [y, 0]:

            print ("\nPhone: {0}, Code: {1}".format (Inputs [x, -1], Phones_Values [y, -1]))
            Inputs[x, -1] = Phones_Values[y, -1]

            break
        y = y + 1

    print ("Line Number: {0}".format (x))
    x = x + 1


Modified_Inputs = panda.DataFrame (Inputs)
print ("\n\nHere is the New Inputs Matrix: \n\n{0}\n".format (Modified_Inputs))

print (panda.DataFrame (Inputs [20:21, 0:]))
print (panda.DataFrame (Inputs [30:31, 0:]))

Modified_Inputs.to_csv (r"C:/Users/Raf/Downloads/Python/Python Project File's 2021/January 2021/Modify File/R40K_Splice_Modified.csv", index = False, header = False)
