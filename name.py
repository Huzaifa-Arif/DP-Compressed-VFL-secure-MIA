import os

folder = r'/home/arifh/Documents/Research/checkpoints/Accuracy/Quantization1/Stochastic'





# iterate all files from a directory
count = 0
ty_check = 1
qt = 1
for file_name in sorted(os.listdir(folder)):
    # Construct old file name
    source = os.path.join(folder,file_name) #folder + file_name
    if(ty_check):
        ty = "stoch"
    else:
        ty = "det" 
    # Adding the count to the new file name and extension
    #destination = folder + "sales_" + str(count) + ".txt"
    emp_str = []
    #print(source)
    #print(file_name)
    for m in file_name:
        if m.isdigit():
            emp_str.append(m)
    print(source)
    destination = os.path.join(folder, "acc_var"+ str(count)+"_" + ty +"_" +str(qt)+".pkl")
    count+=1
    print(destination)
    #break
    
   
    
    # Renaming the file
    os.rename(source, destination)
    #count += 1
#print('All Files Renamed')

#print('New Names are')
# verify the result
res = sorted(os.listdir(folder))
print(res)