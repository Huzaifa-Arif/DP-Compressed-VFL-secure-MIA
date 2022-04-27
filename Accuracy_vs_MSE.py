import pickle 
import matplotlib.pyplot as plt
import numpy as np
#a.detach().to('cpu').numpy(

# #quantizatio
# #for i in range():
#     temp = "acc_"+"var"+"0_"+"det_"+"2.pkl"  
#     # Open the file in binary mode
#     with open('file.pkl', 'rb') as file:
#         # Call load method to deserialze
#         myvar = pickle.load(file)
  
#         print(myvar)
quantization = [2,3,4,5,6,7,8,9,10]
#quantization = [9,10]
quant_ratio = (1/32)* np.log2(quantization)
var = "0"
accuracy = []
for i in quantization:
   
    temp = "acc_var"+ var +"_"+"det_"+str(i)+".pkl" 
    with open(temp, 'rb') as file:
    #         # Call load method to deserialze
             myvar = pickle.load(file)
    accuracy.append(float(myvar[-1]))
    #print(float(myvar[-1]))## Reading a 

plt.plot(quantization,accuracy,"-*")

plt.ylabel("Accuracy")
plt.xlabel("Quantization")
plt.title ("Low Quantization Levels - Accuracy - No Noise")

plt.show()

with open('MSE_det.pkl', 'rb') as file:
## Call load method to deserialze
  error_mean , error_std = pickle.load(file)
  
plt.plot(quantization,error_mean,"-*")

plt.ylabel("MSE")
plt.xlabel("Quantization")
plt.title ("Low Quantization Levels - MSE -  No Noise")
plt.show()
#print(error_mean)
# #print(error_std)
# fig, (ax1,ax2) = plt.subplots(1,2,sharex='all')
# #fig.canvas.draw()
# #labels = quantizationax.set_xticks
# ax1.set_xticks(quantization)
# ax1.plot(quantization,accuracy)
# ax2.plot(quantization,error_mean)
# ax2.errorbar(quant_ratio,error_mean) #,yerr= error_std ,color="blue",elinewidth=0.5,barsabove=True)

#ax2.set_xticklabels(labels)


# ax[0].plt.xlabel("Quantization_Level")
# #ax.xticks( quantization)
# ax[0].ylabel("MSE error")
# ax[0].title("Scalar sparsification with Binomial_noise" )



plt.show()