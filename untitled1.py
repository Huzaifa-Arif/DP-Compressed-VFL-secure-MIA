import pickle
  
# Open the file in binary mode
with open('file.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar = pickle.load(file)
  
    print(myvar)


