import pickle

with open("my-saved-model.pt", "rb") as file:
            Qtable = pickle.load(file)
print(Qtable[3,7,:])
print(Qtable[3,8,:])