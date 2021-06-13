# Guessing Game Challenge
# Let's use while loops to create a guessing game.

# The Challenge:

# Write a program that picks a random integer from 1 to 100, and has players guess the number. The rules are:

# If a player's guess is less than 1 or greater than 100, say "OUT OF BOUNDS"
# On a player's first turn, if their guess is
# within 10 of the number, return "WARM!"
# further than 10 away from the number, return "COLD!"
# On all subsequent turns, if a guess is
# closer to the number than the previous guess return "WARMER!"
# farther from the number than the previous guess, return "COLDER!"
# When the player's guess equals the number, tell them they've guessed correctly and how many guesses it took!
#%%
import numpy as np
x = 0
int_val = np.random.randint(1,8)
print(int_val)
def guess_game(value):
    global x
    global int_val
    if value < 1 or value >100 : 
        return 'OUT OF BOUND'
    if abs(value-int_val) <= 10 :
        return 'Warm!'
    elif abs(value-int_val) <= 10 :
        return 'Cold!'
    if(abs(value-int_val)>abs(x-int_val)):
        return 'colder!'
    elif(abs(value-int_val)<abs(x-int_val)):
        return 'warmer!'
    if(value == int_val):
        return False
    x = value

y = True
z = 0
while y:
    value = input('Enter the number: ')
    final_value = guess_game(int(value))
    z = z+1
    if(final_value==False):
        y = False
        print(f'you have guessed correctly and the guessses u took are {z}')
        break
    else:
        print(final_value)
    




    
# %%
value = input('asdad:')
# %%
