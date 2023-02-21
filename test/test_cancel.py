import os

for i in range(1000000):
    try:
        print(i)
    except EOFError as e:
        break

print(f'Stopped at {i} < 1000000')