# https://rosalind.info/problems/fibd

def fibd(n, m):
  ages = [0] * m
  ages[0] = 1

  for _ in range(1, n):
    new_born = sum(ages[1:]) # The 0th generation can't reproduce

    # after the m-th-month, the rabbits die
    # shifting value will stop counting the died rabbits
    for i in range(m - 1, 0, -1):
      ages[i] = ages[i - 1]
    
    ages[0] = new_born

  return sum(ages)

print(fibd(84, 20)) # 160188618996844053