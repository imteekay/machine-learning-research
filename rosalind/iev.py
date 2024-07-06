# https://rosalind.info/problems/iev

def parse_couples(string):
  return list(map(int, string.split(' ')))

def iev(string):
  OFFSPRINGS = 2
  couples = parse_couples(string)
  offsprings_with_dominant_phenotype = 0

  for index in range(len(couples)):
    if couples[index] > 0:
      if index in [0, 1, 2]: offsprings_with_dominant_phenotype += couples[index]
      if index == 3: offsprings_with_dominant_phenotype += couples[index] * 0.75
      if index == 4: offsprings_with_dominant_phenotype += couples[index] * 0.5

  return OFFSPRINGS * offsprings_with_dominant_phenotype

def iev_with_zip(string):
  OFFSPRINGS = 2
  couples = parse_couples(string)
  offsprings_with_dominant_phenotype = 0
  probabilities = [1, 1, 1, 0.75, 0.5, 0]

  for count, probability in zip(couples, probabilities):
    offsprings_with_dominant_phenotype += count * probability * OFFSPRINGS

  return offsprings_with_dominant_phenotype

print(iev('1 0 0 1 0 1'))
print(iev('16693 19885 19597 17365 18267 19204'))

print(iev_with_zip('1 0 0 1 0 1'))
print(iev_with_zip('16693 19885 19597 17365 18267 19204'))