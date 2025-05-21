import os
import collections

def read_genome(filename):
  genome = []
  filename_path = os.path.join('courses', 'genomic-data-science', 'algorithms-for-dna-sequencing', 'fasta', filename)

  with open(filename_path, 'r') as file:
    for line in file:
      if not line[0] == '>':
        genome.append(line.rstrip())

  return ''.join(genome)

genome = read_genome('lambda_virus.fa')

print(genome)
print(len(genome))

counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

for base in genome:
  counts[base] += 1

print(counts)
print(collections.Counter(genome))
