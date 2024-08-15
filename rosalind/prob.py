import math

def prob(dna_string, A):
  B = []

  for gc_content in A:
    gc_probability = gc_content / 2
    at_probability = (1 - gc_content) / 2

    string_probability = 1

    for nucleotide in dna_string:
      if nucleotide in 'GC': string_probability *= gc_probability
      else: string_probability *= at_probability

    B.append(math.log10(string_probability))

  return B

DNA_STRING = 'ACGATACAA'
A = [0.129, 0.287, 0.423, 0.476, 0.641, 0.742, 0.783]


print(prob('GTATCGAGCCAACCCATTTCGCGCAGTGAGGGACCAGCCCCGCTCGCTTAATCTGCTCCTTTCCTGCAACGATTCAAGTGAAGATGCATCTGTTC', [0.076, 0.114, 0.177, 0.226, 0.286, 0.375, 0.424, 0.466, 0.544, 0.571, 0.640, 0.715, 0.725, 0.812, 0.840, 0.944]))
# -87.18678964807316 -79.00861840688788 -70.6736202542862 -66.4337169215135 -62.76046058034535 -59.303534170825245 -58.14360163122752 -57.498352527631724 -57.0878526262359 -57.1812872114121 -58.00536088113933 -60.01506761584865 -60.38997272442598 -65.14754672917637 -67.47832676206156 -84.95400469060387