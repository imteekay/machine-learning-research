def reverse_complement(s):
  complement = { 'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A' }
  return ''.join(complement[char] for char in reversed(s))