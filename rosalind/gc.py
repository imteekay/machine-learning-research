# https://rosalind.info/problems/gc

def parse_fasta(data):
    sequences = {}
    current_label = None
    current_sequence = []
    
    for line in data.splitlines():
        if line.startswith('>'):
            if current_label is not None:
                sequences[current_label] = ''.join(current_sequence)
            current_label = line[1:].strip()
            current_sequence = []
        else:
            current_sequence.append(line.strip())
    
    if current_label is not None:
        sequences[current_label] = ''.join(current_sequence)
    
    return sequences

def gc_content(sequence):
    gc_count = sum(1 for base in sequence if base in 'GC')
    return (gc_count / len(sequence)) * 100

def highest_gc_content(sequences):
    highest_label = None
    highest_gc = 0.0
    
    for label, sequence in sequences.items():
        gc = gc_content(sequence)
        if gc > highest_gc:
            highest_label = label
            highest_gc = gc
    
    return highest_label, highest_gc

fasta_data = """>Rosalind_2798
CTATACCCACCCAGTCCGCTGTCGAAATATCTCATGGGCCCGTAGCGGCTCTCTCGTTAT
ACCGCTTTTAAGTACCCAGCACGTCATGTAGGGGGCCAGACCAATTAGATTTCGGTAGGA
TGATATTAATATGATCCAGCGCTACTCCCGGAAATTTTTATTATCTGGGACATACGCGCG
TACAGACGTTTCACGTTCAGCTCTCCTCCGGCCTGACGATGTTACATTTTAAACTTTAGG
GCTCAATGCGGCGTTGACATTATCTATGAGAATGTGCGCCCGCAAATTTGTGGATTCGCT
GACAAGGGCGCTTGACCCGACCCCGAGGCAGCCGTCATCTCGGTGTTTTTACATGTGGGA
GCCCAGTGCATACCACACAACACGGACGACGAGTCTATTCGCTCTTCAGATAAACTATGC
CCAAGCATATCCTAACTAACAACAACCTAGCCTTAACGCATACCAATAGTGGTTAACACA
AGCATGGCGCGTGATCGATAAGGTCTATCGTCCGGAGTAGTGGATGGCCATGCGCGACCT
GTTCCAGTAAAAAAAAAAATTCGGATCCAGTGAGTAATTCTGAACAAACATTGGCTAATA
CGATCCGTAAAAATTCGAGACGCATAAGCGTTTGGGAGCGTCTGACAAGATACCATCGGC
AACTCTGATTACAAGTCCTTTTTGGATCTTCTCCAAAAATAGCGTCACGGGTTCGGCGCA
ATTTGAAAACCGTTAGGACACGTTAGGGCGGTAAAGCCTGTATATGTGGTTAGGACTATC
TCACGCACGGAGACTGTAGCTTCGATCGACTGCGGGGTGCCGGTTGCCTAATGGGGCTTA
GTGCGTATCAGATACACGTAATTTCACGAGTTTTCTC
>Rosalind_5158
GTCTTACAGACATCCGGTGTTATCCATACTACGCGTTGGACAGTTCTTCAAGCACAAAGC
ATGGCGATCACGGGCGCTTGGACATACTCGGCCCGGTCTGGATAACTGCCGATATCATTA
AGGCGTTAAGGTTGTGGATGCGGAAGCGAGCTTAACATGAGCTGATCTTTGCGACACGAA
TGACGTAGCCTAACCCGTGGAGGCTAACCTCCTCATTTGGTATCCCGCTACAAGTTTTTA
GGGTTAATGAAGCGACGCCGAGCATAGCGAATGGATGAGTACAGGCTTTTCTGCATATCA
TTTGGGACCTCTTCTCTAGGGCTCGCCACGGACGGTATGAAGTAGGTGGATCCTTAGCCT
TATGTCGCAACGCTGCCCGCTTTCATATCCTTTCGCACATCGTAGAGGGTCCCGAAACGT
CGAGCGCCCACCGGGAAAGGCCGGGGATTAGACTTATGCAGATACCCGTATCGCGGGTAT
TATGCCGAGCTTGGGTGTTAAAACAACAGTTGCCAGCTGATCCACCTGAAATGAGTCAAG
GTGTATGGGGGTCATTTCTATCGGCGTGATGCCCGGGTCTAGTTAGCCCGTAGAGTAGGA
CAATCGGGAGAATGCACTGCCGGATGTAATAGGGACTGTTATGCGTTATCTGATATCCCC
TATTCCATCGGTAATCTAGGTCGTGGAATCAGTTTATTAAGATCCTAGGTCTACTCCACT
AGGGTACACCGTGAGCTCAACGCAAATACGGGAGGACAGTGGGCTGGCAAACACGCAGAC
CGGGGATTTCGTTGACGTTTAGCGACCCTAGCCCC
>Rosalind_8322
TACTCAAATGGGGCATTTTGATTGGTTTCAGATTATTTTCACTCAATCCCTGCCGGGCCC
AAGCCGTTTCCGTATTACATTTTGCTGGACTAATCTCATTGGTTCAGCTCTTGACAGTCT
ACTACCGAAGCGTTCAGTCTTAAGAGTACCGACCGTCACCACAGAATAGATTGCTGCTTG
TCTAAGGAAACGAGCAATCGGATGTTATAGGATAGTTCACTACCACCTGCGGTTTACTGG
CCACGGGCGTCAGCGGTGGAGCATGAACTTGAGACCCCCGTGTGTTACTTCGTCGGAACA
AACGGTGCACTACTGATGTTTGAATCACGCTAGTGATTACCATATGGCTGCTTTCCAGGG
TACGTGAGGCCTTGAACCGTAGGCATGCTTACCTTAATACCAGGTACTTTCAGGTCGCAC
TTCAACAATGGACCTATTCAAATGGTACTAACGGTTGTGACGTGACGCAATGCAGAGCTA
AGGTCGTTCAAAATCTTAGCCTAATAGAAATTGTTGCCCTCACGCTGGTTGGCGTGATCC
ATAAAGTCCGGGCCTTACACGAACGTTGACCACCATACTGCGGCCGATCTTACCCTGAAT
GACCGCCAAACCGTTGCATCATGCCTGGGTGCGCTCGACAAGGTCCTTAGGTTTTTATCC
AGAGGCCCGAAGGAGGGATCGTGACTGGAGAGCATTATCAGTCACACTACTGTAAGCACA
AAAGTCGACCAACCCATCTTTCCTACTTTTTTCTTCGAGAACATATCCTATACGCCTCAG
ATCCGCTGATGAGAGCTTCGTTCGATTGGAGTTCTACCATCTGCGCATACCCAGGTCTAG
TAGAGGGTTCAGTCCCATCCCGGACGTCGGGACTAACCCTACGACGGTCCATATTCCACA
CTTTGGGGAGACCTAGTTCGCTCCCAGGGGGTTCGTGTTGAGGTATAGTAATGGTGCGGG
CCTGCCCG
>Rosalind_4174
GCGACTAGGCAGTGGGGCCACTCAATGGCAAAATGCTTCAGACGTGGTCCATAGCATTAA
GCATATTACTCCCAATGCCAGAAATTTCCCGCGGACTCTTAACCTGCTGAAAACACCGTA
TGGTCGAATTACCGCTCAATAGGCATCCACCTTTTAAGATGATCTTTGTCTCTCAATAAC
GAACAGCGCGGCTCATAGGTAGACAATACGGTCTCGCTGCGGAAATTAGATACTGTTAGC
TTCATGTGGCAGTCCGTGTTACGTTATAACGCTGACTACCGACGCCTGCCACAATCATGT
CAACGGGCCTGTACGAGCCTCGAACTGTGGGGCCCAGGGGTAAGTGCAACAGCTCTTAAC
AGCCTCATAATGATTGCGCTGTCCTTTGAAACCGAGTCGATACCTCTCAACGTAGTCCTG
CGCCGGCGTCTCTCGTGGTACTAGAAGCGCAATATCGCTGGGTCTGATCATTGTCATCGT
TCAAGAATCCTAAATTCCTCGGGGCCAACGTGAAGTAGACTATTACAGGAGTCAGCCGCA
TTATCGGGAAGCGAATCTGCAGGAAACGGACAGGATTACCACCTATCTTGTGTAATAAGT
GTGTTTTAAGAGATCAGCTATGTATTGGGCTTCGTGACCAGCTGACATGAACCTGGCCCT
AACCCCCACTGCAATAGTAACAGGAAACTCGTCTCGCGATGGGCCCACAATATTAGAGTA
ACCCCGTGACAATCGTACATAATAGTCGAATCACGTACACCGCTCGCAGTTCCGACGAGG
TATCTTAGTATTTTACGGACCGAGCCGCATCAATCCCGTGAGCCTCTATATTCATGAACG
G
>Rosalind_8464
ACAGTGATTTGCGAAAGACTCCACCCCGTTATGACATCGGCCACCATCTTACAACTCGGC
AGGTGCCTTACATCCTGTTAAGTTTGATCGCGCATCACGCGATATAAATGAGAATGTCCT
GCAAGATCCCGCTCGACGACTGGCTGGTACTATGTTGAGCAGTGCCACTGCACTAGCCCT
TCTGGATGGCCTGGCCAGGGCGTGATCCAAAAGCGTCTACGCCAACGCCATCGCGGAGGT
GCGTCCGCGTTGCCCTGCTCCACCACTATCATCATAATAGCAGCTCTGCTTAGAGTCCGC
CGGTGTTAACGATCAGTTCGAGATCACTATGGGATCTCGATGCCATAACACGCGCCAGCA
TAAGGCGATTACGGCGTCAGAATTGCAACAGAATACCGCCTAGTTGACGGTGCTCATATG
GATCCAACGGATTACGACTATCGATAGTCATTGCACAGTCTAGCCGAGGTCTGTTCTTAT
TACACCATCCCTCCATTACTTTAAAAGTCTGGAGCGCATCGTGACATGCGTTGAATGGTG
GTAGTTGGCGGTAGTGGCCCCGGCCTCGCGTCGCGTTGTGCCACGCTGCAAGAAACCAGT
GTATGTGGGCCGAAATTCAGCACACTCCTCGATTTGTCCCGACGACATGGGCGTTAAATA
TAGACCTATGCCGAGATTCTTTTTTGATGGTCTGCTTCTTGGGACGTGCACGCCCTTAGA
TGTATCGCTCACTCAAAAATACGGACGAGAAAGCATCCCCAAACTCACGTAGTTAAAGTC
GCCCGCATTGATTAGCCGTCCTCAGAGTTCCCAGTGCCAGCGAAAGAGCTCGAGTAGTAG
ATGAACCCCTGAGTTAGTCCCAGTAATAATAGCGAGGCTCCGGTCGTGCGTTAAATATGT
TGTACGAGCAAATACTTAGAGAGTGTAGCAGCACATTAAATTCTCGTGTAACTAGTCGGG
>Rosalind_5156
GTAGGAGAACTTTCATTGCCTCTCATCGGCGGTCTACAAGGAATATCTTAATAAATGCTG
AGCACGATGTCCAGGAAGGGAAAACTAGGATTACTGAAGCTGGAGAAAGCACCGGCATTA
ACGCCCTTTCAATCCCTCGATTGATCCATGAGTGGGTTCCGTGCTGCGCCCCCTACAATT
TAGATTCCGTAGGTTCATTCCAGGGGTGCTAATGGCTAAATGCTCGTGTAGAAACCTCTG
TACTCAGACACGTAGGTCGGAAGCGTTCTCGAATAGTCTGCTACAAACTAAGCGTAAAGC
ACGATTATGATAGGTGCCTAACAACAAACACCTATGCTCAGACGTAAGTGGCGGTTGGCT
AGTCGTATTGGTTGTAATTAAATTTGCAGTTACCCAGCATCTGAACCGCCTAGCTCCTTG
GGATCGGCGCACCAGCTGCCAGTATGCCGTAGGCGACTCGTGTGACGCCTCAGTGCCGAA
GATGTTATCGTTTATTAGGTGTAGACTGCAGCGCGGACCTCGCGGGTTTATCCGATGTCT
TGGCAATCCGCTCCCTGTAGCGGTTTAGCATGGTAGTGTCAGAAGATAAACCATTATTAC
ATAAAGCCTGGGCGCTTCGCTGTGGTACGCCACTGCGTTAGCAGTTTCTTCGTTTCAGTT
TGAACTATCGTAAATGTTCCTAAGACTAGCGCCATCCGTCGACGAATTTAGATCCTGCTT
TTCCAGAACCTCCCCCTCTATGTAGATTATTCGTTAACACGTATCTTGGAGCAACGGCTG
TTCTGTCGCATGACGACTACGCCCGCTTTTAGCCTTACGATTTCTCAGTAGGGTCTAGTG
CTTAGCTGTGTGTGATTCCCACCGAAATAGTCGCCTTCGAGCCCTAATTCGAATAGGGGA
GGTTAGGCTCTGCCGGCTGTTGACGCTTGGTATGTGTCG
"""

# Parse the input data
sequences = parse_fasta(fasta_data)

# Find the sequence with the highest GC content
label, gc_content = highest_gc_content(sequences)

# Print the result
print(f"{label}\n{gc_content:.6f}")
