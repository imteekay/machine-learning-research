# https://rosalind.info/problems/iprb

def iprb(AA, Aa, aa):
  total = AA + Aa + aa
  AA_AA = (AA / total) * ((AA - 1) / (total - 1))
  AA_Aa = (AA / total) * (Aa / (total - 1))
  Aa_AA = (Aa / total) * (AA / (total - 1))
  Aa_Aa = (Aa / total) * ((Aa - 1) / (total - 1)) * 0.75 # 0.75 chance of possessing a dominant allele
  AA_aa = (AA / total) * (aa / (total - 1))
  aa_AA = (aa / total) * (AA / (total - 1))
  Aa_aa = (Aa / total) * (aa / (total - 1)) * 0.5 # 0.5 change of possing a dominant allele
  aa_Aa = (aa / total) * (Aa / (total - 1)) * 0.5 # 0.5 change of possing a dominant allele

  return AA_AA + AA_Aa + Aa_AA + Aa_Aa + AA_aa+ aa_AA + Aa_aa + aa_Aa

print(iprb(2, 2, 2)) # 0.7833333333333333
print(iprb(17, 19, 20)) # 0.725487012987013