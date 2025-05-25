# Algorithms for DNA Sequencing

Next Generation Sequencing (NGS) in 2007 (second generation sequencing or massive parallel sequencing)

![](images/001.png)

Get the sequence of DNA as a string getting the side of the double helix at a time

- double helix
- base pairs: A-T and C-G

![](images/002.png)

DNA sequencing

- short snippets of DNA (reads): genome is orders of magnitude larger than these shorts snippets
  - chromosomes (1.000.000) vs reads (100)
- use massive parallel sequencing

## String definitions

- String `S` is a finite sequence of characters and { A, T, C, G }
- number of characters using `len(S)`
- empty string when `len(S)` is 0

If we have double stranded DNA, and we knew the sequence along one of those strands from top to bottom, the reverse complement would give us the sequence of the other strand from bottom to top:

```python
def reverse_complement(s):
  complement = { 'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A' }
  return ''.join(complement[char] for char in reversed(s))
```

## How DNA gets copied

Second-generation sequencing, also known as sequencing by synthesis, mimics the natural process of DNA replication. During cell division, the double-stranded DNA unwinds and separates into two single strands, each carrying the genetic sequence.

These single strands act as templates for creating new complementary strands. An enzyme called DNA polymerase facilitates this process by adding the appropriate complementary base to the template strand. For instance, if the template has a G, the polymerase will add a C, and if it has an A, it will add a T. This step-by-step addition of bases by DNA polymerase results in two identical double-stranded copies of the original DNA molecule.

Second-generation sequencing leverages this natural DNA copying mechanism. By observing the activity of the DNA polymerase as it builds the new complementary strand, scientists can deduce the sequence of the template DNA molecule. Importantly, this process is performed simultaneously on millions or even billions of template molecules, allowing for high-throughput sequencing.

## How second-generation sequencers work

High-throughput DNA sequencing method

- DNA sample is fragmented and made single-stranded, resulting in many short single-stranded templates that are then attached to a flat surface (a slide) at random locations. Next, 
- DNA polymerase and modified bases with terminators (and fluorescent labels) are added. The polymerase adds only one complementary base to each template because the terminator prevents further extension. 
- A photo is taken, capturing the fluorescent signal that identifies the incorporated base for each template.
  - Template strands are amplified to create clusters of identical copies. This amplification is necessary to generate enough light for detection when fluorescently labeled terminator bases are incorporated. However, errors can occur when a non-terminated base is accidentally added during a sequencing cycle. This causes some strands within a cluster to become out of sync with others, leading to mixed signals (multiple colors of light) during imaging.
- The terminators are then removed, and the process is repeated. 
  - Sometimes a base isn't terminated so it doesn't block the polymerase and the polymerase keeps going. Another base is added on top of the current base and one of the templates is out of sync with the others: there is not just one color when taking the photo
- By taking a series of photos over multiple cycles, the sequence of each individual template can be determined by tracking the color changes at its specific location on the slide. This method allows for the simultaneous sequencing of billions of templates, making it a massively parallel process. 
  - Software called a "base caller" analyzes these images to determine the sequence. To account for the uncertainty arising from mixed signals, the base caller assigns a "base quality" score (Q) to each base call.
  - The base quality information is crucial for downstream analysis, allowing researchers to identify and account for potentially unreliable base calls
- Key aspects include attaching billions of templates to a single slide, photographing them simultaneously, and using terminators to control the polymerization and enable base identification through fluorescence.

## Sequencing reads in FASTQ format

- A FASTQ file has many reads and each read has a format of `name`, `sequence`, `placeholder`, and `base quality`
  - Read Name: Contains information about the experiment but is generally ignored for downstream analysis.
  - Sequence: The actual DNA sequence reported by the base caller. This is the crucial information.
  - Placeholder Line: This line is typically ignored.
  - Base Quality Line: A string of characters that encodes the quality score for each corresponding base in the sequence line.
- The base quality is represented by a value Q, which is related to the probability P of an incorrect base call (higher Q means higher confidence). These Q values are ASCII-encoded into characters. The specific encoding used in the class is Phred 33.

```python
def QtoPhred33(Q):
  """ Turn Q into Phred+33 ASCII-encoded quality """
  return chr(Q + 33)

def phred33toQ(qual):
  """ Turn Phred+33 ASCII-encoded quality into Q """
  return ord(qual) - 33
```
