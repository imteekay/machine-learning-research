# Fifteen challenges for generative AI applications to cell biology

> Califano A (and possibly co-authors, not retrieved). *Fifteen challenges for generative AI applications to cell biology*. Cell (2026). [Cell](https://www.cell.com/cell/fulltext/S0092-8674(26)00802-0)
>
> cell.com returned HTTP 403. Only the Columbia news release (Aug 17, 2026) was read: it attributes a *Cell* perspective to Andrea Califano posing "15 critical, biologically relevant problems", inspired by Hilbert's 1900 list, and saying current biology AI models have not shown they can predict cell behavior. **The list of 15 challenges was not retrieved.** Author attribution is from the press release and should be confirmed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a Hilbert-style list of 15 open problems for generative AI in cell biology. **(paper, via press release)**
- **Why it matters for the virtual cell**: the roadmap says to pick PhD contributions from this list; it should be the checklist that each roadmap experiment maps onto. **(my take)**
- **Main claim**: current AI models in biology have not shown they can predict cell behavior; the community should focus on biologically relevant problems. **(paper, via press release)**

## 2. Problem it solves

Agenda-setting: too many benchmarks measure what is easy, not what matters biologically. **(paper, via press release)**

## 3. Method

Perspective. Califano's lab is known for regulatory-network / master-regulator methods (ARACNe, VIPER), so expect challenges about causal regulatory mechanisms and cell-state control. **(my take, unverified)**

## 4. Data

n/a.

## 5. Evaluation and results

n/a. The challenge list: not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): lists like this are useful only if each item comes with a falsifiable success criterion; check whether it does.
- What would change my mind: n/a until read.

## 7. Relevance to my research

- **Research proposal**: when the list is available, map each challenge to Master's H1 (epigenetic context → OOD) or PhD H (non-local response). Challenges nobody in the roadmap covers are candidate PhD contributions. **(my take)**
- **Roadmap**: all E-IDs, especially E4.1, E5.2, E6.3.
- To do when read: build a table (challenge → roadmap E-ID → falsifiable metric) and add uncovered challenges as new experiments.
- Hypothesis worth testing now:
  - **H-15C1 (E4.1 / E0.4)**: master-regulator activity (VIPER-style, inferred from RNA) as a context feature gives most of the OOD gain that ATAC gives. If true, ATAC's value is mostly regulator-state information, which changes the Master's story.

## 8. Related work

Bunne et al. 2024 (*Cell*, Califano is a co-author), Virtual Cell Challenge, [virtual-cells-need-context](virtual-cells-need-context.md).

## Resources

- [Columbia news: "It's Time to Teach AI Some Biology"](https://www.cuimc.columbia.edu/news/its-time-teach-ai-some-biology)
