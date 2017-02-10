# Multi-tissue polygenic models for transcriptome-wide association studies

## Contributors

Yongjin Park, Abhishek Sarkar, Kunal Bhutani, Manolis Kellis

Maintained by `ypp@csail.mit.edu`

## Abstract

Transcriptome-wide association studies (TWAS) have proven to be a
powerful tool to identify genes associated with human diseases by
aggregating _cis_-regulatory effects on gene expression. However, TWAS
relies on building predictive models of gene expression, which are
sensitive to the sample size and tissue on which they are trained. The
Gene Tissue Expression Project has produced reference transcriptomes
across 53 human tissues and cell types; however, the data is highly
sparse, making it difficult to build polygenic models in relevant
tissues for TWAS. Here, we propose `fQTL`, a multi-tissue,
multivariate model for mapping expression quantitative trait loci and
predicting gene expression. Our model decomposes eQTL effects into
SNP-specific and tissue-specific components, pooling information
across relevant tissues to effectively boost sample sizes. In
simulation, we demonstrate that our multi-tissue approach outperforms
single-tissue approaches in identifying causal eQTLs and tissues of
action. Using our method, we fit polygenic models for 13,461 genes,
characterized the tissue-specificity of the learned _cis_-eQTLs, and
performed TWAS for Alzheimer's disease and schizophrenia, identifying
107 and 382 associated genes, respectively.

## Installation

Install R package:

```
library(devtools)
library(Rcpp)
install_github("ypark/fqtl")
```

## Results of GTEx TWAS

Tab-separate files of full TWAS statistics on Alzheimer's disease (AD)
and schizophrenia (SCZ).

```
GTEx/ad-twas-full.txt.gz
GTEx/scz-twas-full.txt.gz
```

* tis : GTEx tissues
* chr : chromosome
* tss : transcription start site
* tes : transcription end site
* ensg : ENSEMBL gene ID
* gene : HGNC gene name
* z : TWAS z-score
* beta : TWAS effect size
* beta.se : standard error of TWAS effect size
