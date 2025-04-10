# Welcome to Cluster-Aware Routines for Versatile Embedding (CARVE) 

🏂 CARVE stands for Cluster-Aware Routines for Versatile Embedding and provides a flexible toolbox of implementations of algorithms for tailored and effective application of cluster-aware embedding techniques. This code was developed by Dr. Amanda Buch and Dr. Logan Grosenick in the Grosenick lab at Weill Cornell Medicine.


CARVE is a novel framework that simultaneously performs joint clustering and embedding by combining standard embedding methods using a convex clustering penalty in a modular way.

Currently CARVE is composed of the following algorithms (Please cite Buch et al., AISTATS 2024):

Pathwise Clustered Matrix Factorization (PCMF). PCMF implements cluster-aware principal component analysis on a single-view dataset. 
Locally Linear Pathwise Clustered Matrix Factorization (LL-PCMF). LL-PCMF implements cluster-aware locally linear embedding on a single-view dataset. 
Pathwise Clustered Canonical Correlation Analysis (P3CA). P3CA implements cluster-aware canonical correlation analysis on two-view datasets. 

---

## Installation and Demo of AISTATS code

- Installing Conda environment using .yml file with Mamba:
mamba env create -f aistats_2024.yml

- Mosek licence: To run the code, you will need to get a free Mosek license from: and put it in /home/<user>/mosek/mosek.lic (follow directions at link). 

- If you encounter problems with 'admm_utils', you can rebuild the Cython files locally by first installing Cython:

```
conda install -c anaconda cython
```

...and then running:

```
python setup.py build_ext --inplace
```

- The jupyter notebook 'Example notebook for aistats 2024.ipynb' contains runnable examples. 

---

## License (MIT)
Copyright (c) 2022-present Logan Grosenick, Amanda M. Buch, & Conor Liston

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
## References

📄 Full Paper May 2024: Buch, Amanda M., Liston, Conor & Grosenick, Logan. Simple and Scalable Algorithms for Cluster-Aware Precision Medicine. Proceedings of The 27th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 238:136-144 https://proceedings.mlr.press/v238/m-buch24a/m-buch24a.pdf.

📄 NeurIPS Workshop Short Paper December 2023: Buch, Amanda M., Liston, Conor & Grosenick, Logan. (2023). Cluster-Aware Algorithms for AI-Enabled Precision Medicine. Neural Information Processing Systems Conference: LatinX in AI (LXAI) Research Workshop 2023, New Orleans, Louisiana. https://doi.org/10.52591/lxai2023121011

📄 arXiv Preprint November 2022: Buch, Amanda M., Liston, Conor & Grosenick, Logan. Simple and Scalable Algorithms for Cluster-Aware Precision Medicine. https://arxiv.org/abs/2211.16553

---

## Contact 

📧 Please reach out to Dr. Amanda Buch and Dr. Logan Grosenick with any questions.



