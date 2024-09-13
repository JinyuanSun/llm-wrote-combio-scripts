# llm-wrote-combio-scripts

Here I would like to test how good LLM can write scrpit for computational biology.

**1. SASA calculation**
At 2024.09.13, I conducted multiple rounds chat with the just released o1-preview model. In the end, it implemented the following script for calculating SASA of a protein, you can test it:

```bash
cd sasa
python sasa2.py 1pga.pdb 
```

You can also compare the results with the freesasa:
```bash
python eval.py 1pga_sasa.pdb 1pga_freesasa.pdb
```
Here is the results:
```
Number of common atoms: 436
Pearson correlation coefficient: 0.9473 (p-value: 9.5653e-217)
Spearman correlation coefficient: 0.9383 (p-value: 3.0193e-202)
```

Basically, the results are very close to the freesasa.

**reflection**:  
With multiple rounds of chat, the model finally understood the task and implemented the script. The script is very simple and easy to understand. The results are very close to the freesasa, which is a good sign. However, the model still needs to improve the efficiency of the script. The current script is very slow:
`python sasa2.py 1pga.pdb -d cpu  1.96s user 1.55s system 184% cpu 1.900 total`  

`freesasa 1pga.pdb --output=1pga_freesasa.pdb --format=pdb  0.02s user 0.00s system 83% cpu 0.028 total`
