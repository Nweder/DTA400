```markdown
# DTA400 – Simulation Study

**Author:** Mohamad Nweder  
**University:** University West (Högskolan Väst)  
**Course:** DTA400 – Simulation Project  

## Overview
This project demonstrates how simulation can be used to analyze system behavior and performance.  
The results are presented in a short scientific article following the IEEE conference paper format.

## Objectives
- Apply simulation methodology to a chosen problem  
- Collect, analyze, and interpret simulation data  
- Present results and conclusions in IEEE format  

Det ser ut så där eftersom Markdown tolkar dina mappar och kommentarer (`# Python scripts ...`) som vanlig text på samma rad — inte som en kodblockstruktur.

För att få den att visas korrekt med mappar på egna rader, ändra det till ett kodblock med ``` runt hela strukturen:

```markdown
Det ser ut så där eftersom Markdown tolkar dina mappar och kommentarer (`# Python scripts ...`) som vanlig text på samma rad — inte som en kodblockstruktur.

För att få den att visas korrekt med mappar på egna rader, ändra det till ett kodblock med ``` runt hela strukturen:

```markdown
## Project Structure

📂 DTA400
┣ 📂 src/              # Python scripts (e.g., main.py)
┣ 📂 data/             # dataset1.npy, dataset2.npy
┣ 📂 results/          # output files, logs, tables
┣ 📂 plots/            # generated graphs and figures
┣ 📄 DTA400_SimulationStudy_MohamadNweder.docx  # report (export as PDF for submission)
┗ 📄 README.md

```

> Note: Export the final report as `report.pdf` before submission.
```

## Requirements
- Python 3.10+  
- Libraries: `numpy`, `pandas`, `matplotlib`, `simpy` (if used)

Install dependencies:
```bash
pip install numpy pandas matplotlib simpy
````

## How to Run

Execute the main simulation script:

```bash
python src/main.py
```

### Output

* Simulation data saved in `results/`
* Generated plots saved in `plots/`

## Reproducibility

* Input data: `data/dataset1.npy`, `data/dataset2.npy`
* Set random seed for consistency: `np.random.seed(42)`

## Report

* Written in IEEE format.
* Include figures from `plots/` and tables from `results/`.
* Export the final version to `report.pdf` before submission.
