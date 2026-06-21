# Retail Choice & Location Model – Supporting Code for End of Studies Project Report

This repository contains the codebase supporting the final-year project (PFE) on **retail location analysis and consumer store choice modeling**.

The project combines spatial data processing, demand allocation modeling, Bayesian inference, and geospatial analytics to estimate how consumer expenditure is distributed across competing retail stores.

## Repository Contents

### Retail Choice Model
- Utility-based store choice model
- Expenditure allocation using a multinomial logit framework
- Calibration and scenario analysis

### Bayesian Inference (PyMC)
- Model estimation using PyMC
- Posterior analysis and parameter interpretation
- MCMC diagnostics and convergence checks

### Data Collection
- Web scraping scripts for retail store information
- Automated extraction and cleaning pipelines

### Spatial Preprocessing
- Construction of demand grids
- Geographic feature engineering
- Accessibility and distance calculations

### Traffic Analysis
- TomTom traffic data processing
- Travel time computation
- Accessibility indicators used in the utility model

### Satellite Imagery Segmentation
- SAM3-based segmentation workflows
- Extraction of building and spatial features
- Preprocessing of geospatial imagery

## Purpose

This repository serves as a technical companion to the PFE report. It contains the scripts, experiments, and data-processing pipelines used to build, estimate, and evaluate the retail choice and location model presented in the report.

## Technologies

- Python
- PyMC
- GeoPandas
- Pandas
- NumPy
- TomTom APIs
- SAM3
- Jupyter Notebooks

## Notes

This repository is intended for academic and research purposes. Some datasets and API-based resources are not included due to licensing and access restrictions.
