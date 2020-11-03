# MultipleTemplateMatching-KNIME
Implementation of object(s) detection with one or multiple templates in KNIME.   
Refer to the [wiki section](https://github.com/multi-template-matching/MultipleTemplateMatching-KNIME/wiki) for installation, video tutorial...

The second workflow (nearest template) allows to do classification with a set of templates.  
An image thus gets classified as the category of the top-score template, or as outlier if no score is above a user-defined threshold.  

# Installation
The latest workflow version can be downloaded from the [KNIME Hub](https://kni.me/w/9i0_HPPQlbNzW598).  
In addition to the workflow, a python environment with the Multi-Template-Matching package is necessary.  
`pip install Multi-Template-Matching` (case sensitive)  
and finally setup the python envirronment in KNIME.

<img src="https://github.com/multi-template-matching/MultipleTemplateMatching-KNIME/blob/master/workflow.svg" alt="Workflow" width="900" height="350"> 
