'''
PYTHON3
Template matching and maxima/hit detection (above/below the score threshold depending on the detection method)
Followed by Non-Maxima Supression based on the overlap between detected bbox

input_table_1 : Template
input_table_2 : Target Image (only one since we are looping out of python)

output_table_1 : Hits before NMS
output_table_2 : Hits after NMS
Store the detection result as a dictionnary {TemplateIdx:int, BBox:(x,y,width,height), Score:float} 

NB :
Peak local max do not return the peaks by order of height by default.
'''
from skimage.feature import peak_local_max
from scipy.signal	 import argrelmax, find_peaks
import numpy  as np
import pandas as pd
import cv2

# For NMS
import sys
sys.path.append(flow_variables['context.workflow.absolute-path']) # to have access to the NonMaximaSuppressionModule stored there
from NonMaximaSupression import NMS


# Recover parameters 
MatchMethod	   = flow_variables['MatchMethod_CV']
N			   = flow_variables['NumHit']      # Expected number of hit (different from the experimental number of hit found)
Thresh   	   = flow_variables['ScoreThresh'] # min correlation score to retain a hit
maxOverlap	   = flow_variables['maxOverlap']

# get Image and	 Name
Name  = input_table_2.index[0]
Image = input_table_2['Image'][0].array

# Convert image to 32-bit Gray
if Image.ndim == 3: # conversion from RGB to Gray
	Image =	 np.mean(Image, axis=0, dtype=np.float32)	

else: # we still convert to float 32 (needed for match template)
	Image = np.float32(Image)




# Do the template matching for each template (initial + flipped + rotated) for that image

# Get templates as images and convert to 32-bit for match template
ListTemplate = [np.float32(image.array) for image in input_table_1['Template']]

ListHit = [] # contains n dictionnaries : 1 per template with {'TemplateIdx'= (int),'BBox'=(x,y,width,height),'Score'=(float)}

# Loop over templates
for j in range(len(ListTemplate)):
	
	TemplateName = input_table_1['TemplateName'][j]
	print('\nSearch with template : ',TemplateName)
	
	# Get template
	Template	 = ListTemplate[j] 
	Height,Width = Template.shape 

	# Compute correlation map
	CorrMap = cv2.matchTemplate(Template,Image, method = MatchMethod) # correlation map


	# Get coordinates of the peaks in the correaltion map
	# IF depending on the shape of the correlation map
	if CorrMap.shape == (1,1): # Correlation map is a simple digit, when template size = image size
		print('Template size = Image size -> Correlation map is a single digit')
		
		if (MatchMethod==1 and CorrMap[0,0]<=minScore) or  (MatchMethod in [3,5] and CorrMap[0,0]>=minScore):
			Peaks = np.array([[0,0]])
		else:
			Peaks = []

	# use scipy findpeaks for the 1D cases (would allow to specify the relative threshold for the score directly here rather than in the NMS
	elif CorrMap.shape[0] == 1:	 # Template is as high as the image 
		print('Template is as high as the image, the correlation map is a 1D-array') 
		#Peaks = argrelmax(CorrMap[0], mode="wrap") # CorrMap[0] to have content of CorrMap as a proper line array 

		if MatchMethod==1:
			Peaks = find_peaks(-CorrMap[0], height=-Thresh) # find minima as maxima of inverted corr map
		
		elif MatchMethod in [3,5]:
			Peaks = find_peaks(CorrMap[0], height=Thresh) # find minima as maxima of inverted corr map
		
		Peaks = [[0,i] for i in Peaks[0]] # 0,i since one coordinate is fixed (the one for which Template = Image)
		

	
	elif CorrMap.shape[1] == 1: # Template is as wide as the image
		print('Template is as wide as the image, the correlation map is a 1D-array')
		#Peaks	= argrelmax(CorrMap, mode="wrap")
		if MatchMethod==1:
			Peaks = find_peaks(-CorrMap[:,0], height=-Thresh) # find minima as maxima of inverted corr map, height define the minimum height (threshold)
		
		elif MatchMethod in [3,5]:
			Peaks = find_peaks(CorrMap[:,0], height=Thresh)
		
		Peaks = [[i,0] for i in Peaks[0]]


	else: # Correlatin map is 2D
		# use threshold_abs to have something reproducible (if relative then case dependant)
		if MatchMethod==1:
			Peaks = peak_local_max(-CorrMap, threshold_abs=-Thresh, exclude_border=False).tolist() #BEWARE DO NOT RETURN IN ORDER OF MAXIMUM BY DEFAULT
		
		elif MatchMethod in [3,5]:
			Peaks = peak_local_max(CorrMap, threshold_abs=Thresh, exclude_border=False).tolist() #BEWARE DO NOT RETURN IN ORDER OF MAXIMUM BY DEFAULT

	print('Initially found',len(Peaks),'hit with this template')
	
	# Once every peak was detected for this given template
	# Create a dictionnary for each hit with {'TemplateName':, 'BBox': (x,y,Width, Height), 'Score':coeff}
	for peak in Peaks :
		coeff  = CorrMap[tuple(peak)]
		newHit = {'ImageName':Name, 'TemplateName':TemplateName, 'BBox': [int(peak[1]), int(peak[0]), Width, Height], 'Score':coeff}

		# append to list of potential hit before Non maxima suppression
		ListHit.append(newHit)


## Output table 1 : Hits before NMS
output_table_1 = pd.DataFrame(ListHit)

## NMS
if len(ListHit)>1: # More than one hit, so we need to do NMS

	print('Start NMS')

	if MatchMethod==1: # Difference : Best score = low values
		Hits_AfterNMS = NMS(ListHit, sortDescending=False, N=N, maxOverlap=maxOverlap) # remove scoreThreshold=Thres since done already at peak detection
	
	elif MatchMethod in [3,5]: # Correlation : Best score = high values
		Hits_AfterNMS = NMS(ListHit, sortDescending=True, N=N, maxOverlap=maxOverlap)
	
	# Generate output table
	output_table_2 = pd.DataFrame(Hits_AfterNMS)


else: # only one or 0 hit so no NMS to do
	output_table_2 = output_table_1